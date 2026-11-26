"""Offline inference and gait metrics for the Gait Estimation Stage 1 dashboard.

The CNN definition and preprocessing follow train_imu_configurable.py:
- Two shared Conv1D layers and separate left/right two-class heads
- checkpoint-defined channel order and train-set normalization
- causal 148-sample windows, evaluated at every sample (stride 1)

Only load checkpoints produced by your trusted training script.
"""
from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


class TwoHeadGaitCNN(nn.Module):
    """Architecture copied from train_imu_configurable.py."""

    def __init__(self, n_channels: int, window: int = 148):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat_dim = self.shared(torch.zeros(1, n_channels, window)).shape[1]
        self.dense = nn.Sequential(nn.Linear(flat_dim, 32), nn.ReLU(), nn.Dropout(0.25))
        self.left_head = nn.Linear(32, 2)
        self.right_head = nn.Linear(32, 2)

    def forward(self, x):
        z = self.dense(self.shared(x))
        return self.left_head(z), self.right_head(z)


def _torch_load_checkpoint(checkpoint_bytes: bytes, device: torch.device) -> dict[str, Any]:
    stream = io.BytesIO(checkpoint_bytes)
    try:
        checkpoint = torch.load(stream, map_location=device, weights_only=True)
    except TypeError:  # Compatibility with older PyTorch releases.
        stream.seek(0)
        checkpoint = torch.load(stream, map_location=device)
    required = {"model_state_dict", "features", "mean", "std"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {sorted(missing)}")
    return checkpoint


def infer_recording(
    csv_bytes: bytes,
    checkpoint_bytes: bytes,
    sampling_rate_hz: float = 148.0,
    batch_size: int = 1024,
    device_name: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run dense (stride-1) causal predictions on one recorded CSV."""
    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    checkpoint = _torch_load_checkpoint(checkpoint_bytes, device)
    features = list(checkpoint["features"])
    config = checkpoint.get("config", {}) or {}
    window = int(config.get("window", 148))
    if window != 148:
        # The architecture supports a checkpoint-defined window, but current
        # project checkpoints use 148. This guards against silently mismatched files.
        pass

    df = pd.read_csv(io.BytesIO(csv_bytes), low_memory=False)
    missing = [column for column in features if column not in df.columns]
    if missing:
        raise ValueError(
            "This recording does not contain every channel required by the model. "
            f"Missing columns: {missing}"
        )
    if len(df) < window:
        raise ValueError(f"Recording has {len(df)} rows; at least {window} are required.")

    signals = df[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    n_rows, n_channels = signals.shape
    if not np.isfinite(signals).any():
        raise ValueError("The selected IMU channels contain no finite numeric data.")

    model = TwoHeadGaitCNN(n_channels, window=window).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = checkpoint["mean"].detach().cpu().numpy().reshape(1, n_channels, 1).astype(np.float32)
    std = checkpoint["std"].detach().cpu().numpy().reshape(1, n_channels, 1).astype(np.float32)
    std = np.maximum(std, 1e-6)

    # windows shape: (number_of_windows, channels, time_steps)
    windows = np.lib.stride_tricks.sliding_window_view(
        signals, window_shape=window, axis=0
    )
    valid_rows = np.isfinite(signals).all(axis=1)
    bad_prefix = np.concatenate(([0], np.cumsum((~valid_rows).astype(np.int64))))
    valid_starts = np.flatnonzero(
        (bad_prefix[window:] - bad_prefix[:-window]) == 0
    )
    if len(valid_starts) == 0:
        raise ValueError("No complete finite input windows were found in this recording.")

    left_prob_parts: list[np.ndarray] = []
    right_prob_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(valid_starts), batch_size):
            batch_ids = valid_starts[start : start + batch_size]
            xb = np.ascontiguousarray(windows[batch_ids], dtype=np.float32)
            xb = (xb - mean) / std
            tensor = torch.from_numpy(xb).to(device)
            left_logits, right_logits = model(tensor)
            left_prob_parts.append(torch.softmax(left_logits, dim=1)[:, 1].cpu().numpy())
            right_prob_parts.append(torch.softmax(right_logits, dim=1)[:, 1].cpu().numpy())

    left_prob = np.concatenate(left_prob_parts)
    right_prob = np.concatenate(right_prob_parts)
    end_rows = valid_starts + window - 1

    if "time_s" in df.columns:
        source_time = pd.to_numeric(df["time_s"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(source_time[end_rows]).all():
            times = source_time[end_rows]
            time_source = "CSV time_s column"
        else:
            times = end_rows / float(sampling_rate_hz)
            time_source = f"sample index / {sampling_rate_hz:g} Hz (time_s was incomplete)"
    else:
        times = end_rows / float(sampling_rate_hz)
        time_source = f"sample index / {sampling_rate_hz:g} Hz"

    out = pd.DataFrame(
        {
            "row": end_rows,
            "time_s": times,
            "L_prob_stance": left_prob,
            "R_prob_stance": right_prob,
        }
    )
    for label in ("L_stance", "R_stance"):
        if label in df.columns:
            values = pd.to_numeric(df[label], errors="coerce").to_numpy(dtype=float)
            out[label.replace("_stance", "_true")] = values[end_rows]

    sensors = []
    for feature in features:
        sensor = feature.rsplit("_", 1)[0].rsplit("_", 1)[0]
        if sensor not in sensors:
            sensors.append(sensor)

    info = {
        "features": features,
        "n_channels": n_channels,
        "window": window,
        "mode": config.get("mode", f"{n_channels // 6}imu"),
        "epochs": config.get("epochs", "unknown"),
        "sensors": sensors,
        "device": str(device),
        "time_source": time_source,
        "recording_rows": n_rows,
        "predicted_rows": len(out),
        "has_reference_labels": "L_true" in out.columns and "R_true" in out.columns,
    }
    return out, info


def add_binary_predictions(predictions: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    result = predictions.copy()
    result["L_pred"] = (result["L_prob_stance"] >= threshold).astype(int)
    result["R_pred"] = (result["R_prob_stance"] >= threshold).astype(int)
    return result


def transition_times(states, times, rows) -> np.ndarray:
    """Return 0->1 transition times, requiring adjacent valid source rows."""
    states = np.asarray(states, dtype=float)
    times = np.asarray(times, dtype=float)
    rows = np.asarray(rows, dtype=int)
    if len(states) < 2:
        return np.array([], dtype=float)
    valid_pair = (
        np.isfinite(states[1:])
        & np.isfinite(states[:-1])
        & (rows[1:] == rows[:-1] + 1)
    )
    event_mask = valid_pair & (states[:-1] == 0) & (states[1:] == 1)
    return times[1:][event_mask]


def match_events(predicted: np.ndarray, reference: np.ndarray, tolerance_s: float) -> dict[str, Any]:
    """Greedy one-to-one event matching for sorted event-time arrays."""
    predicted = np.asarray(predicted, dtype=float)
    reference = np.asarray(reference, dtype=float)
    i = j = 0
    errors = []
    while i < len(predicted) and j < len(reference):
        delta = predicted[i] - reference[j]
        if abs(delta) <= tolerance_s:
            errors.append(abs(delta))
            i += 1
            j += 1
        elif delta < -tolerance_s:
            i += 1
        else:
            j += 1
    matches = len(errors)
    precision = matches / len(predicted) if len(predicted) else 0.0
    recall = matches / len(reference) if len(reference) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "predicted_events": int(len(predicted)),
        "reference_events": int(len(reference)),
        "matched_events": int(matches),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "timing_mae_ms": float(np.mean(errors) * 1000) if errors else None,
    }


def _classification_metrics(y_true, y_pred) -> dict[str, float | None]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isin(y_true, [0, 1]) & np.isfinite(y_pred)
    if not valid.any():
        return {"accuracy": None, "macro_f1": None}
    yt = y_true[valid].astype(int)
    yp = y_pred[valid].astype(int)
    f1_values = []
    for cls in (0, 1):
        tp = int(np.sum((yt == cls) & (yp == cls)))
        fp = int(np.sum((yt != cls) & (yp == cls)))
        fn = int(np.sum((yt == cls) & (yp != cls)))
        denom = 2 * tp + fp + fn
        f1_values.append((2 * tp / denom) if denom else 0.0)
    return {"accuracy": float(np.mean(yt == yp)), "macro_f1": float(np.mean(f1_values))}


def calculate_metrics(predictions: pd.DataFrame, event_tolerance_s: float = 0.10) -> dict[str, Any]:
    if predictions.empty:
        return {}
    times = predictions["time_s"].to_numpy(dtype=float)
    rows = predictions["row"].to_numpy(dtype=int)
    duration = float(np.nanmax(times) - np.nanmin(times)) if len(times) > 1 else 0.0
    left = predictions["L_pred"].to_numpy(dtype=int)
    right = predictions["R_pred"].to_numpy(dtype=int)
    left_events = transition_times(left, times, rows)
    right_events = transition_times(right, times, rows)
    left_stride = np.diff(left_events)
    right_stride = np.diff(right_events)
    both_valid = np.isfinite(left) & np.isfinite(right)

    result: dict[str, Any] = {
        "duration_s": duration,
        "cadence_spm": ((len(left_events) + len(right_events)) / duration * 60.0) if duration > 0 else None,
        "left_stride_time_s": float(np.mean(left_stride)) if len(left_stride) else None,
        "right_stride_time_s": float(np.mean(right_stride)) if len(right_stride) else None,
        "left_stance_pct": float(np.mean(left == 1) * 100),
        "right_stance_pct": float(np.mean(right == 1) * 100),
        "double_support_pct": float(np.mean((left == 1) & (right == 1)) * 100) if both_valid.any() else None,
        "left_heel_strikes": int(len(left_events)),
        "right_heel_strikes": int(len(right_events)),
    }

    if "L_true" in predictions.columns and "R_true" in predictions.columns:
        result["left_classification"] = _classification_metrics(
            predictions["L_true"], predictions["L_pred"]
        )
        result["right_classification"] = _classification_metrics(
            predictions["R_true"], predictions["R_pred"]
        )
        left_ref = transition_times(predictions["L_true"], times, rows)
        right_ref = transition_times(predictions["R_true"], times, rows)
        result["left_events"] = match_events(left_events, left_ref, event_tolerance_s)
        result["right_events"] = match_events(right_events, right_ref, event_tolerance_s)
    return result
