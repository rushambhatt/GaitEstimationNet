from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gait_dashboard_core import add_binary_predictions, calculate_metrics, infer_recording


st.set_page_config(
    page_title="Gait Monitoring Dashboard",
    page_icon="🚶",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
      div[data-testid="stMetric"] {background: rgba(120, 120, 120, 0.07); padding: 12px 14px; border-radius: 12px;}
      .small-note {font-size: 0.85rem; opacity: 0.75;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Gait Monitoring Dashboard")
st.caption("Offline research prototype · bilateral stance/swing estimation from wearable IMUs")

with st.sidebar:
    st.header("Load a recording")
    checkpoint_file = st.file_uploader(
        "1. Trained CNN checkpoint (.pt)", type=["pt", "pth"], help="Use a checkpoint produced by train_imu_configurable.py."
    )
    csv_file = st.file_uploader(
        "2. Recorded IMU data (.csv)", type=["csv"], help="Upload a labeled or unlabeled recording with the model's required IMU columns."
    )
    st.divider()
    st.subheader("Inference settings")
    threshold = st.slider("Stance probability threshold", 0.05, 0.95, 0.50, 0.01)
    sampling_rate = st.number_input(
        "Sampling rate if CSV has no usable time_s (Hz)", min_value=1.0, max_value=2000.0, value=148.0, step=1.0
    )
    event_tolerance_ms = st.slider("Heel-strike matching tolerance (ms)", 20, 300, 100, 10)
    st.caption("The CSV's time_s column is used when available. Otherwise, the sampling-rate setting converts sample indices to seconds.")

if not checkpoint_file or not csv_file:
    st.info("Upload a trained .pt checkpoint and an IMU CSV recording in the sidebar to begin.")
    st.markdown("### What this prototype displays")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Bilateral gait states**\n\nLeft and right stance/swing predictions.")
    c2.markdown("**Gait timing**\n\nCadence, stride time, stance percentage, and double support.")
    c3.markdown("**Reference comparison**\n\nWhen FSR labels are present, compare predictions with reference events.")
    st.warning("This is an offline research prototype, not a clinically validated diagnostic system.")
    st.stop()

@st.cache_data(show_spinner="Running CNN inference on the recording...")
def cached_inference(csv_bytes: bytes, model_bytes: bytes, sampling_rate_hz: float):
    return infer_recording(csv_bytes, model_bytes, sampling_rate_hz=sampling_rate_hz)

try:
    raw_predictions, model_info = cached_inference(
        csv_file.getvalue(), checkpoint_file.getvalue(), float(sampling_rate)
    )
    predictions = add_binary_predictions(raw_predictions, threshold)
except Exception as exc:
    st.error(f"Could not run inference: {exc}")
    st.info("Check that the checkpoint and CSV belong to compatible sensor configurations and that the CSV contains every feature listed in the checkpoint.")
    st.stop()

metrics = calculate_metrics(predictions, event_tolerance_s=event_tolerance_ms / 1000.0)

# Model / recording summary
with st.expander("Model and recording details", expanded=False):
    a, b, c, d = st.columns(4)
    a.write(f"**Configuration**\n\n{model_info['mode']}")
    b.write(f"**Input channels**\n\n{model_info['n_channels']}")
    c.write(f"**Window length**\n\n{model_info['window']} samples")
    d.write(f"**Inference device**\n\n{model_info['device']}")
    st.write("**Sensors:** " + ", ".join(model_info["sensors"]))
    st.write("**Required channels:** " + ", ".join(model_info["features"]))
    st.write(f"**Recording rows:** {model_info['recording_rows']:,} · **Predicted rows:** {model_info['predicted_rows']:,}")
    st.write(f"**Time axis:** {model_info['time_source']}")
    if model_info["epochs"] != "unknown":
        st.write(f"**Checkpoint training epochs:** {model_info['epochs']}")

st.divider()
st.subheader("Current predicted gait state")
latest = predictions.iloc[-1]
left_state = "STANCE" if int(latest["L_pred"]) == 1 else "SWING"
right_state = "STANCE" if int(latest["R_pred"]) == 1 else "SWING"
left_conf = latest["L_prob_stance"] if int(latest["L_pred"]) == 1 else 1 - latest["L_prob_stance"]
right_conf = latest["R_prob_stance"] if int(latest["R_pred"]) == 1 else 1 - latest["R_prob_stance"]

state_cols = st.columns(2)
with state_cols[0]:
    st.metric("LEFT LEG", left_state, f"Predicted-state confidence: {left_conf:.1%}")
with state_cols[1]:
    st.metric("RIGHT LEG", right_state, f"Predicted-state confidence: {right_conf:.1%}")
st.caption(f"Latest prediction corresponds to t = {latest['time_s']:.3f} s. Confidence is the model's softmax score, not a clinically calibrated probability.")

st.subheader("Gait summary")
metric_cols = st.columns(4)
def show_metric(column, label, value, suffix=""):
    column.metric(label, "—" if value is None or not np.isfinite(value) else f"{value:.1f}{suffix}")

show_metric(metric_cols[0], "Cadence", metrics.get("cadence_spm"), " steps/min")
show_metric(metric_cols[1], "Left stride time", metrics.get("left_stride_time_s"), " s")
show_metric(metric_cols[2], "Right stride time", metrics.get("right_stride_time_s"), " s")
show_metric(metric_cols[3], "Double support", metrics.get("double_support_pct"), "%")

stance_cols = st.columns(2)
show_metric(stance_cols[0], "Left stance", metrics.get("left_stance_pct"), "% of predicted samples")
show_metric(stance_cols[1], "Right stance", metrics.get("right_stance_pct"), "% of predicted samples")

st.divider()
st.subheader("Bilateral gait-state timeline")
all_times = predictions["time_s"].to_numpy(dtype=float)
time_min, time_max = float(np.nanmin(all_times)), float(np.nanmax(all_times))
if time_max > time_min:
    if time_max - time_min > 0.2:
        display_range = st.slider(
            "Visible time interval (seconds)",
            min_value=time_min,
            max_value=time_max,
            value=(time_min, time_max),
            step=max((time_max - time_min) / 1000, 0.001),
        )
    else:
        display_range = (time_min, time_max)
else:
    display_range = (time_min, time_max)
view = predictions[(predictions["time_s"] >= display_range[0]) & (predictions["time_s"] <= display_range[1])]

fig = go.Figure()
for column, name, color, dash in [
    ("L_pred", "Left leg · CNN", "#16a34a", "solid"),
    ("R_pred", "Right leg · CNN", "#2563eb", "solid"),
]:
    fig.add_trace(
        go.Scatter(
            x=view["time_s"], y=view[column], mode="lines", name=name,
            line=dict(color=color, width=3, shape="hv", dash=dash),
            hovertemplate="Time: %{x:.3f} s<br>State: %{y}<extra>%{fullData.name}</extra>",
        )
    )
if "L_true" in view.columns and "R_true" in view.columns:
    for column, name, color in [
        ("L_true", "Left leg · FSR reference", "#86bfa0"),
        ("R_true", "Right leg · FSR reference", "#8bb6ed"),
    ]:
        ref = view[np.isfinite(pd.to_numeric(view[column], errors="coerce"))]
        fig.add_trace(
            go.Scatter(
                x=ref["time_s"], y=ref[column], mode="lines", name=name,
                line=dict(color=color, width=1.5, dash="dot", shape="hv"),
                opacity=0.8,
                hovertemplate="Time: %{x:.3f} s<br>Reference state: %{y}<extra>%{fullData.name}</extra>",
            )
        )
fig.update_layout(
    height=430, margin=dict(l=20, r=20, t=25, b=20),
    xaxis_title="Time (s)", yaxis=dict(title="State", tickmode="array", tickvals=[0, 1], ticktext=["Swing (0)", "Stance (1)"], range=[-0.2, 1.2]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Stance probability")
prob_fig = go.Figure()
prob_fig.add_trace(go.Scatter(x=view["time_s"], y=view["L_prob_stance"], mode="lines", name="Left stance probability", line=dict(color="#16a34a", width=2)))
prob_fig.add_trace(go.Scatter(x=view["time_s"], y=view["R_prob_stance"], mode="lines", name="Right stance probability", line=dict(color="#2563eb", width=2)))
prob_fig.add_hline(y=threshold, line_dash="dash", line_color="gray", annotation_text=f"Threshold {threshold:.2f}")
prob_fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Time (s)", yaxis_title="P(stance)", yaxis=dict(range=[0, 1]), hovermode="x unified")
st.plotly_chart(prob_fig, use_container_width=True)

if model_info["has_reference_labels"]:
    st.divider()
    st.subheader("Comparison with FSR reference labels")
    st.caption("FSR labels are used only for offline evaluation here; they are not supplied to the CNN as input. These are dense, stride-1 prediction metrics and may differ from the training script's stride-15 window metrics.")
    class_cols = st.columns(4)
    left_cls = metrics.get("left_classification", {})
    right_cls = metrics.get("right_classification", {})
    show_metric(class_cols[0], "Left accuracy", left_cls.get("accuracy") * 100 if left_cls.get("accuracy") is not None else None, "%")
    show_metric(class_cols[1], "Left macro F1", left_cls.get("macro_f1") * 100 if left_cls.get("macro_f1") is not None else None, "%")
    show_metric(class_cols[2], "Right accuracy", right_cls.get("accuracy") * 100 if right_cls.get("accuracy") is not None else None, "%")
    show_metric(class_cols[3], "Right macro F1", right_cls.get("macro_f1") * 100 if right_cls.get("macro_f1") is not None else None, "%")

    st.markdown("#### Heel-strike event comparison")
    st.caption(f"A predicted heel strike is a swing-to-stance transition. Events are matched one-to-one within ±{event_tolerance_ms} ms.")
    event_cols = st.columns(2)
    for col, leg, label in [(event_cols[0], "left", "Left leg"), (event_cols[1], "right", "Right leg")]:
        event = metrics.get(f"{leg}_events", {})
        with col:
            st.markdown(f"**{label}**")
            st.write(f"Predicted events: {event.get('predicted_events', 0)} · FSR events: {event.get('reference_events', 0)}")
            st.write(f"Detection rate (recall): {event.get('recall', 0):.1%}")
            st.write(f"Precision: {event.get('precision', 0):.1%} · Event F1: {event.get('f1', 0):.1%}")
            mae = event.get("timing_mae_ms")
            st.write(f"Heel-strike timing MAE: {mae:.1f} ms" if mae is not None else "Heel-strike timing MAE: not available (no matched events)")
else:
    st.info("This CSV does not contain both L_stance and R_stance columns. Predictions are available, but FSR comparison metrics cannot be calculated.")

st.divider()
st.subheader("Export predictions")
export_cols = ["row", "time_s", "L_prob_stance", "R_prob_stance", "L_pred", "R_pred"]
export_cols += [c for c in ["L_true", "R_true"] if c in predictions.columns]
export_csv = predictions[export_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    "Download dense predictions (.csv)",
    data=export_csv,
    file_name="gait_dashboard_predictions.csv",
    mime="text/csv",
)

st.caption("Research prototype only. Predictions and derived gait metrics are not clinically validated and should not be used for diagnosis or treatment decisions.")
