#!/usr/bin/env python3
"""Two-head gait-state CNN, translated from the supplied RapidMiner DL4J processes.

Outputs: left/right stance-versus-swing for each IMU window. 0=swing, 1=stance.
The model architecture is a close 1-D temporal equivalent, NOT an exact DL4J reproduction.
Use multiple independently recorded CSVs for meaningful generalization testing.
"""
import argparse
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SENSOR_NAMES = ["LeftShank", "RightShank", "LeftThigh", "RightThigh", "Torso", "RightFoot"]
AXES = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
MODE = "6imu"
FEATURES = [f"{sensor}_{axis}" for sensor in (SENSOR_NAMES if MODE == "6imu" else ["LeftShank"]) for axis in AXES]
WINDOW, STEP, SEED = 148, 15, 2001


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TwoHeadGaitCNN(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        # RapidMiner: Conv(32,k=7) -> MaxPool(k=2) -> Conv(64,k=5)
        #              -> Dense(32,ReLU) -> Dropout(.25) -> Softmax(2).
        # Here a time-series Conv1d implements those same temporal sizes.
        self.shared = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat_dim = self.shared(torch.zeros(1, n_channels, WINDOW)).shape[1]
        self.dense = nn.Sequential(nn.Linear(flat_dim, 32), nn.ReLU(), nn.Dropout(.25))
        self.left_head = nn.Linear(32, 2)
        self.right_head = nn.Linear(32, 2)

    def forward(self, x):
        z = self.dense(self.shared(x))
        return self.left_head(z), self.right_head(z)  # raw logits, not probabilities


def collect_windows(paths, split_mode, n_folds):
    windows, lefts, rights, groups, metadata = [], [], [], [], []
    recordings = []  # raw per-file samples and split bounds for dense held-out predictions
    if split_mode == "recordings" and len(paths) < n_folds:
        raise ValueError(f"Recording-level {n_folds}-fold CV requires >= {n_folds} independent recording CSVs. Found {len(paths)}.")
    for recording_idx, filename in enumerate(paths):
        df = pd.read_csv(filename, low_memory=False)
        needed = FEATURES + ["L_stance", "R_stance"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"{filename}: missing CSV columns: {missing}")
        signals = df[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        labels = df[["L_stance", "R_stance"]].apply(pd.to_numeric, errors="coerce").to_numpy()
        n = len(df)
        # Blocks are split BEFORE windowing, to prevent overlapping train/test windows.
        segments = [(0, n, f"recording_{recording_idx}")] if split_mode == "recordings" else [
            (int(a), int(b), f"block_{i}") for i, (a, b) in enumerate(zip(np.linspace(0,n,n_folds+1,dtype=int)[:-1],np.linspace(0,n,n_folds+1,dtype=int)[1:]))
        ]
        timestamps = (pd.to_numeric(df["time_s"], errors="coerce").to_numpy(dtype=float)
                      if "time_s" in df.columns else np.full(n, np.nan))
        recordings.append({"file": Path(filename).name, "signals": signals,
                           "labels": labels, "time_s": timestamps, "segments": segments})
        accepted = 0
        for a,b,group in segments:
            for start in range(a, b-WINDOW+1, STEP):
                end = start+WINDOW
                # Labels are taken at the END of the causal input window.
                target = labels[end-1]
                if not np.all(np.isin(target, [0,1])):
                    continue
                window = signals[start:end]
                if not np.isfinite(window).all():
                    continue
                windows.append(window.T.copy())
                lefts.append(int(target[0])); rights.append(int(target[1]))
                groups.append(group)
                timestamp = df["time_s"].iloc[end-1] if "time_s" in df.columns else np.nan
                metadata.append((Path(filename).name, int(end-1), timestamp))
                accepted += 1
        print(f"{Path(filename).name}: {n:,} rows -> {accepted:,} valid windows")
    if not windows:
        raise ValueError("No valid windows; check labels (0/1), feature columns and file length.")
    X=np.stack(windows).astype(np.float32)
    y_left=np.asarray(lefts,dtype=np.int64)
    y_right=np.asarray(rights,dtype=np.int64)
    return X,y_left,y_right,np.asarray(groups),metadata,recordings


def normalize_train_only(X, train_idx, test_idx):
    # Mean/std of each IMU axis over training windows only.
    train=X[train_idx]
    mean=train.mean(axis=(0,2),keepdims=True)
    std=train.std(axis=(0,2),keepdims=True)
    std=np.maximum(std,1e-6)
    return (train-mean)/std, (X[test_idx]-mean)/std, mean, std


def evaluate(model, X_test, yl, yr, batch_size, device):
    model.eval()
    lpred,rpred=[],[]
    loader=DataLoader(torch.from_numpy(X_test),batch_size=batch_size,shuffle=False)
    with torch.no_grad():
        for xb in loader:
            la,ra=model(xb.to(device))
            lpred.extend(la.argmax(1).cpu().numpy().tolist())
            rpred.extend(ra.argmax(1).cpu().numpy().tolist())
    return np.asarray(lpred),np.asarray(rpred)


def predict_dense(model, signals, a, b, mean, std, device, batch=2048):
    """Stride-1 predictions strictly INSIDE the held-out raw block [a,b).

    Does not materialize all windows or duplicate a huge normalized recording.
    Windows ending on invalid-label rows are predicted; invalid-label rows
    remain -1 in the exported truth. Nonfinite IMU windows are skipped.
    """
    if b - a < WINDOW:
        return np.empty(0, dtype=np.int64), np.empty(0), np.empty(0)
    samples = torch.from_numpy(np.ascontiguousarray(signals[a:b].T))  # (C, N)
    windows = samples.unfold(1, WINDOW, 1).permute(1, 0, 2)  # view (n, C, W)
    # Efficient window-validity mask from per-row nonfinite counts.
    valid_row = np.isfinite(signals[a:b]).all(axis=1)
    bad = (~valid_row).astype(np.int64)
    cumulative = np.concatenate(([0], np.cumsum(bad)))
    valid_start_indices = np.flatnonzero(cumulative[WINDOW:] == cumulative[:-WINDOW])
    if len(valid_start_indices) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0), np.empty(0)
    mu = torch.as_tensor(mean.reshape(1, -1, 1), dtype=torch.float32, device=device)
    sd = torch.as_tensor(std.reshape(1, -1, 1), dtype=torch.float32, device=device)
    probabilities_l, probabilities_r = [], []
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(valid_start_indices), batch):
            ids = torch.from_numpy(valid_start_indices[i:i+batch].astype(np.int64))
            xb = windows.index_select(0, ids).to(device=device, dtype=torch.float32)
            xb = (xb - mu) / sd
            left_logits, right_logits = model(xb)
            probabilities_l.append(torch.softmax(left_logits, dim=1)[:, 1].cpu().numpy())
            probabilities_r.append(torch.softmax(right_logits, dim=1)[:, 1].cpu().numpy())
    end_rows = a + valid_start_indices + WINDOW - 1
    return end_rows, np.concatenate(probabilities_l), np.concatenate(probabilities_r)


def run_training(data_paths, output_dir="results_6imu", epochs=50, batch_size=32,
                 learning_rate=0.001, n_folds=5, split_mode="auto", seed=SEED):
    set_seed(seed)
    paths=sorted({Path(p).resolve() for p in data_paths})
    if not paths:
        raise ValueError("No labeled CSV files were found. Pass --data with file paths or a glob pattern.")
    if split_mode=="auto":
        split_mode="recordings" if len(paths)>=n_folds else "blocks"
    if split_mode=="blocks":
        warnings.warn("Using contiguous blocks of each recording for cross-validation. "
                      "This avoids overlapping windows across folds but does NOT evaluate unseen subjects or independent trials. "
                      "Use >=5 independent recordings with --split-mode recordings for a stronger test.")
    X, yl, yr, groups, metadata, recordings=collect_windows(paths,split_mode,n_folds)
    print(f"Mode={MODE}; shape={X.shape}; CV={split_mode}; groups={np.unique(groups).tolist()}")
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training device:",device)
    if len(np.unique(groups)) < n_folds:
        raise ValueError(f"Only {len(np.unique(groups))} folds have valid labeled windows, but {n_folds} were requested. Supply longer/better labeled recordings or reduce --folds.")
    cv=GroupKFold(n_splits=n_folds)
    predictions=[]; fold_results=[]; dense_parts=[]; fold_mapping=[]
    for fold,(tr,te) in enumerate(cv.split(X, yl, groups=groups),1):
        test_groups = sorted(set(groups[te]))
        print(f"\nFold {fold}/{n_folds}: training={len(tr)}, testing={len(te)}, "
              f"held-out groups={test_groups}")
        fold_mapping.append({"fold": fold, "held_out_groups": test_groups})
        set_seed(seed+fold)
        Xtr,Xte,mean,std=normalize_train_only(X,tr,te)
        model=TwoHeadGaitCNN(len(FEATURES)).to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
        loss_fn=nn.CrossEntropyLoss()
        loader=DataLoader(TensorDataset(torch.from_numpy(Xtr),torch.from_numpy(yl[tr]),torch.from_numpy(yr[tr])),
                          batch_size=batch_size,shuffle=True)
        history=[]
        for epoch in range(1,epochs+1):
            model.train();running=0;seen=0
            for xb,lb,rb in loader:
                xb=xb.to(device);lb=lb.to(device);rb=rb.to(device)
                optimizer.zero_grad(set_to_none=True)
                lo,ro=model(xb)
                # Equal-weight multitask objective. Each head has two classes: swing/stance.
                loss=loss_fn(lo,lb)+loss_fn(ro,rb)
                loss.backward();optimizer.step()
                running+=loss.item()*len(xb);seen+=len(xb)
            epoch_loss=running/seen;history.append({"epoch":epoch,"training_loss":epoch_loss})
            if epoch==1 or epoch%5==0 or epoch==epochs:
                print(f"  Epoch {epoch:>3}/{epochs}: combined training loss {epoch_loss:.4f}")
        lp,rp=evaluate(model,Xte,yl[te],yr[te],batch_size,device)
        metrics={"fold":fold,"n_train":len(tr),"n_test":len(te),
                 "left_accuracy":float(accuracy_score(yl[te],lp)),
                 "right_accuracy":float(accuracy_score(yr[te],rp)),
                 "left_macro_f1":float(f1_score(yl[te],lp,average="macro",zero_division=0)),
                 "right_macro_f1":float(f1_score(yr[te],rp,average="macro",zero_division=0))}
        fold_results.append(metrics)
        print("Metrics:",metrics)
        (output/f"fold_{fold}_history.json").write_text(json.dumps(history,indent=2))
        torch.save({"model_state_dict":model.state_dict(),"features":FEATURES,
                    "mean":torch.from_numpy(mean.copy()),"std":torch.from_numpy(std.copy()),
                    "config":{"mode":MODE,"window":WINDOW,"step":STEP,"epochs":epochs}},output/f"fold_{fold}_model.pt")
        for j,global_idx in enumerate(te):
            file,row,t=metadata[global_idx]
            predictions.append({"file":file,"window_end_row":row,"time_s":t,"fold":fold,
                                "L_true":int(yl[global_idx]),"L_pred":int(lp[j]),
                                "R_true":int(yr[global_idx]),"R_pred":int(rp[j])})
        # Dense inference is a SECOND evaluation pass, not additional training.
        # This preserves the original stride-15 CV metrics and output format.
        for recording in recordings:
            for a, b, group in recording["segments"]:
                if group not in test_groups:
                    continue
                end_rows, left_prob, right_prob = predict_dense(
                    model, recording["signals"], a, b, mean, std, device)
                if len(end_rows) == 0:
                    continue
                targets = recording["labels"][end_rows]
                true_left = np.where(np.isin(targets[:, 0], [0, 1]), targets[:, 0], -1).astype(int)
                true_right = np.where(np.isin(targets[:, 1], [0, 1]), targets[:, 1], -1).astype(int)
                dense_parts.append(pd.DataFrame({
                    "file": recording["file"], "row": end_rows, "time_s": recording["time_s"][end_rows],
                    "fold": fold, "L_true": true_left, "R_true": true_right,
                    "L_prob_stance": left_prob, "R_prob_stance": right_prob,
                    "L_pred": (left_prob >= 0.5).astype(int),
                    "R_pred": (right_prob >= 0.5).astype(int),
                }))
        print(f"  Dense held-out evaluation finished for groups {test_groups}")
    dense_cols = ["file", "row", "time_s", "fold", "L_true", "R_true",
                  "L_prob_stance", "R_prob_stance", "L_pred", "R_pred"]
    dense_df = pd.concat(dense_parts, ignore_index=True) if dense_parts else pd.DataFrame(columns=dense_cols)
    dense_df = dense_df.sort_values(["file", "row"]).reset_index(drop=True)
    dense_df.to_csv(output / "dense_predictions.csv", index=False, columns=dense_cols)
    (output / "fold_group_mapping.json").write_text(json.dumps(fold_mapping, indent=2))
    print(f"Saved {len(dense_df):,} dense predictions to", output / "dense_predictions.csv")
    pdf=pd.DataFrame(predictions)
    pdf.to_csv(output/"cv_predictions.csv",index=False)
    pd.DataFrame(fold_results).to_csv(output/"fold_metrics.csv",index=False)
    summary={"mode":MODE,"n_channels":len(FEATURES),"number_of_windows":len(X),
             "epochs":epochs,"batch_size":batch_size,"learning_rate":learning_rate,
             "split_mode":split_mode,"n_folds":n_folds,"recording_files":[str(p) for p in paths],
             "overall_left_accuracy":float(accuracy_score(pdf.L_true,pdf.L_pred)),
             "overall_right_accuracy":float(accuracy_score(pdf.R_true,pdf.R_pred)),
             "left_confusion_matrix_0_swing_1_stance":confusion_matrix(pdf.L_true,pdf.L_pred,labels=[0,1]).tolist(),
             "right_confusion_matrix_0_swing_1_stance":confusion_matrix(pdf.R_true,pdf.R_pred,labels=[0,1]).tolist(),
             "folds":fold_results}
    (output/"summary.json").write_text(json.dumps(summary,indent=2))
    print("\nOVERALL Left and Right accuracy:",summary["overall_left_accuracy"],summary["overall_right_accuracy"])
    print("Left confusion matrix (rows=true, cols=pred; classes swing, stance):\n",summary["left_confusion_matrix_0_swing_1_stance"])
    print("Right confusion matrix (rows=true, cols=pred; classes swing, stance):\n",summary["right_confusion_matrix_0_swing_1_stance"])
    print("Saved metrics, predictions and models to",output.resolve())
    return summary


def main():
    parser=argparse.ArgumentParser(description="Train a two-head gait CNN from labeled IMU CSV files")
    parser.add_argument("--data",nargs="+",required=True,help="Labeled CSV paths (shell globs accepted)")
    parser.add_argument("--output",default="results_6imu")
    parser.add_argument("--epochs",type=int,default=50)
    parser.add_argument("--batch-size",type=int,default=32)
    parser.add_argument("--learning-rate",type=float,default=0.001)
    parser.add_argument("--folds",type=int,default=5)
    parser.add_argument("--split-mode",choices=["auto","recordings","blocks"],default="auto")
    args=parser.parse_args()
    import glob
    paths=[]
    for item in args.data:
        paths.extend(glob.glob(item))
    run_training(paths,args.output,args.epochs,args.batch_size,args.learning_rate,args.folds,args.split_mode)


if __name__=="__main__":
    main()