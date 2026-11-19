# Gait Monitoring Dashboard — Stage 1 (Offline CSV Playback)

This is an offline Streamlit prototype for the Gait Estimation project. It loads a trained PyTorch checkpoint and a recorded IMU CSV, runs dense bilateral stance/swing inference, visualizes the results, and (when FSR-derived labels are present) compares predictions with those reference labels.

## 1. Environment setup

Activate the existing Conda environment:

```bash
conda activate gait-cnn
```

Install the dashboard-only packages without replacing your existing PyTorch installation:

```bash
python -m pip install -r requirements-dashboard.txt
```

The environment must already contain the packages used by the CNN: `torch`, `numpy`, and `pandas`.

## 2. Start the dashboard

From this folder, run:

```bash
streamlit run dashboard.py
```

Streamlit opens the dashboard in your browser (normally at `http://localhost:8501`).

## 3. Load the files

In the left sidebar, upload:

1. A `.pt` checkpoint saved by `train_imu_configurable.py` (for example, `fold_1_model.pt` or `final_model.pt`).
2. A recorded `.csv` containing the IMU channels required by that checkpoint.

The checkpoint itself determines the sensor configuration and channel order. For example, a LeftShank-only checkpoint requires the six `LeftShank_*` columns; a six-IMU checkpoint requires all 36 channels stored in its `features` list.

If the CSV has `time_s`, that column is used for the time axis. Otherwise, the dashboard uses the sampling-rate setting in the sidebar (default 148 Hz).

## 4. Expected CSV columns

The IMU feature columns must match the checkpoint exactly. The training script uses names such as:

- `LeftShank_acc_x`, `LeftShank_acc_y`, `LeftShank_acc_z`
- `LeftShank_gyr_x`, `LeftShank_gyr_y`, `LeftShank_gyr_z`

The full labeled recordings may also contain:

- `time_s`
- `L_stance`
- `R_stance`

Labels use `0 = swing` and `1 = stance`. FSR-derived labels are only used for offline comparison; they are not passed into the CNN.

## 5. What the dashboard computes

- Current left/right stance or swing prediction and softmax score
- A bilateral gait-state timeline
- Left/right stance probabilities
- Cadence estimated from predicted swing-to-stance transitions
- Mean stride time for each leg (time between successive predicted heel strikes)
- Left/right stance percentage and double-support percentage
- If FSR labels are present: dense sample-level accuracy and macro F1, plus one-to-one heel-strike event matching, detection rate, precision, and timing MAE
- Downloadable dense prediction CSV

A predicted heel strike is defined here as a swing-to-stance transition. Event matching uses the tolerance selected in the sidebar. The event metrics are diagnostic estimates and depend on the FSR labels, prediction quality, and matching tolerance.

## 6. Important evaluation notes

- This dashboard uses stride-1 (dense) inference so the timeline can show transitions. The training script's ordinary cross-validation metrics use stride-15 windows. Dense metrics displayed here are therefore not numerically identical to the training script's window-level CV metrics.
- The first `window - 1` samples cannot be predicted because the CNN needs a complete causal input window.
- Use a checkpoint trained for the sensor configuration represented by the uploaded CSV. The app checks that all required feature columns exist.
- A fold-specific checkpoint was trained without its corresponding held-out participant. For a fair offline test, upload the held-out participant's recording for that fold. Using a fold checkpoint on another participant is exploratory inference, not an independent validation result.
- Only load checkpoints you trust and that were produced by your own training code.
- This is a research prototype, not a clinically validated diagnostic system.

## Files

- `dashboard.py`: Streamlit user interface
- `gait_dashboard_core.py`: model loading, dense inference, and gait metrics
- `requirements-dashboard.txt`: dashboard-only dependencies
