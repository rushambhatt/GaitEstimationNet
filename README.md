# Gait CNN — IMU configuration analysis (GPU-ready)

This package reproduces the **type** of analysis in the teammate's Excel workbooks: stance/swing accuracy by held-out participant and sensor configuration, separately for the left and right legs, with cross-participant mean and SD. It does **not** reproduce the teammate's model or measured values: your PyTorch CNN is trained independently.

## Windows / RTX 4000 Ada setup

In Anaconda Prompt (or a terminal with Conda):

```powershell
conda create -n gait-cnn python=3.11 pip -y
conda activate gait-cnn
# Install a CUDA-capable PyTorch build using the official selector at https://pytorch.org/get-started/locally/
# Example only, if the cu126 channel is supported by your driver/PyTorch release:
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install numpy pandas scikit-learn matplotlib jupyterlab ipykernel artifact-tool
python -m ipykernel install --user --name gait-cnn --display-name "Python (gait-cnn)"
python -c "import torch; print(torch.__version__); print('CUDA available:',torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

For other driver/toolkit situations, use the **current** command from the official PyTorch installer. `nvidia-smi` shows the GPU and driver; installing a system CUDA Toolkit is generally not required for PyTorch pip wheels.

## Directory layout

```text
gait_cnn_analysis/
  imu_config_analysis.py
  export_accuracy_workbook.py
  IMU_Configuration_Analysis.ipynb
  data/
    20261030_0_5mph_Rusham_labeled.csv
    20261110_1_5mph_Vishnu_labeled.csv
    20261112_1_0_Chethan_labeled.csv
    ...
```

Copy **all labeled CSV files** to `data/`, preserving `..._Name_labeled.csv`. The name immediately before `_labeled.csv` is used as the subject group. Missing speeds are fine. Recordings lacking selected IMU channels are skipped **only for that configuration**, with an audit in `recording_eligibility.csv`.

## Start small

```powershell
python imu_config_analysis.py --data "data/*_labeled.csv" --configs LeftShank RightFoot LeftShank+RightShank 6IMU 7IMU --epochs 100 --batch-size 32 --require-cuda --output results_main
python export_accuracy_workbook.py --results results_main
jupyter lab IMU_Configuration_Analysis.ipynb
```

To test all the configurations listed in the friend's accuracy workbook:

```powershell
python imu_config_analysis.py --data "data/*_labeled.csv" --workbook-configs --epochs 100 --batch-size 32 --require-cuda --output results_all
```

To compare **6 vs. 7 IMUs fairly**, restrict both analyses to participants with all seven IMUs. For example, if complete-sensor participants are Chethan and Vishnu:

```powershell
python imu_config_analysis.py --data "data/*_labeled.csv" --configs 6IMU 7IMU --comparison-subjects Chethan Vishnu --matched-trials --epochs 100 --require-cuda --output results_paired_6v7
```

**Important:** `--matched-trials` enforces exactly the same eligible input recordings across all requested configurations; it is strongly recommended for direct 6-vs-7 sensor comparisons. Without this flag, the script includes all available recordings for each configuration and differences can be confounded by different participant sets or speeds.

To keep a future final test participant untouched while evaluating the remaining development subjects:

```powershell
python imu_config_analysis.py --data "data/*_labeled.csv" --configs LeftShank 6IMU --test-subject SubjectF --epochs 100 --require-cuda --output development_only
```

This excludes SubjectF from the analysis but **does not yet train a final model or score SubjectF**. Save that step for AFTER model/sensor selection.

## Output files

- `recording_inventory.csv`: available sensors / known speeds per trial
- `recording_eligibility.csv`: per-config included and excluded recordings
- `held_out_participant_metrics.csv`: per-fold accuracy, balanced accuracy, F1, stance/swing recall
- `metrics_by_speed.csv`: per-subject and walking-speed results
- `configuration_summary.csv`: unweighted mean and SD **across people**
- `accuracy_table_both_legs.csv`: spreadsheet-style wide summary
- `imu_configuration_accuracy.xlsx`: Excel-style summary made by exporter
- `predictions/`: optional predicted windows (with `--save-predictions`) and training-loss JSON files

## Caveats

- These results are **participant-held-out cross-validation**, not independent final test results.
- All prediction targets use `L_stance` and `R_stance` (0=swing, 1=stance).
- Window label is the **final timestamp** of each 148-sample window; 15-sample stride for scored windows.
- The initial CNN weights are reset for each configuration, participant fold and seed; input channel order follows the selected configuration.
- Mean/std is calculated using **only the training windows**, with overlapping-sample weighting matching the original trainer, without allocating the entire window tensor. Small differences due to floating-point summation may remain.
- Several configuration comparisons may have different eligible participants and walking speeds. Do not interpret raw differences as solely caused by sensor count without matching datasets.
- This does **not** compute heel-strike MAE or continuous gait-phase RMSE. That requires event matching and phase estimation beyond stance/swing window accuracy.
- Model size and input order are otherwise preserved. Use the same epochs, batch size, learning rate and seeds to support comparisons.
- Avoid rerunning and adjusting based on a participant designated as the untouched final test participant.
