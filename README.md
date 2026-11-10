# Reproducible bilateral gait-state classification: 6-IMU vs. left-shank two-head CNN

**Project status:** Implementation prepared for 50-epoch experiments.  
**Implementation:** PyTorch on CPU or CUDA, standalone scripts or matching Jupyter notebooks.  
**Source baseline:** Four supplied Altair AI Studio / RapidMiner `.rmp` processes with separate left/right classifiers, 25 epochs.

## 1. Objective and experimental questions

The goal is to test whether a CNN can classify **left-leg and right-leg stance versus swing simultaneously**, using either all six IMUs or only the left-shank IMU. Both experiments use a **shared temporal CNN feature extractor** and two *distinct* two-class output heads:

- **Left output:** `L_stance` = 0 (swing) or 1 (stance).
- **Right output:** `R_stance` = 0 (swing) or 1 (stance).

Each head learns its own classifier from the shared features. This is **not** a four-class combined-state classifier, nor a continuous gait-cycle-percentage regressor. The model does not directly detect heel strikes; transition/event evaluation would require additional postprocessing and comparison with FSR-based reference events.

**Primary comparison:** effect of using 36 versus 6 IMU channels on left/right classification accuracy and macro-F1, with the architecture and training settings held constant except for input size. The experimental comparison is *within this PyTorch implementation*; the older RapidMiner results serve as historical reference, not directly controlled comparisons.

## 2. Files and directory structure

```
two_head_gait_cnn/
  README.md
  train_6imu_twohead.py
  train_6imu_twohead.ipynb
  train_left_shank_twohead.py
  train_left_shank_twohead.ipynb
  results_6imu_twohead_50epochs/              # created after execution
  results_left_shank_twohead_50epochs/       # created after execution
```

**Training inputs:** original synchronized and labeled gait recordings matching `*_labeled.csv`, for example:

```
/home/rusham/Downloads/gait_labels_20261030/
  20261030_0_5mph_labeled.csv
  20261030_1_0mph_labeled.csv
  20261030_1_5mph_labeled.csv
  20261030_2_0mph_labeled.csv
```

## 3. Data dictionary and labels

### 3.1 Required model targets

| Field | Type / values | Interpretation | Model use |
|---|---|---|---|
| `L_stance` | 0 / 1 / -1 | 0 = left swing; 1 = left stance; -1 = invalid/unlabeled | Left-head target |
| `R_stance` | 0 / 1 / -1 | 0 = right swing; 1 = right stance; -1 = invalid/unlabeled | Right-head target |

A window is retained only when **both labels at the window's final row** are exactly 0 or 1. A window with `-1`, NaN, or another label value at that target row is dropped. **Note:** the current implementation does *not* check whether every intermediate row within the window has valid labels; it checks only the target row.

### 3.2 Input channels

Each selected sensor contributes six features:

`<Sensor>_acc_x`, `<Sensor>_acc_y`, `<Sensor>_acc_z`, `<Sensor>_gyr_x`, `<Sensor>_gyr_y`, `<Sensor>_gyr_z`.

`acc_*` = accelerometer axis; `gyr_*` = gyroscope axis. Physical units depend on the export/calibration and have **not been verified** from the provided CSV metadata. Preserve consistent units and sensor orientation across recordings.

| Sensor as named in CSV | 6-IMU experiment | Left-shank experiment |
|---|:---:|:---:|
| `LeftShank` | Yes | Yes |
| `RightShank` | Yes | No |
| `LeftThigh` | Yes | No |
| `RightThigh` | Yes | No |
| `Torso` | Yes | No |
| `RightFoot` | Yes | No |
| **Input channels** | **36** | **6** |

The 6-IMU feature order is exactly the table order above, with `acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z` within each sensor. The 1-IMU experiment uses the left-shank six-channel subset.

### 3.3 Additional columns in the example labeled CSV

| Field(s) | Meaning / current use |
|---|---|
| `time_s`, `lsl_time` | Timestamp fields; `time_s` is exported with predictions when present but is not used as a feature. Confirm its time reference and sampling rate from acquisition metadata. |
| `phase_code` | Combined bilateral gait-state code; not used by the two-head CNN. Check its coding separately before treating it as ground truth. |
| `L_gait_pct`, `R_gait_pct` | Gait-cycle percentage annotations; **not** used as targets in this binary task. |
| `L_HS`, `R_HS` | Heel-strike event indicators; not used in current training or directly evaluated. |
| `L_TO`, `R_TO` | Toe-off event indicators; not used in current training or directly evaluated. |
| `fsr_L1_Heel`, `fsr_L1_Mid`, `fsr_L1_Toe`, `fsr_L2_Heel`, `fsr_L2_Mid`, `fsr_L2_Toe` | Foot-contact/FSR channels; not used as CNN features. Can support independent ground-truth derivation/validation if synchronized and verified. |

The six FSR column names in this example are left-foot-prefixed; do not assume bilateral right-foot FSR coverage without confirming the acquisition setup and annotation pipeline. Record how the stance labels were derived from contact signals before interpreting them as independent ground truth.

## 4. Preprocessing, segmentation and potential leakage

The scripts implement the following operations, in order:

1. Read each `*_labeled.csv` with pandas. Verify the exact 36 or 6 required feature names and both label columns.
2. Convert selected signals and labels to numeric; nonnumeric measurements become NaN.
3. **Determine cross-validation groups before making windows** (see §6).
4. Within each group, form **causal, sliding windows** of **148 rows** with start offsets **15 rows apart**. The window includes rows `start` through `start + 147`.
5. Assign both targets from the **final row (`start + 147`)**, so a prediction is associated with the latest observed sample. Keep only windows with valid 0/1 targets for *both* legs and entirely finite IMU features.
6. Build the tensor as `X.shape = (n_windows, n_channels, 148)` for PyTorch `Conv1d`.
7. Within **each cross-validation fold**, compute one mean and standard deviation per feature channel across **training windows and all their timestamps**. Normalize train and held-out windows using **only those training-fold statistics**: `z = (x - mean_train) / max(std_train, 1e-6)`.
8. Save the training-fold normalization mean/std inside each fold checkpoint so prediction/inference can use the same transformation.

Important methodological details:

- Input data must already be synchronized, labeled and reasonably cleaned. The scripts **do not** resample, filter, synchronize clocks, derive FSR labels, repair missing sensors, or remove motion artifacts beyond excluding nonfinite windows.
- A **148-row** window is not automatically a fixed duration in seconds. Duration = `148 / sampling_rate_Hz`; obtain and document the actual sampling rate for every recording.
- Training windows overlap heavily **within a fold** (stride 15). This is expected, but test windows must not share raw rows with training windows. Splitting groups **before** generating windows enforces this at group boundaries.
- Train-fold standardization avoids normalization leakage, but because raw signals from an individual/trial may be highly correlated, block-based cross-validation is not a test of transfer to unseen subjects.
- Reusing the exact same data, labels and split mode is necessary to compare the two sensor layouts fairly.

## 5. Model architecture: close translation of RapidMiner baseline

### 5.1 Shared feature extractor and separate classifiers

```
Input: [batch, 36 or 6 channels, 148 time samples]
    ↓
Conv1d(input_channels → 32, kernel_size=7, stride=1) + ReLU
    ↓
MaxPool1d(kernel_size=2, stride=2)
    ↓
Conv1d(32 → 64, kernel_size=5, stride=1) + ReLU
    ↓
Flatten
    ↓
Dense(32) + ReLU
    ↓
Dropout(p=0.25)
    ├── Left:  Dense(2) → [swing logit, stance logit]
    └── Right: Dense(2) → [swing logit, stance logit]
```

The code returns **raw logits** for each head. `CrossEntropyLoss` applies the correct log-softmax internally; `argmax` selects class 0 or 1. During training the two classification losses are summed with equal weights:

`loss_total = CrossEntropyLoss(left_logits, L_stance) + CrossEntropyLoss(right_logits, R_stance)`.

| Element | Original AI Studio baseline | This Python implementation |
|---|---|---|
| Feature model | DL4J convolutional/flattened sequential CNN | 1D temporal PyTorch convolution, close analogue |
| First convolution | 32 filters, kernel 7 | 32 filters, kernel 7, stride 1 |
| Max pooling | Kernel 2, stride 2 | Kernel 2, stride 2 |
| Second convolution | 64 filters, kernel 5 | 64 filters, kernel 5, stride 1 |
| Fully connected | 32 units, ReLU | 32 units, ReLU |
| Dropout | 0.25 | 0.25 |
| Output | One 2-class classifier per process | **Two independent 2-logit heads in one model** |
| Sensor inputs | Separate 6-IMU / shank process variants | Same 36-channel / 6-channel variants |

This is **not a byte-for-byte DL4J implementation**, and layer/default differences can affect learned parameters and accuracy. The shared dense layer's input size is determined automatically from the convolution output and fixed 148-row window. The two-head network differs deliberately from the four **separately trained single-head** RapidMiner models.

### 5.2 Training hyperparameters

| Parameter | Setting |
|---|---|
| Epochs per fold | **50** (override with `--epochs`) |
| Batch size | **32** |
| Optimizer | **Adam** |
| Learning rate | **0.001** |
| Loss | Equal-weight sum of two cross-entropy losses |
| Number of folds | **5** |
| Seed | `2001` initially; reinitialized at `2001 + fold_number` for each fold |
| Training order | Shuffled batches each epoch |
| Train/validation/test | Grouped train + held-out fold; **no separate tuning/early-stopping validation set** |
| Early stopping / scheduler | **Not implemented**; trains for all requested epochs |
| Device | CUDA when available, otherwise **CPU** |
| Class weighting | None; inspect class imbalance separately |

The original RapidMiner runs used **25 epochs**. A 50-epoch five-fold execution trains **five separate models for 50 epochs each**, i.e. 250 fold-epochs per sensor configuration (not a single 250-epoch model).

## 6. Five-fold cross-validation: EXACT behavior

The scripts support `--split-mode auto`, `recordings`, or `blocks` (default `auto`). Group assignment occurs **before windowing**, and `sklearn.model_selection.GroupKFold(n_splits=5)` ensures each group's windows occur entirely in either training or held-out testing for a fold.

**A. Recording-level grouping — preferred when there are at least five independent CSVs**

- `--split-mode recordings` requires **at least five** input CSV files.
- Each CSV gets its own group ID. On each fold, whole recordings are held out, though folds can hold out more than one recording when the number of files exceeds five.
- This evaluates **held-out recording generalization**, not necessarily held-out participant generalization. If multiple CSVs belong to the same person, person-level separation is *not* guaranteed. Use participant-aware grouping if independent-subject evaluation is required.

**B. Contiguous-block grouping — automatically used with fewer than five CSVs**

- With one to four files, `auto` switches to `blocks` and emits a warning.
- Each recording is split into five consecutive **raw-row blocks before windowing**. Groups are identified by block index `block_0` through `block_4`, **shared across files**. A fold holds out the corresponding block index across all recordings. Boundary windows are not allowed to cross block edges.
- This prevents exact train/test overlap in source rows, but **neighboring blocks can remain temporally correlated**. It does **not** establish performance on independent trials, speeds, or participants.
- With the four example speed files, default `auto` still uses **blocks**, *not* a four-file leave-one-recording-out evaluation. Do not describe it as recording-held-out CV.

**C. Relationship to RapidMiner's “5fold” processes**

The original processes used batch-based splitting; exact underlying folds and preprocessing may not match the Python `GroupKFold` procedure. Therefore, compare the two Python sensor configurations **using their common fold design**, and treat RapidMiner's results as historical baselines only.

**For a publication-quality evaluation:** split complete trials by participant first, and then split train/validation/test without cross-participant leakage. Fold selection, normalizer fitting and any hyperparameter tuning must remain inside the training/development portion.

## 7. Installation and CPU operation (Ubuntu)

Create and activate a Conda environment:

```bash
conda create -n gait_cnn python=3.11 -y
conda activate gait_cnn
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas scikit-learn matplotlib jupyterlab notebook ipykernel
python -m ipykernel install --user --name gait_cnn --display-name "Python (Gait CNN)"
```

`matplotlib` is optional for custom plots. The scripts themselves require `torch`, `numpy`, `pandas`, and `scikit-learn`; Jupyter packages are needed only for notebooks. For GPU training, follow PyTorch's appropriate platform installation instructions instead of the CPU wheel command.

Check the runtime:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import numpy,pandas,sklearn; print(numpy.__version__,pandas.__version__,sklearn.__version__)"
```

To **record the exact reproducibility environment after installation**:

```bash
python --version > python_version.txt
pip freeze > requirements_frozen.txt
conda env export --from-history > conda_environment_history.yml
```

CPU execution is supported automatically and requires no code changes. Runtime and memory demand scale with number of windows, fold count and 50 training epochs. The current code constructs all eligible windows in memory, which can be a limitation on modest-RAM systems.

## 8. Run the two experiments

From the extracted project directory:

```bash
# Experiment E1: all six IMUs, outputs both left and right state predictions
python train_6imu_twohead.py \
  --data '/home/rusham/Downloads/gait_labels_20261030/*_labeled.csv' \
  --epochs 50 --batch-size 32 --learning-rate 0.001 \
  --folds 5 --split-mode auto \
  --output results_6imu_twohead_50epochs

# Experiment E2: left-shank IMU only, outputs both left and right state predictions
python train_left_shank_twohead.py \
  --data '/home/rusham/Downloads/gait_labels_20261030/*_labeled.csv' \
  --epochs 50 --batch-size 32 --learning-rate 0.001 \
  --folds 5 --split-mode auto \
  --output results_left_shank_twohead_50epochs
```

Quotes around the wildcard allow the script to expand matching CSVs itself. For stricter evaluation with **at least five independent recordings**, explicitly set `--split-mode recordings` on *both* commands.

**Notebook alternative:** open `train_6imu_twohead.ipynb` or `train_left_shank_twohead.ipynb`, select the `Python (Gait CNN)` kernel, edit `DATA_DIR` in the configuration cell, and choose **Run All**. Notebook configurations default to `epochs=50`, `batch_size=32`, `learning_rate=0.001`, `n_folds=5`, and `split_mode="auto"`.

**Pre-run checklist:** confirm file count; feature channel mapping; sensor calibration/units; stance label convention; whether the folds are blocks or entire recordings; file hashes and package versions; free disk space and RAM; same CSVs and fold settings for E1 and E2.

## 9. Saved outputs and how to interpret them

Each run creates its selected result folder:

| File | Contents |
|---|---|
| `summary.json` | Sensor mode/channel count; total valid windows; epochs, learning rate, batch size, split mode, fold count, input paths; pooled overall left/right accuracy; left/right confusion matrices; per-fold metrics. |
| `fold_metrics.csv` | For each fold: number of training/test windows, left/right accuracy and left/right **macro-F1**. |
| `cv_predictions.csv` | One out-of-fold prediction per retained window, with `file`, `window_end_row`, `time_s`, `fold`, `L_true`, `L_pred`, `R_true`, `R_pred`. |
| `fold_1_model.pt` ... `fold_5_model.pt` | Independent PyTorch fold-model state dictionaries, selected feature names, training-fold normalizer means/stds and a subset of configuration settings. |
| `fold_1_history.json` ... `fold_5_history.json` | Training loss for each epoch; there is **no validation-loss curve** in the current implementation. |

`summary.json` overall accuracy pools all **out-of-fold windows**. It is not necessarily equal to the **unweighted average** of five fold accuracies, especially when folds have different numbers of valid windows. For paper/report comparisons, report per-fold mean and standard deviation and clarify whether aggregate scores are window-weighted.

The confusion matrix ordering is **rows = true label, columns = predicted label**; labels are `[0 = swing, 1 = stance]`.

The outputs do **not** currently provide probabilities, separate precision/recall reports, event-timing errors, participant-level metrics, or a single final all-data-trained deployment checkpoint. Do not treat a saved CV fold checkpoint as a separately trained production model.

## 10. Experiment register and results

### 10.1 Planned controlled two-head experiments (50 epochs)

| ID | Input | Channels | Outputs | Epochs | CV | Status | Left accuracy | Right accuracy | Left macro-F1 | Right macro-F1 |
|---|---|---:|---|---:|---|---|---|---|---|---|
| **E1** | Six IMUs | 36 | L/R stance-swing | 50 | 5-fold, mode to record | **Not yet reported** | TBD | TBD | TBD | TBD |
| **E2** | Left-shank IMU | 6 | L/R stance-swing | 50 | Same folds as E1 | **Not yet reported** | TBD | TBD | TBD | TBD |

When runs finish, extract accuracy from each `summary.json` (`overall_left_accuracy`, `overall_right_accuracy`) and macro-F1 distributions from `fold_metrics.csv`. Preserve the result folders intact. Record run date, host CPU, training time, package versions, CSV hash list and actual split mode alongside the metrics.

### 10.2 Historical RapidMiner results (different models and validation)

The previously supplied 25-epoch figure screenshots were summarized in conversation as:

| Historical configuration | Target | Reported classification accuracy | Source/qualifier |
|---|---|---:|---|
| Six IMUs, separate CNN | Left stance/swing | 97.31% ± 0.92% | 25-epoch RapidMiner screenshot; fold variability as displayed |
| Six IMUs, separate CNN | Right stance/swing | 96.29% ± 2.48% | Same |
| Left-shank IMU, separate CNN | Left stance/swing | 97.13% ± 0.96% | Same |
| Left-shank IMU, separate CNN | Right stance/swing | 95.70% ± 1.12% | Same |

**Do not claim that E1/E2 outperform or underperform these baselines solely from raw accuracy differences**: the two-head architecture, epoch count, DL4J-to-PyTorch translation, and fold construction differ. These historical figures are retained as contextual records, not as directly matched ablations. Recheck screenshots or original logs if a published result requires more precision.

### 10.3 Run record template (copy one per execution)

```text
Run ID:
Date / time / time zone:
Git commit or script SHA256:
Experiment E1/E2:
CSV files and SHA256 hashes:
Participant IDs / session IDs / trial IDs:
Speed / recording condition(s):
Sensor hardware and physical placement verified?:
Sensor units, sampling rate and synchronization method:
Input CSV row counts / valid window counts:
Split mode (recordings or blocks) / actual held-out groups per fold:
Seed, epochs, batch size, learning rate:
Python, PyTorch, pandas, numpy, scikit-learn versions:
CPU/GPU, RAM, execution duration:
Left accuracy pooled / fold mean ± SD:
Right accuracy pooled / fold mean ± SD:
Left macro-F1 mean ± SD:
Right macro-F1 mean ± SD:
Confusion matrices and false transition patterns:
Notes: failures, preprocessing deviations, anomalies, label uncertainties:
Paths to model checkpoints, predictions and summary.json:
```

For data integrity, you can record file hashes with:

```bash
sha256sum /home/rusham/Downloads/gait_labels_20261030/*_labeled.csv > input_sha256.txt
sha256sum train_6imu_twohead.py train_left_shank_twohead.py > code_sha256.txt
```

## 11. Limitations and next steps

1. **Generalizability:** Five sequential blocks from four recordings are *not* five independent trials or participants. Nearby samples and shared subjects can yield optimistic results. Preferred evaluation: subject-held-out or independent-trial splits.
2. **Data availability:** Only the supplied 2.0-mph labeled CSV was inspected; do not assume other speed files have equivalent completeness, sensor mapping, sampling or class balance.
3. **Reference labels:** `L_stance` and `R_stance` are assumed correct; labeling procedures, FSR thresholds and synchronization must be audited. Any `-1` at the window endpoint is excluded; internal invalid intervals are not additionally filtered.
4. **Model comparison:** The original RapidMiner models were four single-head CNNs trained for 25 epochs; these are **two multitask two-head CNNs trained for 50 epochs**. They are similar but not architecturally or numerically identical to DL4J.
5. **Epoch count:** More epochs can overfit. There is no early stopping or untouched validation set for choosing 50 epochs, and loss histories contain only training losses. Hyperparameter selection should use nested/development data, not the held-out test folds.
6. **Metrics:** Accuracy and macro-F1 may obscure minority phases or transitional timing mistakes. Report per-leg confusion matrices, stance/swing class balance, per-condition scores, and preferably transition detection F1 plus heel-strike and toe-off timing MAE against validated FSR event labels.
7. **Physical realism:** The left-shank-only model estimates the contralateral right-leg state from indirect movement cues; performance may fall under different speeds, terrain, pathologies, or participants. Avoid assuming robust bilateral inference without external testing.
8. **Duration and streaming:** Inputs are fixed row-count windows and predictions are assigned to window ends. Sampling rate, time alignment, stride-related update latency and real-time buffering must be documented before claiming streaming deployment.
9. **Reproducibility:** Random seeds reduce variability but are not a guarantee of bitwise-deterministic CPU/GPU results across devices, PyTorch versions, or parallel backends. Save versions, data hashes and actual held-out groups.
10. **Memory and evaluation design:** This implementation materializes all windows in memory. For large datasets, use a window-index Dataset or streaming loader. Avoid using the same CV test folds both to select design choices and to report final generalization performance.

# Windows Setup Guide – Two-Head Gait Phase CNN

## 1. Overview

This guide explains how to set up, train, and evaluate the two-head gait-phase CNN models on a Windows computer using Python, Conda, and Jupyter Notebook.

The project includes two sensor configurations:

1. **6-IMU CNN:** Uses all six IMU sensors (36 input channels).
2. **1-IMU CNN:** Uses only the left-shank IMU (6 input channels).

Both models use a shared CNN feature extractor with two output heads:

- **Left-leg head:** Predicts swing (0) or stance (1).
- **Right-leg head:** Predicts swing (0) or stance (1).

The models are trained for 50 epochs using five-fold cross-validation.

## 2. Prerequisites

Before beginning, install:

- Anaconda or Miniconda
- Python 3.11 (installed through Conda)
- Jupyter Notebook or JupyterLab
- PyTorch and the required Python packages

An NVIDIA GPU is not required. Both CNN models can be trained entirely on a CPU.

## 3. Create the Conda Environment

Open **Anaconda Prompt** from the Windows Start menu.

Create an environment named `gait_cnn`:

```bash
conda create -n gait_cnn python=3.11 -y
```

Activate the environment:

```bash
conda activate gait_cnn
```

Install the CPU-compatible version of PyTorch:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Install the remaining dependencies:

```bash
python -m pip install numpy pandas scikit-learn matplotlib seaborn jupyterlab notebook ipykernel tqdm
```

Register the Conda environment with Jupyter:

```bash
python -m ipykernel install --user --name gait_cnn --display-name "Python (Gait CNN)"
```

Verify the installation:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If CPU training is being used, `torch.cuda.is_available()` should return `False`.

## 4. Organize the Project Files

Create a project directory containing the Python scripts, notebooks, and labeled CSV recordings.

Example directory structure:

```text
C:\Users\YOUR_USERNAME\Documents\Gait_CNN\
│
├── train_6imu_twohead.py
├── train_left_shank_twohead.py
├── train_6imu_twohead.ipynb
├── train_left_shank_twohead.ipynb
│
├── gait_labels_20261030\
│   ├── 20261030_0_5mph_labeled.csv
│   ├── 20261030_1_0mph_labeled.csv
│   ├── 20261030_1_5mph_labeled.csv
│   └── 20261030_2_0mph_labeled.csv
│
├── results_6imu\
└── results_left_shank\
```

The results folders will be created when the models execute.

The input CSV files must contain the original synchronized IMU measurements and ground-truth labels.

Previously generated prediction files, such as `Left_6IMU_25epochs_CV_predictions.csv`, should not be used as training inputs.

## 5. Configure the Dataset Paths

Open the Python notebook or script corresponding to the desired sensor configuration.

For Jupyter notebooks, locate the dataset configuration section and update the directory path:

```python
from pathlib import Path

DATA_DIR = Path(
    r"C:\Users\YOUR_USERNAME\Documents\Gait_CNN\gait_labels_20261030"
)

DATA_FILES = sorted(DATA_DIR.glob("*_labeled.csv"))

print("Found", len(DATA_FILES), "recordings")
for file in DATA_FILES:
    print(file.name)
```

Replace `YOUR_USERNAME` with your Windows account username.

The `r` before the path string prevents Windows backslashes from being interpreted as escape characters.

Confirm that all four recordings are detected before starting training.

## 6. Input Variables and Labels

Each CNN processes fixed-length windows of IMU measurements.

### Input variables

| Configuration | Sensors | Input channels |
|---|---|---|
| 6 IMUs | All six IMUs | 36 |
| 1 IMU | Left-shank IMU | 6 |

Each IMU contributes six measurements:

- Acceleration X, Y, Z
- Angular velocity X, Y, Z

Each input window contains 148 timestamps, with a window step of 15 samples.

### Output labels

The CNN predicts two separate binary labels:

| Label | 0 | 1 |
|---|---|---|
| `L_stance` | Left swing | Left stance |
| `R_stance` | Right swing | Right stance |

Invalid or unlabeled states (`-1`) are excluded from supervised training.

The two heads are trained simultaneously using the corresponding left- and right-leg ground-truth labels.

## 7. CNN Architecture and Training Configuration

Both experiments use the same baseline CNN feature-extraction architecture, adapted from the original Altair AI Studio models.

| Parameter | Configuration |
|---|---|
| First convolution | 32 filters, kernel size 7 |
| Max pooling | Kernel size 2, stride 2 |
| Second convolution | 64 filters, kernel size 5 |
| Fully connected layer | 32 neurons, ReLU |
| Dropout | 0.25 |
| Output heads | Two independent 2-class classifiers |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 32 |
| Epochs | 50 |
| Validation | 5 folds |
| Input window | 148 samples |
| Window step | 15 samples |

The shared CNN learns gait-related features from the IMU data. Each output head then predicts the stance/swing state of its assigned leg.

The total training loss is calculated by adding the classification losses from both heads:

\[
L_{\text{total}}=L_{\text{left}}+L_{\text{right}}
\]

Both networks preserve the same architecture except for the number of input channels.

## 8. Run the Models Using Jupyter Notebook

Open Anaconda Prompt and activate the environment:

```bash
conda activate gait_cnn
```

Navigate to the project directory:

```bash
cd "C:\Users\YOUR_USERNAME\Documents\Gait_CNN"
```

Launch JupyterLab:

```bash
jupyter lab
```

Open one of the notebooks:

- `train_6imu_twohead.ipynb`
- `train_left_shank_twohead.ipynb`

Make sure the selected kernel is **Python (Gait CNN)**.

Verify that the dataset directory is correct, then select **Run → Run All Cells**.

The notebook will perform preprocessing, create the training and testing folds, train the CNN, evaluate predictions, and save the results.

## 9. Run the Models Directly from Anaconda Prompt

Alternatively, train without opening Jupyter.

Activate the environment and navigate to the project directory:

```bash
conda activate gait_cnn
cd "C:\Users\YOUR_USERNAME\Documents\Gait_CNN"
```

### Experiment 1: Six-IMU CNN

```bash
python train_6imu_twohead.py --data "gait_labels_20261030\*_labeled.csv" --epochs 50 --output results_6imu
```

### Experiment 2: Single Left-Shank CNN

```bash
python train_left_shank_twohead.py --data "gait_labels_20261030\*_labeled.csv" --epochs 50 --output results_left_shank
```

Both commands use the same four labeled datasets, but the scripts select different IMU input channels.

Running both experiments separately ensures that their results are saved to different folders.

## 10. Five-Fold Cross-Validation

Both models use five-fold cross-validation.

For each fold:

1. The available data are divided into training and testing groups.
2. Normalization parameters are calculated using training data only.
3. The CNN is trained for 50 epochs.
4. Predictions are generated for the held-out test group.
5. Left- and right-leg classification metrics are calculated.
6. Results are saved for comparison.

The provided scripts use block-based splitting when there are fewer than five independent recordings.

For the current four-recording dataset, this means dividing the recordings into contiguous blocks rather than holding out entire recordings.

This approach reduces direct overlap between training and testing windows, but adjacent blocks may still contain correlated gait patterns. Therefore, block-based cross-validation does not establish generalization to new participants or entirely independent walking trials.

## 11. Saved Results

Each experiment saves its outputs in the specified results directory.

Important files include:

| Output | Description |
|---|---|
| `summary.json` | Overall metrics and experiment configuration |
| `cv_predictions.csv` | Ground-truth and predicted stance/swing labels |
| `fold_metrics.csv` | Accuracy and macro F1 for each fold |
| Model checkpoints | Trained model parameters for each fold |
| Training histories | Training loss across epochs |

The prediction CSV contains separate predictions for both legs:

- `L_true`: Actual left-leg label
- `L_pred`: Predicted left-leg label
- `R_true`: Actual right-leg label
- `R_pred`: Predicted right-leg label

### Example 50-epoch results

The following results were obtained using the four walking-speed recordings and five-fold block-based cross-validation.

| Metric | 6 IMUs | 1 left-shank IMU |
|---|---:|---:|
| Number of windows | 4,468 | 4,468 |
| Left-leg accuracy | 97.40% | 97.54% |
| Right-leg accuracy | 97.07% | 96.51% |
| Input channels | 36 | 6 |
| Epochs | 50 | 50 |

These results suggest that a single left-shank IMU can classify both legs' stance/swing states with accuracy approaching that of the six-IMU configuration.

However, the results are specific to the available dataset and block-based validation procedure.

## 12. Troubleshooting

### Conda is not recognized

Use Anaconda Prompt rather than a standard Windows Command Prompt.

If necessary, reinstall Miniconda or Anaconda and initialize Conda for the shell being used.

### ModuleNotFoundError

Verify that the correct environment is active:

```bash
conda activate gait_cnn
```

Then reinstall any missing dependency using:

```bash
python -m pip install PACKAGE_NAME
```

### Jupyter uses the wrong Python environment

Select **Kernel → Change Kernel → Python (Gait CNN)**.

### No CSV files found

Verify that `DATA_DIR` points to the directory containing the labeled recordings and that the filenames end with `_labeled.csv`.

### Training is slow

CPU training is supported but may be slower than GPU training.

To confirm that the scripts work before running the full experiment, use a short test such as:

```bash
python train_6imu_twohead.py --data "gait_labels_20261030\*_labeled.csv" --epochs 1 --output test_6imu
```

After confirming that the script runs correctly, restore the requested 50 epochs.

### Out-of-memory errors

Close unnecessary applications and reduce background memory usage.

If necessary, lower the batch size for testing, but record any changes because modifying batch size affects experimental reproducibility.

## 13. Reproducibility and Experimental Limitations

To ensure that the two sensor configurations can be compared fairly, keep the following consistent:

- Original labeled recordings
- Window length and step
- Normalization procedure
- CNN architecture, except for input channels
- Learning rate, batch size, and epochs
- Cross-validation folds
- Random seed and evaluation metrics

Record the Python and PyTorch versions, operating system, training hardware, input data versions, and experiment settings.

The primary limitations of these experiments include the use of four recordings, block-based cross-validation, possible temporal correlation between nearby gait segments, and evaluation based on classification accuracy rather than event-timing error.

Additional evaluations should include independent walking trials, new participants, heel-strike detection rate, false detections, and heel-strike timing mean absolute error against the synchronized FSR reference signals.

## 14. Summary

The Windows workflow supports reproducible training and evaluation of two shared-backbone, dual-output CNN models.

The first model uses all six IMUs, while the second uses only the left-shank IMU. Both predict stance or swing independently for the left and right legs.

Using a common preprocessing pipeline, architecture, training configuration, and cross-validation procedure enables a controlled comparison of sensor configurations.

The goal is to determine whether bilateral gait-state classification can be maintained while substantially reducing the number of wearable IMU sensors.

