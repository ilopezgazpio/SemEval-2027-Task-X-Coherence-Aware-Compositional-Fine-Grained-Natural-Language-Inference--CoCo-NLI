## Setting up the Environment

### Python Version

The starter kit requires Python 3.10 or newer. The official scorer integration
uses the repository-level `evaluation_functions/` package, which requires
Python 3.10+.

Check your Python version with:

```bash
python3 --version
```

### Installation

Create a virtual environment named `dicoNLI` from the starter-kit root:

```bash
cd starter_kit
python3 -m venv dicoNLI
source dicoNLI/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This installs the default PyTorch wheel.  If you need a CUDA-specific PyTorch
build, install the matching PyTorch wheel first and then install the remaining
requirements:

```bash
cd starter_kit
python3 -m venv dicoNLI
source dicoNLI/bin/activate
python -m pip install --upgrade pip

# Replace cu121 with the CUDA version available on your system if needed.
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

To leave the environment:

```bash
deactivate
```

---

## DiCo-NLI Starter Kit Usage

This starter kit fine-tunes a Hugging Face sequence-classification model on
DiCo-NLI CSV files and writes prediction files in the official submission
format.  The task-level README in the repository root is the canonical source
for full task definitions, data policy, and evaluation rules.

### Tracks

| Track | Setting |
|-------|---------|
| `track1` | English monolingual |
| `track2` | Spanish monolingual |
| `track3` | Basque monolingual |
| `track4` | Mixed cross-language |

### Expected Data Files

For each track, the released data bundle may contain:

| File pattern | Use in this kit |
|--------------|-----------------|
| `*_participant_labeled.csv` | Training, local development, or trial evaluation with visible labels. |
| `*_participant_unlabeled.csv` | Test-style inference data without labels. |
| `*_hidden_gold.csv` | Optional local reference file for official scoring when released. |
| `*_submission_template.csv` | Minimal example of the required submission columns. |

The model input is built from the `text1` and `text2` columns.  Labels, when
available, must use one of:

```text
EQUIVALENCE
FORWARD_ENTAILMENT
BACKWARD_ENTAILMENT
NEGATIVE_OTHER
```

### Submission Format

Evaluation runs outside Optuna write a timestamped prediction file to
`--results_dir`.  The file uses the official two-column format:

```csv
instance_id,label
```

### Scoring

If `--reference_path` is not provided, the kit writes predictions and reports
local sklearn diagnostics when labels are present in the evaluated file.  If
`--reference_path` points to the matching `*_hidden_gold.csv`, the kit also
calls the official scorer and reports `weighted_f1`, `soft_cons`, and
`hard_cons`.

---

## Repository Structure

Below is an overview of the repository layout and the purpose of each main component:

```
starter_kit/
├── data/                                               # Data storage (place here your data files)
|
├── SLURM/                                              # SLURM scripts to run basic training and evaluation experiments
│   ├── dummy_optuna.slurm                                # Hyperparameter search using Optuna
│   └── dummy.slurm                                       # Basic fine-tuning experiment
|
├── results/                                            # Stores experiment outputs
|
├── src/                                                # Source code for training, evaluation, and model utilities
│   ├── trainer/                                          # Core training logic and related modules
│       ├── helpers/                                      # Helper modules for data, evaluation, and model utilities
│       │   ├── data_reader.py                              # Functions to read and load Dico data
│       │   ├── dataset_generator.py                        # Functions to generate torchvision.datasets
│       │   ├── evaluation.py                               # Metrics and evaluation utilities
│       │   └── model.py                                    # Model definitions and utility functions
│       └── trainer.py                                    # Main training and evaluation loop and orchestration
|
├── README.md                                           # Repository documentation
└── main.py                                             # Main entry point for Optuna hyperparameter search, and model fine-tuning.
```

---

## Quick Start for Replication

### `main.py` Command-Line Arguments

`main.py` is the entry point for training, evaluation, and optional hyperparameter optimization. The following command-line arguments are supported:

| Argument            | Type    | Required | Description                                                                                                              |
| ------------------- | ------- | -------- | ------------------------------------------------------------------------------------------------------------------------ |
| `--model_path`      | `str`   | Yes      | Path to a directory containing model weights saved via `save_pretrained()`, or a Hugging Face model ID.                  |
| `--tokenizer_path`  | `str`   | Yes      | Path to a directory containing tokenizer vocabulary files saved via `save_pretrained()`, or a Hugging Face tokenizer ID. |
| `--do_train`        | `bool`  | Yes      | Whether to perform training.                                                                                             |
| `--train_path`      | `str`   | No       | Path to the training dataset.                                                                                            |
| `--do_dev`          | `bool`  | Yes      | Whether to evaluate on a development dataset.                                                                            |
| `--dev_path`        | `str`   | No       | Path to the organizer-provided development dataset.                                                                       |
| `--do_test`         | `bool`  | Yes      | Whether to perform testing.                                                                                              |
| `--test_path`       | `str`   | No       | Path to the testing dataset.                                                                                             |
| `--seed`            | `int`   | Yes      | Random seed for reproducibility.                                                                                         |
| `--max_length`      | `int`   | Yes      | Maximum sequence length for truncation/padding.                                                                          |
| `--dropout`         | `float` | No       | Dropout rate (0–0.5).                                                                                                    |
| `--warmup_pcrt`     | `float` | No       | Fraction of training steps for linear learning rate warmup (0–0.1, step 0.01).                                           |
| `--lr`              | `float` | No       | Learning rate for the optimizer (common values: 1e-6 to 1e-4).                                                           |
| `--batch_size`      | `int`   | No       | Batch size for training and evaluation (multiple of 8, ≤64).                                                             |
| `--wd`              | `float` | No       | Weight decay for the optimizer (common values: 1e-6 to 1e-1).                                                            |
| `--num_epochs`      | `int`   | No       | Number of training epochs (1–10).                                                                                        |
| `--is_optuna_trial` | `bool`  | Yes      | Whether this run is an Optuna hyperparameter optimization trial.                                                         |
| `--study_name`      | `str`   | No       | Name of the Optuna study.                                                                                                |
| `--storage_name`    | `str`   | No       | Name of the Optuna storage.                                                                                              |
| `--n_trials`        | `int`   | No       | Number of Optuna trials.                                                                                                 |
| `--save_path`       | `str`   | No       | Path to save the trained model.                                                                                          |
| `--epoch_to_stop`   | `int`   | No       | Epoch number at which to stop training for intermediate saving.                                                          |
| `--results_dir`     | `str`   | No       | Directory where timestamped prediction CSV files are written. Defaults to `results`.                                     |
| `--reference_path`  | `str`   | No       | Hidden reference CSV used to compute official DiCo-NLI scores. Must contain the same `instance_id`s as the evaluated file. |

**Notes:**

Some command-line arguments have dependencies. Make sure to set them correctly. We strongly recommend using the examples below as a starting point.

The kit does not create random development splits. Use the organizer-provided
train, development, and test CSV files for the official task setup. Participants
who want different experimental mixtures can edit the CSV files locally at their
own risk.

### Example Usage

**Training with a dataset and evaluation on dev set: ([check file here](SLURM/dummy.slurm))**

```bash
python3 -u "${projectPath}/main.py" \
    --model_path "answerdotai/ModernBERT-base" \
    --tokenizer_path "answerdotai/ModernBERT-base" \
    --do_train \
    --train_path "${projectPath}/data/${trainFile}" \
    --do_dev \
    --dev_path "${projectPath}/data/${devFile}" \
    --do_test \
    --test_path "${projectPath}/data/${testFile}" \
    --seed 42 \
    --dropout 0.1 \
    --warmup_pcrt 0.01 \
    --lr 5e-5 \
    --batch_size 64 \
    --wd 1e-5 \
    --num_epochs 3 \
    --no-is_optuna_trial \
    --max_length 256 \
    --results_dir "${projectPath}/results"
```

Optionally add `--save_path` to save the fine-tuned model and
`--epoch_to_stop` to stop after a selected epoch.

**Hyperparameter search using Optuna: ([check file here](SLURM/dummy_optuna.slurm))**

```bash
python3 -u "${projectPath}/main.py" \
    --model_path "answerdotai/ModernBERT-base" \
    --tokenizer_path "answerdotai/ModernBERT-base" \
    --do_train \
    --train_path "${projectPath}/data/${trainFile}" \
    --do_dev \
    --dev_path "${projectPath}/data/${devFile}" \
    --no-do_test \
    --seed 42 \
    --is_optuna_trial \
    --study_name "${study_name}_0" \
    --storage_name "sqlite:///${projectPath}/results/${study_name}_dummy.db" \
    --n_trials 2 \
    --max_length 256
```

The `--reference_path` file must match the evaluated instances exactly.  For
example, when evaluating a trial participant file, use the corresponding
`*_hidden_gold.csv` file as the reference.

---

## Smoke Test Only: Architecture and Pipeline Validation

The following commands are a **SMOKE TEST ONLY**.  They are intended to verify
that the starter kit can load data, fine-tune a model, write prediction CSVs,
and call the official scorer for each track.

This is **not** an official training protocol and it does **not** define task
splits.  The commands below deliberately reuse the trial data for training,
development, and test-style prediction so that participants can quickly validate
their local environment.  Scores from this smoke test should not be interpreted
as model quality or task performance.

Create the output directory:

```bash
mkdir -p results/smoke
```

### Encoder Smoke Test

```bash
for track in 1 2 3 4; do
  python -u main.py \
    --model_path "distilbert-base-multilingual-cased" \
    --tokenizer_path "distilbert-base-multilingual-cased" \
    --do_train \
    --train_path "../trial_data/dico_nli_trial_track${track}_participant_labeled.csv" \
    --do_dev \
    --dev_path "../trial_data/dico_nli_trial_track${track}_participant_labeled.csv" \
    --do_test \
    --test_path "../trial_data/dico_nli_trial_track${track}_participant_unlabeled.csv" \
    --seed 42 \
    --dropout 0.1 \
    --warmup_pcrt 0.01 \
    --lr 5e-5 \
    --batch_size 8 \
    --wd 1e-5 \
    --num_epochs 1 \
    --no-is_optuna_trial \
    --max_length 128 \
    --results_dir "results/smoke/encoder_track${track}" \
    --reference_path "../trial_data/dico_nli_trial_track${track}_hidden_gold.csv"
done
```

### Decoder Smoke Test

```bash
for track in 1 2 3 4; do
  python -u main.py \
    --model_path "EleutherAI/pythia-160m" \
    --tokenizer_path "EleutherAI/pythia-160m" \
    --do_train \
    --train_path "../trial_data/dico_nli_trial_track${track}_participant_labeled.csv" \
    --do_dev \
    --dev_path "../trial_data/dico_nli_trial_track${track}_participant_labeled.csv" \
    --do_test \
    --test_path "../trial_data/dico_nli_trial_track${track}_participant_unlabeled.csv" \
    --seed 42 \
    --dropout 0.1 \
    --warmup_pcrt 0.01 \
    --lr 5e-5 \
    --batch_size 8 \
    --wd 1e-5 \
    --num_epochs 1 \
    --no-is_optuna_trial \
    --max_length 128 \
    --results_dir "results/smoke/decoder_track${track}" \
    --reference_path "../trial_data/dico_nli_trial_track${track}_hidden_gold.csv"
done
```

### Encoder-Decoder Smoke Test

```bash
for track in 1 2 3 4; do
  python -u main.py \
    --model_path "google/mt5-small" \
    --tokenizer_path "google/mt5-small" \
    --do_train \
    --train_path "../trial_data/dico_nli_trial_track${track}_participant_labeled.csv" \
    --do_dev \
    --dev_path "../trial_data/dico_nli_trial_track${track}_participant_labeled.csv" \
    --do_test \
    --test_path "../trial_data/dico_nli_trial_track${track}_participant_unlabeled.csv" \
    --seed 42 \
    --dropout 0.1 \
    --warmup_pcrt 0.01 \
    --lr 5e-5 \
    --batch_size 8 \
    --wd 1e-5 \
    --num_epochs 1 \
    --no-is_optuna_trial \
    --max_length 128 \
    --results_dir "results/smoke/encoder_decoder_track${track}" \
    --reference_path "../trial_data/dico_nli_trial_track${track}_hidden_gold.csv"
done
```

Collect the generated official score files:

```bash
for f in results/smoke/*track*/predictions_*_official_scores/scores.txt; do
  echo
  echo "$f"
  cat "$f"
done
```
