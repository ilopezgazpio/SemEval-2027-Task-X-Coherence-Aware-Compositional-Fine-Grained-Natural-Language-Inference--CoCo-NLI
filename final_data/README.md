# DiCo-NLI Final Training And Development Data

This folder contains the public training and development data for SemEval 2027
Task 2 DiCo-NLI.

The evaluation test files and test gold labels are not included here. They will
be released through the official evaluation process.

The zip files in this folder are convenience packages with the same contents as
the corresponding `train/` and `dev/` folders:

```text
final_data_train.zip
final_data_dev.zip
```

## Files

Each split has one set of files per track:

```text
dico_nli_<split>_<track>_participant_labeled.csv
dico_nli_<split>_<track>_submission_template.csv
dico_nli_<split>_<track>_reference.csv
```

Use `participant_labeled.csv` files for training, development, and local
experiments. Use `submission_template.csv` as the required prediction format.
Use `reference.csv` with the official scorer when evaluating local predictions.

The reference files contain the additional `reverse_pair_id` field needed to
compute the official directional consistency metrics.

## Splits

| Split | Track 1 | Track 2 | Track 3 | Track 4 |
|-------|--------:|--------:|--------:|--------:|
| Train | 3042 | 3042 | 3042 | 18252 |
| Dev | 660 | 660 | 660 | 3960 |

## Official Local Scoring

Example for Track 1 development predictions from a Python 3.10+ environment:

```bash
python3 -m evaluation_functions \
  --gold final_data/dev/dico_nli_dev_track1_reference.csv \
  --predictions results/my_dev_track1_predictions.csv \
  --output-dir results/dev_track1_scores
```

Prediction files must contain exactly two columns: `instance_id,label`.
