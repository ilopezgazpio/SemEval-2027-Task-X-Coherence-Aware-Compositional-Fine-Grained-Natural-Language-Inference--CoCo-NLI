# DiCo-NLI Trial Data Bundle

Trial folder contains the trial data produced for DICO-NLI task.

## Organizer Source

- `trial_dataset.json`: grouped organizer JSON with 120 source pairs.
- Label balance: 30 `EQUIVALENCE`, 30 `FORWARD_ENTAILMENT`, 30
  `BACKWARD_ENTAILMENT`, 30 `NEGATIVE_OTHER`.
- Source balance: for each label, 15 pairs come from PhrasIS image files and 15
  from PhrasIS headline files.
- Each pair contains `en`, `es`, and `eu` text fields and metadata tracing the
  pair to the original PhrasIS file and line.

## Track Files

For each `track1` to `track4`, the bundle contains:

- `*_participant_labeled.csv`: trial/train-style participant file with labels.
- `*_participant_unlabeled.csv`: test-style participant file without labels.
- `*_hidden_gold.csv`: fixed hidden gold file consumed by the scorer.
- `*_submission_template.csv`: minimal `instance_id,label` submission template.

Track definitions:

- `track1`: English monolingual.
- `track2`: Spanish monolingual.
- `track3`: Basque monolingual.
- `track4`: mixed cross-language pairs only.

## Baselines

`baselines/random/` and `baselines/majority/` contain naive participant
submission files and scorer outputs. `baselines/baseline_summary.json` summarizes
weighted F1, SoftCons, and HardCons for each track.
