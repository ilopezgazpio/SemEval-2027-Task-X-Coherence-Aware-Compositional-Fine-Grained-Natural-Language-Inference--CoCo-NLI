# DiCo-NLI Trial Data Bundle

The trial data folder contains the trial data produced for DICO-NLI task.

## Organizer Source

- `trial_dataset.json`: grouped organizer JSON with 461 source pairs selected
  from the 500-pair stratified translation experiment.
- Label distribution: 86 `EQUIVALENCE`, 125 `FORWARD_ENTAILMENT`, 125
  `BACKWARD_ENTAILMENT`, and 125 `NEGATIVE_OTHER`.
- Each pair contains accepted `en`, `es`, and `eu` text fields and metadata
  tracing the pair to the original PhrasIS source and accepted translation
  candidate.
- Spanish and Basque candidates were selected with an exact-collapse filter:
  prefer `gpt-5.4`, fall back to `gpt-5.6-sol` when needed, and reject a source
  pair if either target language has no non-collapsed candidate.
- `trial_translation_audit.json` and `trial_translation_audit.md` summarize the
  accepted translation audit. The accepted trial has zero collapsed Spanish or
  Basque pairs.

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
