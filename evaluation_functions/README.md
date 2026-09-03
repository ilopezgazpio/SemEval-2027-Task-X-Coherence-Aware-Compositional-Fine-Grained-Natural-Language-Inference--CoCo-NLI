# DiCo-NLI Evaluation Functions

This folder contains the dependency-free scorer for fixed DiCo-NLI CSV files.
It is intended for local validation, starter kits, CI, and CodaBench scoring containers.

The scorer requires Python 3.10 or newer.

The scorer is intentionally agnostic to track construction. Tracks are produced
beforehand by `participant_dataset_production`; the evaluator only compares a hidden fixed
reference CSV with a participant prediction CSV.

## Hidden Reference CSV

The hidden reference file must contain at least these columns:

```text
instance_id,pair_id,reverse_pair_id,label
```

It may contain extra organizer/debug columns such as `text1_lang`, `text2_lang`,
`text1`, and `text2`; the scorer ignores them.

`reverse_pair_id` is empty for instances that do not participate in consistency
metrics, such as `NEGATIVE_OTHER`. For reversible labels, it must point to the
reciprocal `instance_id`, and that reciprocal row must point back.

Accepted labels:

```text
EQUIVALENCE
FORWARD_ENTAILMENT
BACKWARD_ENTAILMENT
NEGATIVE_OTHER
```

## Prediction CSV

Participant submissions must contain exactly:

```text
instance_id,label
```

The prediction file must contain one prediction for every hidden reference
`instance_id`, with no duplicates and no unknown ids.

## Official Scores

The scorer reports the three main task scores:

- `weighted_f1`: weighted F1 over all four labels.
- `soft_cons`: prediction-level directional compatibility over reciprocal
  reverse pairs.
- `hard_cons`: strict paired correctness over reciprocal reverse pairs.

It also returns diagnostic information: macro-F1, accuracy, per-label
precision/recall/F1, confusion matrix, reversible-pair counts, and a bounded list
of pair-level errors.

## CLI Usage

```bash
python3 -m evaluation_functions \
  --gold gold.csv \
  --predictions predictions.csv \
  --output-dir scores
```

This writes:

- `scores/scores.json`: complete structured report;
- `scores/scores.txt`: compact key-value output for competition platforms.

## CodaBench Directory Mode

```bash
python3 -m evaluation_functions \
  --reference-dir /app/input/ref \
  --submission-dir /app/input/res \
  --output-dir /app/output
```

By default this expects:

- `/app/input/ref/gold.csv`
- `/app/input/res/predictions.csv`

Use `--gold-filename` and `--prediction-filename` to customize those names.

## Validation Policy

The scorer rejects malformed submissions before scoring. Rejection conditions
include:

- invalid, missing, or duplicate reference/prediction identifiers;
- invalid labels;
- non-reciprocal `reverse_pair_id` links;
- incompatible reverse labels;
- missing prediction columns;
- unexpected prediction columns;
- duplicate prediction `instance_id` values;
- missing predictions;
- predictions for unknown instances;
- empty files or empty prediction rows;
- NUL bytes or oversized files.

## Modules

- `labels.py`: official label inventory and reversal operator.
- `io.py`: fixed reference and prediction loading.
- `validation.py`: reference and prediction validation.
- `metrics.py`: pure metric computation.
- `scorer.py`: orchestration and output writing.
- `templates.py`: minimal prediction-template generation.
- `aggregation.py`: optional cross-track macro-average helpers.
- `cli.py`: command-line and CodaBench entry point.
