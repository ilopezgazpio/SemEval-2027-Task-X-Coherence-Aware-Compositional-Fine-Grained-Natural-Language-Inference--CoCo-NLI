# Item 2: Pilot/Trial Data Demonstration

This folder contains the reviewer-facing PDF for SemEval revision item 2.

The actual pilot/trial data bundle is in the repository-level `trial/` folder.
It contains:

- `trial_dataset.json`: grouped organizer JSON with 120 source pairs.
- Four derived track bundles for `track1`, `track2`, `track3`, and `track4`.
- Labeled participant CSVs for trial/train simulation.
- Unlabeled participant CSVs for blind-test simulation.
- Hidden gold CSVs consumed by the scorer.
- Minimal submission templates.
- Random and majority baseline submissions plus scorer reports.

The PDF `2.Pilot_Trial_Data_Demonstration.pdf` describes this design and
summarizes the baseline results.

The commands used to produce the trial data and baseline reports are documented
in `participant_dataset_production/README.md`.
