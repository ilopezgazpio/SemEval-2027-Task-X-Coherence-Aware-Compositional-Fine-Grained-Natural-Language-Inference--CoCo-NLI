# Item 4: Evaluation Plan

This folder contains the supplementary evaluation-plan document requested for the conditional SemEval-2027 revision.

Files:

- `4.Evaluation_Plan.pdf`: reviewer-facing evaluation plan.
- `4.Evaluation_Plan.tex`: LaTeX source for the PDF.

The plan defines the official prediction format, label inventory, track-level reporting, validation rules, and evaluation metrics. DiCo-NLI reports three main official scores for each track:

- weighted F1 over all four labels;
- `SoftCons`, directional self-consistency over reversible source pairs;
- `HardCons`, strict paired correctness over reversible source pairs.

Secondary diagnostics include macro-F1, per-label scores, confusion matrices, validation reports, and optional macro-averages across tracks. These diagnostics do not replace the three official per-track scores.
