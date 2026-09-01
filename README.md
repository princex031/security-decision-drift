Security Decision Drift

Reproducible research artifact for studying defensive-intervention-induced security decision drift.

Overview

This repository accompanies the ICCSAI research work on Security Decision Drift.

The central question is whether a defensive intervention can alter subsequent security evidence and influence a later security decision, even when the external state is otherwise held constant.

Research Artifact

The repository will contain:

- experimental implementation
- canonical configuration
- repeated-seed experiments
- threshold-sensitivity analysis
- statistical analysis
- public-trace evaluation
- reproducibility materials
- automated tests

Evidence Boundary

The controlled simulation is the primary experimental evidence.

Public cyber-trace evaluation is treated separately from deployment-level validation because public datasets do not provide complete recorded defender intervention histories.

Author

Prince
princesoni3365@gmail.com
## UNSW-NB15 Evaluation

The proposed decision mechanism was additionally evaluated using the
publicly available UNSW-NB15 network intrusion dataset.

The dataset was used as real network-traffic data, while intervention
effects were introduced through a controlled experimental layer because
UNSW-NB15 does not contain historical defensive-intervention records.

### Dataset

- Training samples: 82,332
- Testing samples: 175,341

### Baseline

- Accuracy: 87.77%
- Precision: 97.86%
- Recall: 83.86%
- F1: 90.32%

### Reality Check

- Accuracy: 84.07%
- Precision: 98.80%
- Recall: 77.54%
- F1: 86.89%

False actions decreased from 2,189 to 1,123,
representing approximately a 48.7% reduction.

Decision drift rate was 4.91%.

The intervention adjustment is a controlled experimental parameter
and is not claimed to be directly observed in UNSW-NB15.
