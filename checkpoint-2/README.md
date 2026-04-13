# Checkpoint 2 README

This folder contains the final Checkpoint 2 presentation material for the WTA sessional.

It is written to satisfy the checkpoint guideline requirements and is aligned with the latest presentation content.

## Guideline Coverage

The Checkpoint 2 guideline asks for the following items. This checkpoint addresses each one:

1. Clear problem definition and complexity/hardness discussion.
2. Brief overview of existing algorithms with references.
3. Detailed explanation of two selected algorithms with stepwise illustration.
4. Description and rationale of proposed modifications.
5. Experimental comparison on original vs modified algorithms using genuine outputs.
6. GitHub link and single-script reproducibility path.

Guideline source: `guidelines.txt` in this folder.

## Problem and Algorithms Presented

- Problem: static Weapon-Target Assignment minimization objective.
- Original algorithms: MMR and GA.
- Modified algorithms:
  - MMR-IR: tie-aware greedy plus local refinement (1-opt and 2-opt).
  - Hybrid GA: MMR seeding, elitism with stagnation-triggered re-diversification, and threat-proportional mutation.

## Dataset and Experiment Protocol

- Dataset is synthetic and intentionally designed for controlled scaling analysis.
- 9 categories = 3 size tiers x 3 scenario types.
- 20 instances per category; total 180 instances.
- Four algorithms evaluated on each instance, producing 720 result rows.
- Fixed random seed for reproducibility.

This design supports scale and scenario sensitivity analysis required by the checkpoint.

## Reported Findings (from presentation)

- MMR-IR consistently improves MMR.
- Hybrid GA strongly improves GA on medium/large scales.
- Runtime and quality tradeoffs are visualized via log-log scalability and comparison plots.
- Wilcoxon signed-rank testing is used for statistical validation.

## Reproducibility

All experiments can be reproduced using the master script in the code folder:

```bash
cd ../codes
python run_all.py
```

Pipeline stages executed by the script:

1. generate benchmark dataset,
2. run all four algorithms,
3. generate final analysis plots and summary statistics.

Expected outputs from full run:

- 1 CSV result table (`experiment_results.csv`)
- 10 analysis figures (`01` through `10` PNG plots)

Generated artifacts are stored in:

- `../codes/datasets/`
- `../codes/results/experiment_results.csv`
- `../codes/results/figures/`

## Slide Build

To build the current presentation PDF from this folder:

```bash
pdflatex presentation.tex
pdflatex presentation.tex
```

Main produced slide deck:

- `Group_5_Presentation_Checkpoint_2.pdf`

## Files in This Folder

- `guidelines.txt` - official evaluation instructions.
- `presentation.tex` - latest slide source.
- `presentation_prev.tex` - previous draft snapshot.
- `images/` - figures and visual assets used by slides.

## Academic Integrity Note

All stated results are produced from real runs in this repository. Fabricated plots or edited experimental numbers violate course policy and are not used here.
