# CSE 462 Algorithm Engineering Sessional

Weapon-Target Assignment (WTA) project for CSE 462, Group 5.

This repository contains both checkpoints:

- Checkpoint 1: problem framing, complexity discussion, algorithm survey, and first presentation build.
- Checkpoint 2: full implementation, experimental evaluation, and reproducible pipeline.

## Checkpoint 2 Summary

We solve the static WTA minimization objective:

Z = sum_j V_j * prod_i (1 - p_ij)^x_ij

where V_j is target value, p_ij is weapon-target kill probability, and x_ij is assignment count.

### Algorithms implemented

- MMR (original): greedy marginal-return heuristic.
- MMR-IR (modified): tie-aware greedy + 1-opt local reassignment + 2-opt swap refinement.
- GA (original): baseline genetic algorithm with tournament selection, crossover, mutation, time budget.
- Hybrid GA (modified): MMR-seeded initial population + elitism with stagnation control + threat-proportional mutation.

### Dataset design

- 180 synthetic instances across 9 categories:
  - tiers: small (20), medium (100), large (250)
  - scenarios: balanced (W=T), scarce (W<T), rich (W>T)
- 20 instances per category.
- RNG seed fixed at 42 for reproducibility.

### Key findings from experiments

- MMR-IR improves MMR consistently (roughly 0.2% to 4.8% by tier/scenario).
- Hybrid GA strongly improves GA on larger scales (up to ~36% to 54% on large-tier settings).
- Wilcoxon signed-rank tests support significance of improvements.

## Repository Structure

```text
CSE_462-Algorithm-Engineering-Sessional/
|- README.md
|- checkpoint-1/
|  |- README.md
|  |- guidelines.txt
|  |- presentation.tex
|  `- images/
|- checkpoint-2/
|  |- README.md
|  |- guidelines.txt
|  |- presentation.tex
|  |- presentation_prev.tex
|  `- images/
|- codes/
|  |- README.md
|  |- run_all.py
|  |- dataset_generator.py
|  |- experiment_runner.py
|  |- analysis.py
|  |- mmr_original.py
|  |- mmr_modified.py
|  |- ga_original.py
|  |- ga_modified.py
|  |- wta_utils.py
|  |- requirements.txt
|  |- datasets/
|  `- results/
`- files/
```

## Reproducibility Quick Start

```bash
# from repository root
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r codes/requirements.txt

cd codes
python run_all.py
```

This single command chain will:

1. generate dataset instances,
2. run all four algorithms,
3. produce results CSV and 9 analysis plots.

Outputs are written under:

- codes/datasets/
- codes/results/experiment_results.csv
- codes/results/figures/

## Documentation Index

- Checkpoint 1 details: checkpoint-1/README.md
- Checkpoint 2 presentation and compliance notes: checkpoint-2/README.md
- Code-level usage and CLI: codes/README.md

## Academic Integrity Note

All reported values and plots are generated from actual experiment runs in this repository.
