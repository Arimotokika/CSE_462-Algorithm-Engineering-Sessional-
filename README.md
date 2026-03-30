# CSE 462 Algorithm Engineering Sessional

Weapon-Target Assignment (WTA) project repository for CSE 462, Group 5.

## Repository Scope

This repository contains two course milestones:

- Checkpoint 1: problem definition, hardness discussion, algorithm survey, and early presentation build.
- Checkpoint 2: full implementation, modification design, experiments, and reproducible outputs.

## WTA Objective

Static WTA is solved as a minimization problem:

```text
min Z = sum_j V_j * prod_i (1 - p_ij)^x_ij
```

where:

- `V_j` = value/importance of target `j`
- `p_ij` = kill probability of weapon `i` on target `j`
- `x_ij` = assignment decision for weapon-target pairing

## Checkpoint 2 At A Glance

### Implemented Algorithms

- MMR (original): greedy marginal-return heuristic.
- MMR-IR (modified): tie-aware greedy plus 1-opt and 2-opt refinement.
- GA (original): baseline GA with tournament selection, crossover, mutation, time budget.
- Hybrid GA (modified): MMR-seeded population, elitism with stagnation recovery, and threat-proportional mutation.

### Benchmark Design

- 180 synthetic instances across 9 categories.
- Size tiers: small (20), medium (100), large (250).
- Scenario types: balanced (`W=T`), scarce (`W<T`), rich (`W>T`).
- 20 instances per category, RNG seed fixed at 42.

### Result Summary

- MMR-IR improves MMR consistently across tiers/scenarios.
- Hybrid GA gives the largest gains on medium and large scales.
- Statistical support uses Wilcoxon signed-rank testing.

## Quick Reproduction

Run from repository root:

```bash
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r codes/requirements.txt

cd codes
python run_all.py
```

`run_all.py` executes:

1. dataset generation,
2. experiment execution for all 4 algorithms,
3. analysis and figure generation (9 plots).

Generated outputs:

- `codes/datasets/`
- `codes/results/experiment_results.csv`
- `codes/results/figures/`

## Repository Layout

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

## Documentation Index

- Checkpoint 1 details: `checkpoint-1/README.md`
- Checkpoint 2 guideline alignment: `checkpoint-2/README.md`
- Code usage and CLI guide: `codes/README.md`

## Academic Integrity

All reported plots and values are produced from real experiment execution in this repository.
