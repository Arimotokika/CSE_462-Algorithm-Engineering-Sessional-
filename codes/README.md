# WTA Codes — CSE 462 Group 5 Checkpoint 2

This folder contains all executable code for dataset generation, experiments, and analysis.

---

## Files

| File                   | Description                                                                     |
| ---------------------- | ------------------------------------------------------------------------------- |
| `run_all.py`           | **Master script** — runs the full pipeline end-to-end                           |
| `dataset_generator.py` | Generates 180 synthetic WTA instances across 9 categories                       |
| `experiment_runner.py` | Runs all 4 algorithms on every instance, saves results to CSV                   |
| `analysis.py`          | Loads CSV results, generates 10 plots + Wilcoxon statistical tests              |
| `mmr_original.py`      | MMR greedy heuristic (Hasan & Barua, faithful implementation)                   |
| `mmr_modified.py`      | MMR-IR: greedy + 1-opt + 2-opt local search (our modification)                  |
| `ga_original.py`       | Standard GA (Hasan & Barua, faithful implementation)                            |
| `ga_modified.py`       | Hybrid GA: MMR seed + elitism + threat-proportional mutation (our modification) |
| `wta_utils.py`         | Shared utilities: `compute_solution_value`, `expand_instance`                   |

---

## Environment Setup (one-time)

```bash
# from repository root
py -3 -m venv .venv

# Activate — Git Bash / MSYS2
source ../.venv/Scripts/activate

# Activate — Windows CMD / PowerShell
..\.venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

If you run setup from repository root instead, use:

```bash
pip install -r codes/requirements.txt
```

---

## Commands

### Run everything at once

```bash
python run_all.py
```

_Runtime: approximately 3 to 4 hours total (large tiers dominate)._

---

### Run tiers separately (recommended)

```bash
# Generate dataset (5 seconds)
python dataset_generator.py

# Small tier: 60 instances, ~10 minutes
python experiment_runner.py --categories small_balanced,small_scarce,small_rich

# Medium tier: 60 instances, ~50 minutes
python experiment_runner.py --categories medium_balanced,medium_scarce,medium_rich --append

# Large tier: 60 instances, ~2 hours
python experiment_runner.py --categories large_balanced,large_scarce,large_rich --append

# Generate all plots + stats (after experiments complete)
python analysis.py
```

This mode is useful if you need resumable execution by tier.

---

### Quick test (5 small instances, ~1 minute)

```bash
python experiment_runner.py --categories small_balanced --instances 1-5
python analysis.py
```

---

### Per-algorithm tests

```bash
python mmr_original.py   # smoke test: 3x3 example
python mmr_modified.py   # MMR-O vs MMR-IR comparison
python ga_original.py    # smoke test: 3x3 example
python ga_modified.py    # all 4 algorithms on 3x3
```

---

## Dataset Categories

| Category        | W   | T   | Scenario       |
| --------------- | --- | --- | -------------- |
| small_balanced  | 20  | 20  | Balanced (W=T) |
| small_scarce    | 10  | 20  | Scarce (W<T)   |
| small_rich      | 30  | 20  | Rich (W>T)     |
| medium_balanced | 100 | 100 | Balanced       |
| medium_scarce   | 50  | 100 | Scarce         |
| medium_rich     | 150 | 100 | Rich           |
| large_balanced  | 250 | 250 | Balanced       |
| large_scarce    | 125 | 250 | Scarce         |
| large_rich      | 375 | 250 | Rich           |

---

## Output

After full pipeline:

- `results/experiment_results.csv` — 720 rows (180 instances × 4 algorithms)
- `results/figures/01_box_solution_quality.png` — Box plots per tier
- `results/figures/02_bar_mean_objective.png` — Mean objective per tier
- `results/figures/03_bar_improvement.png` — % improvement of modifications
- `results/figures/04_line_time_scalability.png` — Runtime scalability
- `results/figures/05_bar_gap.png` — Optimality gap
- `results/figures/06_scenario_comparison.png` — Balanced vs scarce vs rich
- `results/figures/07_violin_value_dist.png` — Value distributions
- `results/figures/08_bar_scenario_improvement.png` — Improvement by scenario
- `results/figures/09_ga_improvement_vs_size.png` — Hybrid GA gain vs problem size

- `results/figures/10_mmr_improvement_vs_size.png` -- MMR-IR gain vs problem size

All generated figures are intended to match the plots used in the Checkpoint 2 presentation.

Main tabular output:

- `results/experiment_results.csv`

Expected scale:

- 720 rows = 180 instances * 4 algorithms.

---

## Experiment CLI Options

```
python experiment_runner.py [OPTIONS]

Options:
  --categories  Comma-separated category names (default: all 9)
                Example: --categories small_balanced,medium_scarce
  --instances   Instance range START-END (default: all 20)
                Example: --instances 1-5
  --append      Append to existing CSV instead of overwriting
                Use this when running tiers separately
```

## Reproducibility Contract

- Single-command reproduction entrypoint is `python run_all.py`.
- Do not manually edit `results/experiment_results.csv`.
- Regenerate figures with `python analysis.py` after any experiment rerun.
- Keep dataset seed and category definitions unchanged unless reporting a new benchmark configuration.
