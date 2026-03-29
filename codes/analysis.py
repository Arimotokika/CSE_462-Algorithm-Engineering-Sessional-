"""
Analysis & Visualisation
=========================
Reads experiment_results.csv and produces publication-quality plots
for the Checkpoint 2 presentation.

Dataset structure: 3 tiers (small, medium, large) x 3 scenarios
(balanced, scarce, rich) = 9 categories, 20 instances each.

Algorithms: MMR_Original, MMR_Modified, GA_Original, GA_Modified.

Plots generated (saved to results/figures/):
  01_box_solution_quality.png     -- box plots per tier x algorithm (balanced)
  02_bar_mean_objective.png       -- grouped bar: mean objective per tier
  03_bar_improvement.png          -- % improvement modified vs original per tier
  04_line_time_scalability.png    -- log-log runtime vs problem size
  05_bar_gap.png                  -- optimality gap from best-known per tier
  06_scenario_comparison.png      -- balanced vs scarce vs rich per tier
  07_violin_value_dist.png        -- violin plots: modified vs original
  08_bar_scenario_improvement.png -- improvement % by scenario type
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

RESULTS_CSV = "results/experiment_results.csv"
FIGURES_DIR = "results/figures"
DPI         = 150

# ── category definitions ──────────────────────────────────────────────────────

TIERS     = ["small", "medium", "large"]
SCENARIOS = ["balanced", "scarce", "rich"]

ALL_CATS = [f"{t}_{s}" for t in TIERS for s in SCENARIOS]

TIER_LABELS = {
    "small":  "Small\n(20 targets)",
    "medium": "Medium\n(100 targets)",
    "large":  "Large\n(250 targets)",
}

# Representative problem size per tier (max(W,T) for balanced case)
TIER_SIZES = {"small": 20, "medium": 100, "large": 250}

ALGO_ORDER  = ["MMR_Original", "MMR_Modified", "GA_Original", "GA_Modified"]
ALGO_LABELS = {
    "MMR_Original": "MMR (orig)",
    "MMR_Modified": "MMR-IR (mod)",
    "GA_Original":  "GA (orig)",
    "GA_Modified":  "Hybrid GA (mod)",
}
ALGO_COLORS = {
    "MMR_Original": "#4878CF",
    "MMR_Modified": "#1F3A6E",
    "GA_Original":  "#6ABF69",
    "GA_Modified":  "#2E7D32",
}

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": DPI, "savefig.bbox": "tight"})


# ── helpers ───────────────────────────────────────────────────────────────────

def _tier(cat: str) -> str:
    for t in TIERS:
        if cat.startswith(t):
            return t
    return "unknown"


def _scenario(cat: str) -> str:
    for s in SCENARIOS:
        if cat.endswith(s):
            return s
    return "unknown"


def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV)
    present = [c for c in ALL_CATS if c in df["category"].unique()]
    df["category"] = pd.Categorical(df["category"], categories=present, ordered=True)
    df["label"]    = df["algorithm"].map(ALGO_LABELS)
    df["tier"]     = df["category"].astype(str).map(_tier)
    df["scenario"] = df["category"].astype(str).map(_scenario)
    df["tier"]     = pd.Categorical(df["tier"], categories=TIERS, ordered=True)
    df["scenario"] = pd.Categorical(df["scenario"], categories=SCENARIOS, ordered=True)
    return df


def best_known(df: pd.DataFrame) -> pd.DataFrame:
    bk = (
        df.groupby(["category", "instance_id"])["value"]
        .min().reset_index()
        .rename(columns={"value": "best_known"})
    )
    merged = df.merge(bk, on=["category", "instance_id"])
    merged["best_known"] = merged["best_known"].replace(0, np.nan)
    return merged


def _safe_pct_improvement(orig, mod):
    orig = np.where(np.asarray(orig) == 0, np.nan, np.asarray(orig, dtype=float))
    mod  = np.asarray(mod, dtype=float)
    return 100.0 * (orig - mod) / orig


def _savefig(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 1: Box plots per tier (balanced instances) ──────────────────────────

def plot_box_solution_quality(df):
    balanced = df[df["scenario"] == "balanced"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for ax, tier in zip(axes, TIERS):
        sub = balanced[balanced["tier"] == tier]
        if sub.empty:
            continue
        order = [ALGO_LABELS[a] for a in ALGO_ORDER if a in sub["algorithm"].unique()]
        palette = {ALGO_LABELS[a]: ALGO_COLORS[a] for a in ALGO_ORDER}
        sns.boxplot(data=sub, x="label", y="value", order=order,
                    palette=palette, ax=ax, width=0.6)
        ax.set_title(TIER_LABELS[tier], fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("Expected Survival Value" if ax == axes[0] else "")
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Solution Quality Distribution (Balanced Instances)", fontsize=14, y=1.02)
    fig.tight_layout()
    _savefig(fig, "01_box_solution_quality.png")


# ── Plot 2: Grouped bar — mean objective per tier ────────────────────────────

def plot_bar_mean_objective(df):
    balanced = df[df["scenario"] == "balanced"].copy()
    summary = (balanced.groupby(["tier", "algorithm"])["value"]
               .agg(["mean", "std"]).reset_index())
    summary["label"] = summary["algorithm"].map(ALGO_LABELS)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(TIERS))
    width = 0.18
    for i, algo in enumerate(ALGO_ORDER):
        sub = summary[summary["algorithm"] == algo]
        means = [sub[sub["tier"] == t]["mean"].values[0] if t in sub["tier"].values else 0
                 for t in TIERS]
        stds  = [sub[sub["tier"] == t]["std"].values[0] if t in sub["tier"].values else 0
                 for t in TIERS]
        ax.bar(x + i * width, means, width, yerr=stds, capsize=3,
               label=ALGO_LABELS[algo], color=ALGO_COLORS[algo])

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([TIER_LABELS[t] for t in TIERS])
    ax.set_ylabel("Mean Expected Survival Value (lower = better)")
    ax.set_title("Mean Objective Value by Problem Size (Balanced)")
    ax.legend()
    fig.tight_layout()
    _savefig(fig, "02_bar_mean_objective.png")


# ── Plot 3: % improvement modified vs original per tier ──────────────────────

def plot_bar_improvement(df):
    balanced = df[df["scenario"] == "balanced"].copy()
    pairs = [("MMR_Original", "MMR_Modified", "MMR-IR vs MMR"),
             ("GA_Original",  "GA_Modified",  "Hybrid GA vs GA")]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(TIERS))
    width = 0.30

    for i, (orig_algo, mod_algo, label) in enumerate(pairs):
        improvements = []
        for tier in TIERS:
            sub = balanced[balanced["tier"] == tier]
            orig = sub[sub["algorithm"] == orig_algo].sort_values("instance_id")["value"]
            mod  = sub[sub["algorithm"] == mod_algo].sort_values("instance_id")["value"]
            if len(orig) > 0 and len(mod) > 0:
                pct = _safe_pct_improvement(orig.values, mod.values)
                improvements.append(np.nanmean(pct))
            else:
                improvements.append(0)
        color = ALGO_COLORS[mod_algo]
        ax.bar(x + i * width, improvements, width, label=label, color=color)

    ax.set_xticks(x + 0.5 * width)
    ax.set_xticklabels([TIER_LABELS[t] for t in TIERS])
    ax.set_ylabel("Mean % Improvement")
    ax.set_title("Improvement of Modified over Original (Balanced)")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    _savefig(fig, "03_bar_improvement.png")


# ── Plot 4: Time scalability (log-log) ───────────────────────────────────────

def plot_time_scalability(df):
    balanced = df[df["scenario"] == "balanced"].copy()
    summary = balanced.groupby(["tier", "algorithm"])["time_sec"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(9, 6))
    for algo in ALGO_ORDER:
        sub = summary[summary["algorithm"] == algo]
        sizes = [TIER_SIZES[t] for t in sub["tier"]]
        times = sub["time_sec"].values
        ax.plot(sizes, times, "o-", label=ALGO_LABELS[algo],
                color=ALGO_COLORS[algo], linewidth=2, markersize=7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Problem Size (max targets)")
    ax.set_ylabel("Mean Runtime (seconds)")
    ax.set_title("Runtime Scalability (Balanced Instances)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, "04_line_time_scalability.png")


# ── Plot 5: Optimality gap from best-known ───────────────────────────────────

def plot_bar_gap(df):
    bk_df = best_known(df)
    balanced = bk_df[bk_df["scenario"] == "balanced"].copy()
    balanced["gap_pct"] = 100.0 * (balanced["value"] - balanced["best_known"]) / balanced["best_known"]

    summary = balanced.groupby(["tier", "algorithm"])["gap_pct"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(TIERS))
    width = 0.18
    for i, algo in enumerate(ALGO_ORDER):
        sub = summary[summary["algorithm"] == algo]
        gaps = [sub[sub["tier"] == t]["gap_pct"].values[0]
                if t in sub["tier"].values else 0 for t in TIERS]
        ax.bar(x + i * width, gaps, width,
               label=ALGO_LABELS[algo], color=ALGO_COLORS[algo])

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([TIER_LABELS[t] for t in TIERS])
    ax.set_ylabel("Mean % Gap from Best-Known")
    ax.set_title("Optimality Gap (Balanced Instances)")
    ax.legend()
    fig.tight_layout()
    _savefig(fig, "05_bar_gap.png")


# ── Plot 6: Scenario comparison (balanced / scarce / rich) ───────────────────

def plot_scenario_comparison(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    for ax, tier in zip(axes, TIERS):
        sub = df[df["tier"] == tier]
        summary = sub.groupby(["scenario", "algorithm"])["value"].mean().reset_index()
        summary["label"] = summary["algorithm"].map(ALGO_LABELS)

        x = np.arange(len(SCENARIOS))
        width = 0.18
        for i, algo in enumerate(ALGO_ORDER):
            asub = summary[summary["algorithm"] == algo]
            vals = [asub[asub["scenario"] == s]["value"].values[0]
                    if s in asub["scenario"].values else 0 for s in SCENARIOS]
            ax.bar(x + i * width, vals, width,
                   label=ALGO_LABELS[algo] if tier == TIERS[0] else "",
                   color=ALGO_COLORS[algo])

        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels([s.capitalize() for s in SCENARIOS])
        ax.set_title(TIER_LABELS[tier], fontsize=12)
        ax.set_ylabel("Mean Survival Value" if ax == axes[0] else "")

    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Performance Across Weapon-Target Scenarios", fontsize=14, y=1.02)
    fig.tight_layout()
    _savefig(fig, "06_scenario_comparison.png")


# ── Plot 7: Violin plots ─────────────────────────────────────────────────────

def plot_violin_value_dist(df):
    balanced = df[df["scenario"] == "balanced"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    pairs = [
        (axes[0], "MMR_Original", "MMR_Modified", "MMR vs MMR-IR"),
        (axes[1], "GA_Original",  "GA_Modified",  "GA vs Hybrid GA"),
    ]

    for ax, orig_algo, mod_algo, title in pairs:
        sub = balanced[balanced["algorithm"].isin([orig_algo, mod_algo])].copy()
        sub["label"] = sub["algorithm"].map(ALGO_LABELS)
        palette = {ALGO_LABELS[orig_algo]: ALGO_COLORS[orig_algo],
                   ALGO_LABELS[mod_algo]:  ALGO_COLORS[mod_algo]}
        order = [ALGO_LABELS[orig_algo], ALGO_LABELS[mod_algo]]

        # Seaborn 0.13 compatible: use hue with dodge
        sns.violinplot(data=sub, x="tier", y="value", hue="label",
                       hue_order=order, palette=palette,
                       inner="quartile", ax=ax)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel("Expected Survival Value" if ax == axes[0] else "")
        ax.legend(fontsize=9)

    fig.suptitle("Value Distribution: Original vs Modified (Balanced)", fontsize=14, y=1.02)
    fig.tight_layout()
    _savefig(fig, "07_violin_value_dist.png")


# ── Plot 8: Improvement by scenario type ─────────────────────────────────────

def plot_bar_scenario_improvement(df):
    pairs = [("MMR_Original", "MMR_Modified", "MMR-IR vs MMR"),
             ("GA_Original",  "GA_Modified",  "Hybrid GA vs GA")]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(SCENARIOS))
    width = 0.30

    for i, (orig_algo, mod_algo, label) in enumerate(pairs):
        improvements = []
        for scenario in SCENARIOS:
            sub = df[df["scenario"] == scenario]
            orig = sub[sub["algorithm"] == orig_algo].sort_values(
                ["category", "instance_id"])["value"]
            mod = sub[sub["algorithm"] == mod_algo].sort_values(
                ["category", "instance_id"])["value"]
            if len(orig) > 0 and len(mod) > 0:
                pct = _safe_pct_improvement(orig.values, mod.values)
                improvements.append(np.nanmean(pct))
            else:
                improvements.append(0)
        color = ALGO_COLORS[mod_algo]
        ax.bar(x + i * width, improvements, width, label=label, color=color)

    ax.set_xticks(x + 0.5 * width)
    ax.set_xticklabels([s.capitalize() for s in SCENARIOS])
    ax.set_ylabel("Mean % Improvement")
    ax.set_title("Improvement by Scenario Type (All Tiers)")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    _savefig(fig, "08_bar_scenario_improvement.png")


# ── Summary tables & statistical tests ───────────────────────────────────────

def print_summary(df):
    print("\n" + "=" * 80)
    print("SUMMARY TABLES")
    print("=" * 80)

    balanced = df[df["scenario"] == "balanced"]

    # Mean objective value
    print("\n--- Mean Objective Value (Balanced) ---")
    pivot = balanced.pivot_table(values="value", index="tier",
                                 columns="algorithm", aggfunc="mean")
    pivot = pivot.reindex(columns=ALGO_ORDER)
    pivot.columns = [ALGO_LABELS.get(c, c) for c in pivot.columns]
    print(pivot.round(2).to_string())

    # Mean runtime
    print("\n--- Mean Runtime in seconds (Balanced) ---")
    pivot_t = balanced.pivot_table(values="time_sec", index="tier",
                                    columns="algorithm", aggfunc="mean")
    pivot_t = pivot_t.reindex(columns=ALGO_ORDER)
    pivot_t.columns = [ALGO_LABELS.get(c, c) for c in pivot_t.columns]
    print(pivot_t.round(4).to_string())

    # Wilcoxon signed-rank tests
    print("\n--- Wilcoxon Signed-Rank Tests (alpha=0.05) ---")
    test_pairs = [("MMR_Original", "MMR_Modified"),
                  ("GA_Original",  "GA_Modified")]
    for tier in TIERS:
        sub = balanced[balanced["tier"] == tier]
        for orig_algo, mod_algo in test_pairs:
            orig_vals = sub[sub["algorithm"] == orig_algo].sort_values("instance_id")["value"].values
            mod_vals  = sub[sub["algorithm"] == mod_algo].sort_values("instance_id")["value"].values
            if len(orig_vals) < 5 or len(mod_vals) < 5:
                continue
            diff = orig_vals - mod_vals
            if np.all(diff == 0):
                print(f"  {tier:6s} | {ALGO_LABELS[orig_algo]:15s} vs {ALGO_LABELS[mod_algo]:18s} "
                      f"| identical (no difference)")
                continue
            try:
                stat, pval = stats.wilcoxon(orig_vals, mod_vals, alternative="greater")
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                print(f"  {tier:6s} | {ALGO_LABELS[orig_algo]:15s} vs {ALGO_LABELS[mod_algo]:18s} "
                      f"| W={stat:8.1f}  p={pval:.4f}  {sig}")
            except Exception as e:
                print(f"  {tier:6s} | {ALGO_LABELS[orig_algo]:15s} vs {ALGO_LABELS[mod_algo]:18s} "
                      f"| Error: {e}")

    # All-scenario summary
    print("\n--- Mean Objective Value (All Scenarios) ---")
    pivot_all = df.pivot_table(values="value", index=["tier", "scenario"],
                                columns="algorithm", aggfunc="mean")
    pivot_all = pivot_all.reindex(columns=ALGO_ORDER)
    pivot_all.columns = [ALGO_LABELS.get(c, c) for c in pivot_all.columns]
    print(pivot_all.round(2).to_string())

    print("\n" + "=" * 80)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading results ...")
    df = load_results()
    print(f"  {len(df)} rows, {df['category'].nunique()} categories, "
          f"{df['algorithm'].nunique()} algorithms")

    print("\nGenerating plots ...")
    plot_box_solution_quality(df)
    plot_bar_mean_objective(df)
    plot_bar_improvement(df)
    plot_time_scalability(df)
    plot_bar_gap(df)
    plot_scenario_comparison(df)
    plot_violin_value_dist(df)
    plot_bar_scenario_improvement(df)

    print_summary(df)
    print("\nDone!")


if __name__ == "__main__":
    main()
