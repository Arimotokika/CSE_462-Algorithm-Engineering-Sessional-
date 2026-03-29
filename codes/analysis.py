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


def _label_bars(ax, bars, fmt="{:.1f}", fontsize=7.5, min_show=0.0):
    """Add a value label centred on top of every bar (skip near-zero bars)."""
    ylim = ax.get_ylim()
    span = max(ylim[1] - ylim[0], 1e-9)
    for bar in bars:
        h = bar.get_height()
        if abs(h) < min_show:
            continue
        label = fmt.format(h)
        # place label just above the bar; if bar is tiny use a small fixed offset
        y = h + span * 0.012 if h >= 0 else h - span * 0.025
        ax.text(bar.get_x() + bar.get_width() / 2.0, y,
                label, ha="center", va="bottom" if h >= 0 else "top",
                fontsize=fontsize, fontweight="bold", clip_on=True)


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
        # Annotate median value above each box
        for j, lbl in enumerate(order):
            vals = sub[sub["label"] == lbl]["value"]
            med = vals.median()
            ax.text(j, ax.get_ylim()[1] * 0.97, f"med={med:.0f}",
                    ha="center", va="top", fontsize=7.5, color="black",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, ec="none"))
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
        bars = ax.bar(x + i * width, means, width, yerr=stds, capsize=3,
                      label=ALGO_LABELS[algo], color=ALGO_COLORS[algo])
        _label_bars(ax, bars, fmt="{:.0f}", fontsize=7)

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
        bars = ax.bar(x + i * width, improvements, width, label=label, color=color)
        _label_bars(ax, bars, fmt="{:.1f}%", fontsize=8)

    ax.set_xticks(x + 0.5 * width)
    ax.set_xticklabels([TIER_LABELS[t] for t in TIERS])
    ax.set_ylabel("Mean % Improvement (lower survival = better)")
    ax.set_title("Improvement of Modified over Original (Balanced Instances)")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    _savefig(fig, "03_bar_improvement.png")


# ── Plot 4: Time scalability (log-log) with trend lines ──────────────────────

def plot_time_scalability(df):
    balanced = df[df["scenario"] == "balanced"].copy()
    summary = balanced.groupby(["tier", "algorithm"])["time_sec"].mean().reset_index()

    # Extended x-axis range for trend line extrapolation
    size_vals = np.array(sorted(TIER_SIZES.values()))
    x_fine = np.logspace(np.log10(size_vals[0] * 0.7),
                         np.log10(size_vals[-1] * 1.5), 200)

    fig, ax = plt.subplots(figsize=(10, 6))
    for algo in ALGO_ORDER:
        sub = summary[summary["algorithm"] == algo]
        sizes = np.array([TIER_SIZES[t] for t in sub["tier"]], dtype=float)
        times = sub["time_sec"].values

        # Plot actual data points
        ax.plot(sizes, times, "o", color=ALGO_COLORS[algo], markersize=8, zorder=5)

        # Power-law trend line: log(t) = a*log(n) + b
        if len(sizes) >= 2:
            log_s = np.log10(sizes)
            log_t = np.log10(np.maximum(times, 1e-9))
            coeffs = np.polyfit(log_s, log_t, 1)
            trend = 10 ** np.polyval(coeffs, np.log10(x_fine))
            ax.plot(x_fine, trend, "--", color=ALGO_COLORS[algo], linewidth=1.5,
                    alpha=0.7, label=f"{ALGO_LABELS[algo]} (slope={coeffs[0]:.2f})")

    # Tick labels at actual sizes
    ax.set_xticks(size_vals)
    ax.set_xticklabels([str(int(s)) for s in size_vals])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Problem Size (number of targets, balanced)")
    ax.set_ylabel("Mean Runtime (seconds)")
    ax.set_title("Runtime Scalability — Log-Log with Power-Law Trend")
    ax.legend(fontsize=9)
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
        bars = ax.bar(x + i * width, gaps, width,
                      label=ALGO_LABELS[algo], color=ALGO_COLORS[algo])
        _label_bars(ax, bars, fmt="{:.1f}%", fontsize=7)

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
            bars = ax.bar(x + i * width, vals, width,
                          label=ALGO_LABELS[algo] if tier == TIERS[0] else "",
                          color=ALGO_COLORS[algo])
            _label_bars(ax, bars, fmt="{:.0f}", fontsize=6.5)

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
        bars = ax.bar(x + i * width, improvements, width, label=label, color=color)
        _label_bars(ax, bars, fmt="{:.1f}%", fontsize=8)

    ax.set_xticks(x + 0.5 * width)
    ax.set_xticklabels([s.capitalize() for s in SCENARIOS])
    ax.set_ylabel("Mean % Improvement")
    ax.set_title("Improvement by Scenario Type (All Tiers)")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    _savefig(fig, "08_bar_scenario_improvement.png")


# ── Plot 9: GA improvement vs problem size (key story plot) ──────────────────

def plot_ga_improvement_vs_size(df):
    """
    Line plot showing % improvement of GA_Modified over GA_Original across
    all 3 scenarios and 3 tiers.  Shows that improvement grows with problem size.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    scenario_styles = {
        "balanced": {"linestyle": "-",  "marker": "o"},
        "scarce":   {"linestyle": "--", "marker": "s"},
        "rich":     {"linestyle": ":",  "marker": "^"},
    }
    scenario_colors = {
        "balanced": "#2E7D32",
        "scarce":   "#1565C0",
        "rich":     "#B71C1C",
    }

    size_order = ["small", "medium", "large"]
    size_labels = ["Small (20)", "Medium (100)", "Large (250)"]

    for scenario in SCENARIOS:
        improvements = []
        for tier in size_order:
            sub = df[(df["tier"] == tier) & (df["scenario"] == scenario)]
            gao = sub[sub["algorithm"] == "GA_Original"].sort_values(
                ["category", "instance_id"])["value"].values
            gam = sub[sub["algorithm"] == "GA_Modified"].sort_values(
                ["category", "instance_id"])["value"].values
            if len(gao) > 0 and len(gam) > 0:
                pct = np.nanmean(_safe_pct_improvement(gao, gam))
                improvements.append(pct)
            else:
                improvements.append(np.nan)

        ax.plot(size_labels, improvements,
                color=scenario_colors[scenario],
                linestyle=scenario_styles[scenario]["linestyle"],
                marker=scenario_styles[scenario]["marker"],
                linewidth=2.5, markersize=9,
                label=f"{scenario.capitalize()} scenario")

        # Annotate each point
        for i, v in enumerate(improvements):
            if not np.isnan(v):
                ax.annotate(f"{v:.1f}%", (i, v),
                            textcoords="offset points", xytext=(5, 6),
                            fontsize=9, color=scenario_colors[scenario])

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Problem Size (tier)")
    ax.set_ylabel("% Improvement: Hybrid GA over GA Original")
    ax.set_title("Hybrid GA Improvement Grows with Problem Size\n"
                 "(Why MMR seeding + elitism + threat mutation matter more at scale)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, "09_ga_improvement_vs_size.png")


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
    plot_ga_improvement_vs_size(df)

    print_summary(df)
    print("\nDone!")


if __name__ == "__main__":
    main()
