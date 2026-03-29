"""
WTA Benchmark Dataset Generator
=================================
Generates synthetic Weapon-Target Assignment (WTA) problem instances.

Justification for synthetic data:
    No standardized, publicly available WTA benchmark datasets exist in the
    literature. Prior studies (Ahuja et al. 2007, Lee et al. 2003, Manne 1958)
    each used proprietary instances.  Following common practice in
    combinatorial-optimization research, we generate random instances whose
    statistical properties match those reported in the surveyed literature.

Dataset design — 3 tiers x 3 scenarios:
    Size tiers (target dimension):
        small  (20)  — fast to run, good for debugging and validating correctness
        medium (100) — moderate computational cost, reveals algorithmic trends
        large  (250) — stress-tests scalability, differentiates heuristics

    Weapon-target scenarios (within each tier):
        balanced (W = T)  — each weapon can be matched one-to-one
        scarce   (W < T)  — fewer weapons than targets; hardest case, most realistic
        rich     (W > T)  — more weapons than targets; concentrated fire possible

    Flat weapon model: each weapon is an individual unit with its own kill
    probabilities.  No weapon types or quantities -- simpler and directly
    consumed by all algorithms.

Total: 9 categories x 20 instances = 180 instances.
"""

import numpy as np
import json
import os


# Reproducibility
RNG_SEED = 42

# ── Category definitions ─────────────────────────────────────────────────────

CATEGORIES = [
    # Small tier: target dimension = 20
    {"name": "small_balanced",  "weapons": 20,  "targets": 20,  "count": 20,
     "tier": "small",  "scenario": "balanced"},
    {"name": "small_scarce",    "weapons": 10,  "targets": 20,  "count": 20,
     "tier": "small",  "scenario": "scarce"},
    {"name": "small_rich",      "weapons": 30,  "targets": 20,  "count": 20,
     "tier": "small",  "scenario": "rich"},

    # Medium tier: target dimension = 100
    {"name": "medium_balanced", "weapons": 100, "targets": 100, "count": 20,
     "tier": "medium", "scenario": "balanced"},
    {"name": "medium_scarce",   "weapons": 50,  "targets": 100, "count": 20,
     "tier": "medium", "scenario": "scarce"},
    {"name": "medium_rich",     "weapons": 150, "targets": 100, "count": 20,
     "tier": "medium", "scenario": "rich"},

    # Large tier: target dimension = 250
    {"name": "large_balanced",  "weapons": 250, "targets": 250, "count": 20,
     "tier": "large",  "scenario": "balanced"},
    {"name": "large_scarce",    "weapons": 125, "targets": 250, "count": 20,
     "tier": "large",  "scenario": "scarce"},
    {"name": "large_rich",      "weapons": 375, "targets": 250, "count": 20,
     "tier": "large",  "scenario": "rich"},
]

TARGET_VALUE_RANGE = (10, 100)     # integer threat/value scores
KILL_PROB_RANGE    = (0.05, 0.90)  # kill probabilities p_ij


def generate_instance(n_weapons: int, n_targets: int, rng: np.random.Generator) -> dict:
    """
    Generate one flat WTA instance.

    Parameters:
        n_weapons : W  -- number of individual weapons
        n_targets : T  -- number of targets
        kill_prob[i][j] : p_ij -- P(weapon i destroys target j)

    kill_prob shape is (n_weapons x n_targets) -- one row per weapon.
    """
    target_values = rng.integers(
        TARGET_VALUE_RANGE[0], TARGET_VALUE_RANGE[1] + 1,
        size=n_targets
    ).tolist()

    kill_prob = rng.uniform(
        KILL_PROB_RANGE[0], KILL_PROB_RANGE[1],
        size=(n_weapons, n_targets)
    ).tolist()

    return {
        "n_weapons":      n_weapons,
        "n_targets":      n_targets,
        "target_values":  target_values,
        "kill_prob":      kill_prob,
    }


def generate_all_instances(output_dir: str = "datasets") -> dict:
    """Generate all benchmark instances and save them."""
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    metadata = {
        "seed": RNG_SEED,
        "description": (
            "Synthetic WTA benchmark instances. "
            f"Kill probs ~ U({KILL_PROB_RANGE[0]}, {KILL_PROB_RANGE[1]}). "
            f"Target values ~ U({TARGET_VALUE_RANGE[0]}, {TARGET_VALUE_RANGE[1]}). "
            "Includes balanced (W=T), weapon-scarce (W<T), and weapon-rich (W>T) scenarios."
        ),
        "categories": [],
        "total_instances": 0,
    }

    for cat in CATEGORIES:
        cat_dir = os.path.join(output_dir, cat["name"])
        os.makedirs(cat_dir, exist_ok=True)
        instances = []

        for idx in range(cat["count"]):
            inst = generate_instance(cat["weapons"], cat["targets"], rng)
            fname = f"instance_{idx+1:03d}.json"
            fpath = os.path.join(cat_dir, fname)
            with open(fpath, "w") as f:
                json.dump(inst, f, separators=(",", ":"))
            instances.append(fname)

        metadata["categories"].append({
            "name":      cat["name"],
            "weapons":   cat["weapons"],
            "targets":   cat["targets"],
            "count":     cat["count"],
            "tier":      cat["tier"],
            "scenario":  cat["scenario"],
            "directory": cat_dir,
            "files":     instances,
        })
        metadata["total_instances"] += cat["count"]
        print(f"  [{cat['scenario']:8s}] {cat['name']:18s} "
              f"W={cat['weapons']:3d} T={cat['targets']:3d} "
              f"-> {cat['count']} instances")

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved -> {meta_path}")
    print(f"Total instances: {metadata['total_instances']}")
    return metadata


def load_instance(path: str) -> dict:
    """Load a single WTA instance from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    data["kill_prob"] = [list(row) for row in data["kill_prob"]]
    return data


if __name__ == "__main__":
    print("Generating WTA benchmark dataset ...")
    generate_all_instances()
