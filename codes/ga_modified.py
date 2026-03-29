"""
Modified Genetic Algorithm — Hybrid GA for WTA
================================================
Proposed improvement over the standard GA (Hasan & Barua, wta-theory.pdf).

Three independently-designed modifications, each addressing a distinct
weakness of the standard GA applied to the WTA problem:

Modification 1: MMR-Seeded Initialisation (Warm Start)
-------------------------------------------------------
The standard GA starts with a fully random population and wastes many early
generations rediscovering solutions that a simple greedy heuristic finds
instantly.  We seed one individual with the MMR greedy solution, giving the
population a strong starting reference point without sacrificing diversity
(all other individuals remain random).

Modification 2: Elitism with Stagnation Detection
--------------------------------------------------
The standard GA replaces the entire population each generation, risking
loss of the best solution found so far.  We preserve the top 10% of
individuals (elitism) to guarantee monotonic improvement.  Additionally,
we track a stagnation counter: if the global best has not improved for
STAGNATION_LIMIT generations, we inject fresh random individuals to
re-diversify the population and escape local optima.

Modification 3: Threat-Proportional Mutation (WTA-Specific)
------------------------------------------------------------
This is our key novel contribution.  Standard random mutation treats all
targets equally, but in WTA, targets with high remaining threat value
(V_j * survival_j) need more weapon coverage.  Instead of reassigning a
gene to a uniformly random target, we bias the mutation toward high-value
poorly-covered targets using remaining_threat[j] as selection weights.
This directly encodes WTA domain knowledge into the evolutionary operator,
guiding the search toward better regions of the solution space.
"""

import time
import numpy as np
from typing import Dict, Any, List

from wta_utils import compute_solution_value
from mmr_original import mmr_original
from ga_original import ga_budget, P_CROSSOVER, TOURNAMENT_K


# ── Modified GA Parameters ───────────────────────────────────────────────────

ELITE_FRAC       = 0.10   # preserve top 10% of population
STAGNATION_LIMIT = 5      # re-diversify after this many stagnant generations


# ── Operators (shared with ga_original where unchanged) ──────────────────────

def _random_individual(n_weapons, n_targets, rng):
    return rng.integers(0, n_targets, size=n_weapons).tolist()


def _tournament_select(population, fitnesses, k, rng):
    indices = rng.choice(len(population), size=min(k, len(population)), replace=False)
    best = min(indices, key=lambda i: fitnesses[i])
    return list(population[best])


def _uniform_crossover(p1, p2, rng):
    mask = rng.random(len(p1)) < 0.5
    c1 = [a if m else b for a, b, m in zip(p1, p2, mask)]
    c2 = [b if m else a for a, b, m in zip(p1, p2, mask)]
    return c1, c2


# ── Modification 3: Threat-Proportional Mutation ─────────────────────────────

def _compute_remaining_threat(allocation, target_values, kill_prob, n_targets):
    """Compute remaining_threat[j] = V_j * survival_j for the current allocation."""
    survival = np.ones(n_targets)
    for w, t in enumerate(allocation):
        if 0 <= t < n_targets:
            survival[t] *= (1.0 - kill_prob[w][t])
    tv = np.asarray(target_values, dtype=float)
    return tv * survival


def _threat_proportional_mutate(individual, target_values, kill_prob,
                                 n_targets, p_mutation, rng):
    """
    Threat-proportional mutation: when a gene mutates, the new target is
    chosen with probability proportional to remaining_threat[j], directing
    weapons toward high-value under-covered targets.
    """
    mutated = list(individual)
    # Only compute weights if at least one mutation will happen
    needs_mutation = rng.random(len(mutated)) < p_mutation
    if not np.any(needs_mutation):
        return mutated

    remaining = _compute_remaining_threat(mutated, target_values, kill_prob, n_targets)
    # Ensure positive weights (add small epsilon to avoid zero-sum)
    weights = remaining + 1e-12
    weights = weights / weights.sum()

    for i in range(len(mutated)):
        if needs_mutation[i]:
            mutated[i] = int(rng.choice(n_targets, p=weights))

    return mutated


# ── Main Modified GA function ────────────────────────────────────────────────

def ga_modified(instance: Dict[str, Any],
                time_budget_sec: float = None,
                seed: int = 42) -> Dict[str, Any]:
    """
    Run the Modified (Hybrid) GA on a (flat) WTA instance.

    Modifications:
        1. MMR-Seeded Initialisation
        2. Elitism with Stagnation Detection
        3. Threat-Proportional Mutation

    Parameters
    ----------
    instance : dict with keys n_weapons, n_targets, target_values, kill_prob
    time_budget_sec : max wall-clock seconds (default: auto-scaled)
    seed : RNG seed

    Returns
    -------
    dict with allocation, value, time_sec, iterations, mmr_seed_value,
         improvement_pct
    """
    n_weapons     = instance["n_weapons"]
    n_targets     = instance["n_targets"]
    target_values = instance["target_values"]
    kill_prob     = instance["kill_prob"]

    if time_budget_sec is None:
        time_budget_sec = ga_budget(n_weapons, n_targets)

    rng = np.random.default_rng(seed)

    pop_size    = max(n_weapons, n_targets)
    elite_count = max(2, int(pop_size * ELITE_FRAC))
    p_mutation  = 1.0 / max(n_weapons, 1)

    t0 = time.perf_counter()

    # ── Modification 1: MMR-Seeded Initialisation ────────────────────────
    mmr_result = mmr_original(instance)
    mmr_alloc  = mmr_result["allocation"]
    mmr_value  = mmr_result["value"]

    population = [list(mmr_alloc)]          # individual 0 = MMR solution
    for _ in range(pop_size - 1):
        population.append(_random_individual(n_weapons, n_targets, rng))

    fitnesses = [compute_solution_value(ind, target_values, kill_prob, n_targets)
                 for ind in population]

    best_idx   = int(np.argmin(fitnesses))
    best_alloc = list(population[best_idx])
    best_value = fitnesses[best_idx]

    # ── Evolutionary loop ────────────────────────────────────────────────
    generations      = 0
    stagnation_count = 0
    end_time         = t0 + time_budget_sec

    while time.perf_counter() < end_time:
        prev_best = best_value

        # ── Modification 2: Elitism ──────────────────────────────────────
        sorted_idx = np.argsort(fitnesses)
        elites     = [list(population[i]) for i in sorted_idx[:elite_count]]
        elite_fits = [fitnesses[i]        for i in sorted_idx[:elite_count]]

        new_pop  = list(elites)
        new_fits = list(elite_fits)

        # Generate offspring to fill remaining slots
        while len(new_pop) < pop_size:
            p1 = _tournament_select(population, fitnesses, TOURNAMENT_K, rng)
            p2 = _tournament_select(population, fitnesses, TOURNAMENT_K, rng)

            if rng.random() < P_CROSSOVER:
                c1, c2 = _uniform_crossover(p1, p2, rng)
            else:
                c1, c2 = list(p1), list(p2)

            # ── Modification 3: Threat-Proportional Mutation ─────────────
            c1 = _threat_proportional_mutate(
                c1, target_values, kill_prob, n_targets, p_mutation, rng)
            c2 = _threat_proportional_mutate(
                c2, target_values, kill_prob, n_targets, p_mutation, rng)

            new_pop.append(c1)
            new_fits.append(
                compute_solution_value(c1, target_values, kill_prob, n_targets))
            if len(new_pop) < pop_size:
                new_pop.append(c2)
                new_fits.append(
                    compute_solution_value(c2, target_values, kill_prob, n_targets))

        new_pop  = new_pop[:pop_size]
        new_fits = new_fits[:pop_size]

        # Update global best
        gen_best = int(np.argmin(new_fits))
        if new_fits[gen_best] < best_value:
            best_value = new_fits[gen_best]
            best_alloc = list(new_pop[gen_best])

        population = new_pop
        fitnesses  = new_fits
        generations += 1

        # ── Modification 2b: Stagnation Detection ───────────────────────
        if best_value < prev_best - 1e-12:
            stagnation_count = 0
        else:
            stagnation_count += 1

        if stagnation_count >= STAGNATION_LIMIT:
            # Inject fresh random individuals (replace worst, keep elites)
            for idx in sorted_idx[-elite_count:]:
                if idx < len(population):
                    population[idx] = _random_individual(n_weapons, n_targets, rng)
                    fitnesses[idx]  = compute_solution_value(
                        population[idx], target_values, kill_prob, n_targets)
            stagnation_count = 0

    elapsed = time.perf_counter() - t0

    improvement_pct = (
        100.0 * (mmr_value - best_value) / mmr_value
        if mmr_value > 0 else 0.0
    )

    return {
        "allocation":      best_alloc,
        "value":           best_value,
        "time_sec":        elapsed,
        "iterations":      generations,
        "mmr_seed_value":  mmr_value,
        "improvement_pct": improvement_pct,
    }


# ── smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from mmr_original import mmr_original as _mmr
    from ga_original  import ga_original  as _ga_orig

    inst = {
        "n_weapons":     3,
        "n_targets":     3,
        "target_values": [10, 20, 30],
        "kill_prob": [
            [0.3, 0.2, 0.1],
            [0.1, 0.4, 0.2],
            [0.2, 0.3, 0.5],
        ],
    }
    budget = 2.0

    mmr  = _mmr(inst)
    orig = _ga_orig(inst, time_budget_sec=budget)
    mod  = ga_modified(inst, time_budget_sec=budget)

    print("MMR Original  :", round(mmr["value"], 4))
    print("GA  Original  :", round(orig["value"], 4), f"({orig['iterations']} gens)")
    print("GA  Modified  :", round(mod["value"], 4),  f"({mod['iterations']} gens)")
    print(f"  MMR seed val : {mod['mmr_seed_value']:.4f}")
    print(f"  Improvement  : {mod['improvement_pct']:.2f}%")
