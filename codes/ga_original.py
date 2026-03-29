"""
Original Genetic Algorithm (GA) for WTA
========================================
Faithfully implements the GA pseudocode from:
    Hasan & Barua, "Weapon-Target Assignment Problem", wta-theory.pdf, pp. 11-12.

Algorithm logic (time-based evolutionary loop):
    1. Generate a random initial population of size max(W, T).
    2. Evaluate fitness of every individual (= WTA objective value; lower is better).
    3. While time budget not exhausted:
        a. Select parents via tournament selection (k=3).
        b. Apply uniform crossover with rate p_c = 0.8.
        c. Apply random-reassignment mutation with rate p_m = 1/W.
        d. Replace population with offspring.
        e. Track the best solution found so far.
    4. Return the best solution.

Encoding:
    chromosome[i] = target index (0..T-1) assigned to weapon i.
    Length = n_weapons (individual weapon tokens after expand_instance).

Complexity per generation: O(pop_size * W) for fitness evaluation.
"""

import time
import numpy as np
from typing import Dict, Any, List

from wta_utils import compute_solution_value


# ── GA Parameters (from Hasan & Barua) ──────────────────────────────────────

P_CROSSOVER  = 0.8      # crossover probability
TOURNAMENT_K = 3        # tournament selection size


def ga_budget(n_weapons: int, n_targets: int) -> float:
    """Time budget in seconds, scaled by problem size."""
    size = max(n_weapons, n_targets)
    if size <= 30:
        return 5.0
    if size <= 120:
        return 20.0
    return 60.0


# ── Core GA operators ────────────────────────────────────────────────────────

def _random_individual(n_weapons: int, n_targets: int, rng) -> List[int]:
    """Create one random allocation (chromosome)."""
    return rng.integers(0, n_targets, size=n_weapons).tolist()


def _tournament_select(population, fitnesses, k, rng):
    """Select one individual via tournament selection (minimisation)."""
    indices = rng.choice(len(population), size=min(k, len(population)), replace=False)
    best = min(indices, key=lambda i: fitnesses[i])
    return list(population[best])


def _uniform_crossover(p1, p2, rng):
    """Uniform crossover: each gene independently from one parent."""
    mask = rng.random(len(p1)) < 0.5
    c1 = [a if m else b for a, b, m in zip(p1, p2, mask)]
    c2 = [b if m else a for a, b, m in zip(p1, p2, mask)]
    return c1, c2


def _mutate(individual, n_targets, p_mutation, rng):
    """Random-reassignment mutation: each gene flips with probability p_m."""
    mutated = list(individual)
    for i in range(len(mutated)):
        if rng.random() < p_mutation:
            mutated[i] = int(rng.integers(0, n_targets))
    return mutated


# ── Main GA function ─────────────────────────────────────────────────────────

def ga_original(instance: Dict[str, Any],
                time_budget_sec: float = None,
                seed: int = 42) -> Dict[str, Any]:
    """
    Run the original GA on a (flat) WTA instance.

    Parameters
    ----------
    instance : dict with keys n_weapons, n_targets, target_values, kill_prob
    time_budget_sec : max wall-clock seconds (default: auto-scaled)
    seed : RNG seed for reproducibility

    Returns
    -------
    dict with allocation, value, time_sec, iterations (generations)
    """
    n_weapons     = instance["n_weapons"]
    n_targets     = instance["n_targets"]
    target_values = instance["target_values"]
    kill_prob     = instance["kill_prob"]

    if time_budget_sec is None:
        time_budget_sec = ga_budget(n_weapons, n_targets)

    rng = np.random.default_rng(seed)

    # Population size = max(W, T) per Hasan & Barua
    pop_size   = max(n_weapons, n_targets)
    p_mutation = 1.0 / max(n_weapons, 1)

    # ── Initialise population ────────────────────────────────────────────
    population = [_random_individual(n_weapons, n_targets, rng)
                  for _ in range(pop_size)]
    fitnesses  = [compute_solution_value(ind, target_values, kill_prob, n_targets)
                  for ind in population]

    best_idx   = int(np.argmin(fitnesses))
    best_alloc = list(population[best_idx])
    best_value = fitnesses[best_idx]

    # ── Evolutionary loop (time-based) ───────────────────────────────────
    generations = 0
    t0       = time.perf_counter()
    end_time = t0 + time_budget_sec

    while time.perf_counter() < end_time:
        new_pop = []

        # Generate offspring
        while len(new_pop) < pop_size:
            parent1 = _tournament_select(population, fitnesses, TOURNAMENT_K, rng)
            parent2 = _tournament_select(population, fitnesses, TOURNAMENT_K, rng)

            if rng.random() < P_CROSSOVER:
                child1, child2 = _uniform_crossover(parent1, parent2, rng)
            else:
                child1, child2 = list(parent1), list(parent2)

            child1 = _mutate(child1, n_targets, p_mutation, rng)
            child2 = _mutate(child2, n_targets, p_mutation, rng)

            new_pop.extend([child1, child2])

        new_pop  = new_pop[:pop_size]
        new_fits = [compute_solution_value(ind, target_values, kill_prob, n_targets)
                    for ind in new_pop]

        # Track best
        gen_best = int(np.argmin(new_fits))
        if new_fits[gen_best] < best_value:
            best_value = new_fits[gen_best]
            best_alloc = list(new_pop[gen_best])

        population = new_pop
        fitnesses  = new_fits
        generations += 1

    elapsed = time.perf_counter() - t0

    return {
        "allocation": best_alloc,
        "value":      best_value,
        "time_sec":   elapsed,
        "iterations": generations,
    }


# ── smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
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
    result = ga_original(inst, time_budget_sec=2.0)
    print("GA Original - smoke test")
    print(f"  Allocation : {result['allocation']}")
    print(f"  Value      : {result['value']:.4f}")
    print(f"  Generations: {result['iterations']}")
    print(f"  Time       : {result['time_sec']:.3f}s")
