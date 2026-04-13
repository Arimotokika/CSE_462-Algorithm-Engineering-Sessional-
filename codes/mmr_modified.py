"""
Modified MMR Algorithm — MMR with Iterative Re-evaluation (MMR-IR)
===================================================================
Proposed improvement over the original MMR (Hasan & Barua, wta.pdf).

Motivation / Rationale
----------------------
The original MMR is a pure greedy algorithm: once a weapon is assigned and
the target value is updated, no weapon assignment is ever reconsidered.  This
leads to suboptimal solutions when an early high-marginal-return assignment
prevents a better global configuration.

Proposed Modifications
----------------------
1. Tie-breaking by Target Priority:
   Original MMR breaks ties arbitrarily.  We break ties in favour of the
   highest current-value target, concentrating early fire on the most
   threatening targets that remain under-covered at that step.

2. Iterative 1-opt Local Search:
   After the greedy pass, for each weapon try reassigning to every other
   target; accept if it improves the objective.  Repeat until convergence.
   Uses incremental evaluation (O(1) per trial) for large-instance speed.

3. 2-opt Swap Search:
   After 1-opt converges, try swapping the targets of two weapons.  This
   explores a larger neighbourhood and can escape 1-opt local optima.

Why this helps:
    - Weapons assigned early in the greedy pass may have been placed on
      targets that later received many weapons, diluting their impact.
    - 1-opt corrects single-weapon misplacements.
    - 2-opt corrects paired misplacements that 1-opt cannot find.
"""

import time
import numpy as np
from typing import List, Dict, Any

from wta_utils import compute_solution_value


def _greedy_pass(
    n_weapons:     int,
    n_targets:     int,
    target_values: List[float],
    kill_prob:     List[List[float]],
) -> List[int]:
    """Original MMR greedy pass with tie-break enhancement — returns allocation list."""
    current_values      = list(map(float, target_values))
    unallocated_weapons = list(range(n_weapons))
    allocation          = [-1] * n_weapons
    allocated_count     = 0

    while allocated_count < n_weapons:
        max_decrease = float("-inf")
        best_weapon  = -1
        best_target  = -1

        for k in unallocated_weapons:
            for i in range(n_targets):
                decrease = current_values[i] * kill_prob[k][i]
                # Break ties by the current residual target value, not the
                # original static target value.
                if (decrease > max_decrease) or (
                    abs(decrease - max_decrease) < 1e-12
                    and best_target >= 0
                    and current_values[i] > current_values[best_target]
                ):
                    max_decrease = decrease
                    best_target  = i
                    best_weapon  = k

        unallocated_weapons.remove(best_weapon)
        allocation[best_weapon]      = best_target
        current_values[best_target] -= max_decrease
        allocated_count             += 1

    return allocation


def _build_survival(allocation, kill_prob, n_targets):
    """Build per-target survival array from an allocation."""
    survival = np.ones(n_targets)
    for w, t in enumerate(allocation):
        if t >= 0:
            survival[t] *= (1.0 - kill_prob[w][t])
    return survival


def _local_search_1opt(
    allocation:    List[int],
    target_values: List[float],
    kill_prob:     List[List[float]],
    n_targets:     int,
    max_passes:    int = 20,
) -> tuple:
    """
    Incremental 1-opt local search.

    Instead of recomputing the full objective for each trial, we maintain
    a survival array and compute the delta from reassigning one weapon.
    This is O(1) per trial instead of O(W), critical for large instances.

    Returns (improved_allocation, improved_value, passes_done).
    """
    best_alloc = list(allocation)
    n_weapons  = len(allocation)
    tv         = np.asarray(target_values, dtype=float)

    # Build survival array
    survival = _build_survival(best_alloc, kill_prob, n_targets)
    best_val = float(np.dot(tv, survival))

    for pass_no in range(max_passes):
        improved = False
        for w in range(n_weapons):
            curr_t = best_alloc[w]
            p_curr = kill_prob[w][curr_t]

            # Survival of curr_t WITHOUT weapon w
            if abs(1.0 - p_curr) < 1e-15:
                # Avoid division by zero when p = 1
                surv_curr_without = 0.0
            else:
                surv_curr_without = survival[curr_t] / (1.0 - p_curr)

            best_delta  = 0.0
            best_new_t  = -1

            for new_t in range(n_targets):
                if new_t == curr_t:
                    continue
                # Delta from removing w from curr_t
                delta_curr = tv[curr_t] * (surv_curr_without - survival[curr_t])
                # Delta from adding w to new_t
                new_surv_new_t = survival[new_t] * (1.0 - kill_prob[w][new_t])
                delta_new  = tv[new_t] * (new_surv_new_t - survival[new_t])

                total_delta = delta_curr + delta_new
                if total_delta < best_delta - 1e-12:
                    best_delta = total_delta
                    best_new_t = new_t

            if best_new_t >= 0:
                # Commit the move
                survival[curr_t] = surv_curr_without
                survival[best_new_t] *= (1.0 - kill_prob[w][best_new_t])
                best_alloc[w] = best_new_t
                best_val += best_delta
                improved = True

        if not improved:
            return best_alloc, best_val, pass_no + 1

    return best_alloc, best_val, max_passes


def _local_search_2opt(
    allocation:    List[int],
    target_values: List[float],
    kill_prob:     List[List[float]],
    n_targets:     int,
    max_passes:    int = 3,
) -> tuple:
    """
    2-opt swap search: try swapping targets of two weapons.

    Limited to max_passes to control runtime on large instances.
    Returns (improved_allocation, improved_value, passes_done).
    """
    best_alloc = list(allocation)
    n_weapons  = len(allocation)
    tv         = np.asarray(target_values, dtype=float)

    survival = _build_survival(best_alloc, kill_prob, n_targets)
    best_val = float(np.dot(tv, survival))

    for pass_no in range(max_passes):
        improved = False

        for w1 in range(n_weapons):
            t1 = best_alloc[w1]
            for w2 in range(w1 + 1, n_weapons):
                t2 = best_alloc[w2]
                if t1 == t2:
                    continue

                # Compute delta for swapping: w1->t2, w2->t1
                # Remove w1 from t1 and w2 from t2
                s_t1 = survival[t1]
                s_t2 = survival[t2]

                p_w1_t1 = kill_prob[w1][t1]
                p_w2_t2 = kill_prob[w2][t2]
                p_w1_t2 = kill_prob[w1][t2]
                p_w2_t1 = kill_prob[w2][t1]

                # New survival for t1: remove w1, add w2
                if abs(1.0 - p_w1_t1) < 1e-15:
                    new_s_t1 = 0.0
                else:
                    new_s_t1 = (s_t1 / (1.0 - p_w1_t1)) * (1.0 - p_w2_t1)

                # New survival for t2: remove w2, add w1
                if abs(1.0 - p_w2_t2) < 1e-15:
                    new_s_t2 = 0.0
                else:
                    new_s_t2 = (s_t2 / (1.0 - p_w2_t2)) * (1.0 - p_w1_t2)

                delta = (tv[t1] * (new_s_t1 - s_t1) +
                         tv[t2] * (new_s_t2 - s_t2))

                if delta < -1e-12:
                    # Commit the swap
                    survival[t1] = new_s_t1
                    survival[t2] = new_s_t2
                    best_alloc[w1] = t2
                    best_alloc[w2] = t1
                    best_val += delta
                    improved = True

            # Early time check after each w1 (2-opt is O(W^2))
            if improved:
                break  # restart pass from the beginning

        if not improved:
            return best_alloc, best_val, pass_no + 1

    return best_alloc, best_val, max_passes


def mmr_modified(instance: Dict[str, Any], max_ls_passes: int = 20) -> Dict[str, Any]:
    """
    Run the Modified MMR (MMR-IR) algorithm on a WTA instance.

    Parameters
    ----------
    instance      : dict -- same format as mmr_original
    max_ls_passes : int  -- maximum 1-opt local-search passes (default 20)

    Returns
    -------
    dict with keys
        allocation      : list[int]
        value           : float
        time_sec        : float
        greedy_value    : float  -- value after greedy phase (before LS)
        ls_passes       : int    -- number of local-search passes (1-opt + 2-opt)
        improvement_pct : float  -- % improvement from local search
    """
    n_weapons     = instance["n_weapons"]
    n_targets     = instance["n_targets"]
    target_values = list(instance["target_values"])
    kill_prob     = instance["kill_prob"]

    t0 = time.perf_counter()

    # Phase 1: greedy pass (same as original MMR, with tie-break enhancement)
    greedy_alloc = _greedy_pass(n_weapons, n_targets, target_values, kill_prob)
    greedy_value = compute_solution_value(greedy_alloc, target_values, kill_prob, n_targets)

    # Phase 2: iterative 1-opt local search (incremental evaluation)
    alloc_1opt, val_1opt, passes_1opt = _local_search_1opt(
        greedy_alloc, target_values, kill_prob, n_targets, max_ls_passes
    )

    # Phase 3: 2-opt swap search
    final_alloc, final_value, passes_2opt = _local_search_2opt(
        alloc_1opt, target_values, kill_prob, n_targets, max_passes=3
    )

    elapsed = time.perf_counter() - t0

    improvement_pct = (
        100.0 * (greedy_value - final_value) / greedy_value
        if greedy_value > 0 else 0.0
    )

    return {
        "allocation":      final_alloc,
        "value":           final_value,
        "time_sec":        elapsed,
        "greedy_value":    greedy_value,
        "ls_passes":       passes_1opt + passes_2opt,
        "improvement_pct": improvement_pct,
    }


# ── smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from mmr_original import mmr_original

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
    orig = mmr_original(inst)
    mod  = mmr_modified(inst)

    print("MMR Original  value:", round(orig["value"], 4))
    print("MMR Modified  value:", round(mod["value"], 4))
    print(f"  Greedy value      : {mod['greedy_value']:.4f}")
    print(f"  LS passes         : {mod['ls_passes']}")
    print(f"  Improvement       : {mod['improvement_pct']:.2f}%")
