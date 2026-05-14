"""evaluate_baselines.py
========================

Standalone evaluation for the GA / AVNR / Greedy baselines on the four
paper tiers, with optional LTL constraints.

Each baseline is executed inside the same discrete-event simulator and
the same LTL shield used by ``evaluate_rl.py``; ``num_envs`` random
instances per tier are evaluated and metrics are aggregated as
``mean ± std`` (matching the paper's reporting format).

The three baselines integrate with the simulator as follows:

- **Greedy**: runs entirely inside the simulator, querying the
  three-mask pipeline at every decision epoch. Inherently LTL-aware.
- **GA**: every chromosome is evaluated by simulating its routing
  through the shielded simulator (LTL-violating tasks are skipped, the
  completion-rate drop enters the fitness as a hard penalty). Inherently
  LTL-aware.
- **AVNR**: variable-neighbourhood search uses an analytical
  multi-trip-time surrogate to score thousands of neighbours per
  iteration (full simulator would be prohibitive); the best returned
  solution is then evaluated through the same shielded simulator that
  the other methods use. AVNR's search is not LTL-aware, but its
  reported metrics are LTL-shielded just like the other baselines.

Usage
-----

::

    # GA on Tier 1, 30 random instances, no LTL
    python evaluate_baselines.py --method ga --tier tier1 --num-envs 30

    # Greedy + GA + AVNR on Tier 1, with LTL clauses
    python evaluate_baselines.py --method ga,avnr,greedy --tier tier1 --num-envs 30 --enable-ltl

The script writes a JSON results file (timestamped by default).
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, List

import numpy as np

from env.task_env import TaskEnv
from env.ltl_utils import LTLMonitor
from parameters import EnvParams

from baseline.simulator import (
    BENCHMARK_CONFIGS,
    TIME_LIMITS,
    GA_PARAMS,
    AVNR_PARAMS,
    generate_fixed_ltl_constraints,
    calculate_solution_metrics,
)
from baseline.MILP_GA_GreenVRP import GeneticAlgorithm, GreenVRPInstance
from baseline.AVNR_Metaheuristic import AVNRSolver
from baseline.greedy_solver import GreedySolver


# ---------------------------------------------------------------------------
# Env construction
# ---------------------------------------------------------------------------


def _build_tier_env(tier: str, seed: int, enable_ltl: bool):
    """Create a fresh TaskEnv at the paper's tier dimensions, seeded by
    ``seed``. Returns (env, ltl_monitor)."""

    cfg = BENCHMARK_CONFIGS[tier]
    env = TaskEnv(
        per_species_range=(cfg["agents_per_species"], cfg["agents_per_species"]),
        species_range=(cfg["species"], cfg["species"]),
        tasks_range=(cfg["tasks"], cfg["tasks"]),
        depot_num_range=(cfg["depots"], cfg["depots"]),
        traits_dim=EnvParams.TRAIT_DIM,
        decision_dim=EnvParams.DECISION_DIM,
        max_task_size=EnvParams.MAX_TASK_SIZE,
        max_cargo_per_type=EnvParams.MAX_AGENT_CAPACITY,
        duration_scale=EnvParams.DURATION_SCALE,
        seed=seed,
        plot_figure=False,
    )
    env.init_state()

    ltl_monitor = None
    if enable_ltl:
        clauses = generate_fixed_ltl_constraints(
            env,
            num_safety=cfg["ltl_safety"],
            num_sequential=cfg["ltl_sequential"],
            seed=seed,
        )
        ltl_monitor = LTLMonitor(clauses)
    return env, ltl_monitor


# ---------------------------------------------------------------------------
# Per-method runners (one instance each)
# ---------------------------------------------------------------------------


def _run_ga(env, ltl_monitor, tier: str, time_limit: float, decide_quantity: bool):
    instance = GreenVRPInstance.from_env(env)
    ga_params = GA_PARAMS[tier]
    ga = GeneticAlgorithm(
        instance,
        pop_size=ga_params["pop_size"],
        max_generations=ga_params["max_generations"],
        crossover_rate=0.8,
        mutation_rate=0.2,
        niching_radius=0.1,
        time_limit=time_limit,
        env=env,
        ltl_monitor=ltl_monitor,
        decide_quantity=decide_quantity,
    )
    t0 = time.time()
    best_solution = ga.run()
    elapsed = time.time() - t0
    return calculate_solution_metrics(
        env=env,
        solution=best_solution,
        ltl_monitor=ltl_monitor,
        method_name="GA",
        elapsed_time=elapsed,
        decide_quantity=decide_quantity,
        use_in_flight_reservation=False,
    )


def _run_avnr(env, ltl_monitor, tier: str, time_limit: float, decide_quantity: bool):
    instance = GreenVRPInstance.from_env(env)
    avnr_params = AVNR_PARAMS[tier]
    solver = AVNRSolver(
        instance,
        max_iterations=avnr_params["max_iterations"],
        max_time=time_limit,
        shaking_strength=3,
        env=env,
        ltl_monitor=ltl_monitor,
    )
    t0 = time.time()
    best_solution_obj = solver.solve()
    elapsed = time.time() - t0
    # AVNRSolution -> dict-shaped solution (GA format) for the shared metric path
    best_solution = best_solution_obj.convert_to_ga_format()
    return calculate_solution_metrics(
        env=env,
        solution=best_solution,
        ltl_monitor=ltl_monitor,
        method_name="AVNR",
        elapsed_time=elapsed,
        decide_quantity=decide_quantity,
        use_in_flight_reservation=False,
    )


def _run_greedy(env, ltl_monitor, tier: str, time_limit: float, decide_quantity: bool):
    # Greedy runs inside the simulator directly; decide_quantity / time_limit
    # are not exposed (kept in signature for symmetry).
    env.rng = np.random.default_rng(seed=42)
    solver = GreedySolver(env, ltl_monitor=ltl_monitor, seed=42)
    t0 = time.time()
    g = solver.solve()
    elapsed = time.time() - t0
    metrics = {
        "method": "Greedy",
        "success": g["success"],
        "makespan": g["makespan"],
        "travel_distance": g["total_distance"],
        "time_cost": g["makespan"],
        "task_completion_rate": (
            g["completed_tasks"] / env.tasks_num if env.tasks_num > 0 else 0.0
        ),
        "num_vehicles_used": sum(
            1 for a in env.agent_dic.values() if a.get("travel_dist", 0) > 0
        ),
        "vehicle_utilization": (
            sum(1 for a in env.agent_dic.values() if a.get("travel_dist", 0) > 0)
            / max(env.agents_num, 1)
        ),
        "load_balance_std": 0.0,
        "solving_time": elapsed,
        "ltl_satisfaction_rate": 1.0,
        "ltl_violations_count": 0,
        "ltl_total_constraints": 0,
    }
    if ltl_monitor is not None:
        stats = ltl_monitor.get_statistics()
        metrics["ltl_total_constraints"] = stats.get("num_clauses", 0)
        metrics["ltl_satisfaction_rate"] = (
            stats.get("overall_satisfaction_rate", 0.0) * 100.0
        )
        metrics["ltl_violations_count"] = (
            stats.get("safety_violated", 0)
            + stats.get("sequential_initial", 0)
            + stats.get("sequential_violated", 0)
        )
    return metrics


_RUNNERS = {"ga": _run_ga, "avnr": _run_avnr, "greedy": _run_greedy}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in results if r.get("success")]
    n_valid = len(valid)
    summary: Dict[str, Any] = {
        "num_evaluated": len(results),
        "num_succeeded": n_valid,
        "success_rate": n_valid / max(len(results), 1),
    }

    def _agg(key: str):
        vals = [r[key] for r in valid if r.get(key) not in (None, float("inf"))]
        if not vals:
            return None, None
        return float(np.mean(vals)), float(np.std(vals))

    for key in ("makespan", "travel_distance", "task_completion_rate", "solving_time"):
        m, s = _agg(key)
        summary[f"{key}_mean"] = m
        summary[f"{key}_std"] = s

    if any(r.get("ltl_total_constraints", 0) > 0 for r in results):
        m, s = _agg("ltl_satisfaction_rate")
        summary["ltl_satisfaction_rate_mean"] = m
        summary["ltl_satisfaction_rate_std"] = s
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GA / AVNR / Greedy baselines on paper tiers."
    )
    parser.add_argument(
        "--method", default="ga", help="Comma-separated subset of {ga, avnr, greedy}."
    )
    parser.add_argument(
        "--tier",
        required=True,
        choices=sorted(BENCHMARK_CONFIGS),
        help="Paper difficulty tier.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=30,
        help="Number of random benchmark instances to evaluate per method.",
    )
    parser.add_argument(
        "--enable-ltl",
        action="store_true",
        help="Generate fixed LTL clauses from the tier config and enforce "
        "them through the shared LTL shield.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Override per-instance time limit (seconds). Defaults to "
        "TIME_LIMITS for the chosen tier.",
    )
    parser.add_argument(
        "--decide-quantity",
        action="store_true",
        help="When set, GA / AVNR use the quantity field of their solution. "
        "When unset (default and matches the paper), LOAD always uses the "
        "agent's full capacity for the chosen cargo type (consistent with "
        "evaluate_rl.py's force-full-load policy).",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=1000,
        help="Base seed for synthesised benchmark seeds.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Where to write per-instance results (default: timestamped).",
    )
    args = parser.parse_args()

    methods = [m.strip().lower() for m in args.method.split(",") if m.strip()]
    for m in methods:
        if m not in _RUNNERS:
            parser.error(f"Unknown --method '{m}'. Choose from {sorted(_RUNNERS)}.")

    time_limit = (
        args.time_limit if args.time_limit is not None else TIME_LIMITS[args.tier]
    )
    rng = np.random.default_rng(args.seed_base)
    seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(args.num_envs)]

    print(
        f"[evaluate_baselines] tier={args.tier} methods={methods} "
        f"num_envs={args.num_envs} enable_ltl={args.enable_ltl} "
        f"time_limit={time_limit}s decide_quantity={args.decide_quantity}"
    )

    per_method_results: Dict[str, List[Dict[str, Any]]] = {m: [] for m in methods}

    for i, seed in enumerate(seeds):
        for m in methods:
            print(f"\n[{i + 1:>3}/{args.num_envs}] method={m} seed={seed}")
            env, ltl_monitor = _build_tier_env(args.tier, seed, args.enable_ltl)
            try:
                metrics = _RUNNERS[m](
                    env,
                    ltl_monitor,
                    args.tier,
                    time_limit=time_limit,
                    decide_quantity=args.decide_quantity,
                )
                metrics["seed"] = seed
                per_method_results[m].append(metrics)
                ms = (
                    f"{metrics['makespan']:.1f}"
                    if metrics["makespan"] != float("inf")
                    else "inf"
                )
                print(
                    f"  -> success={metrics['success']} "
                    f"completion={metrics['task_completion_rate']:.1%} "
                    f"makespan={ms}"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  FAILED: {type(exc).__name__}: {exc}")
                per_method_results[m].append(
                    {"seed": seed, "error": f"{type(exc).__name__}: {exc}"}
                )

    # Aggregate
    print("\n" + "=" * 60)
    print(f"Aggregate summary  (tier={args.tier}, num_envs={args.num_envs})")
    print("=" * 60)
    summaries: Dict[str, Dict[str, Any]] = {}
    for m in methods:
        s = _aggregate(per_method_results[m])
        summaries[m] = s
        print(f"\n--- {m.upper()} ---")
        print(
            f"  success_rate:         {s['success_rate']:.4f} "
            f"({s['num_succeeded']}/{s['num_evaluated']})"
        )
        for key in (
            "makespan",
            "travel_distance",
            "task_completion_rate",
            "solving_time",
        ):
            mean, std = s.get(f"{key}_mean"), s.get(f"{key}_std")
            if mean is not None:
                print(f"  {key}: {mean:.4f} ± {std:.4f}")
        if "ltl_satisfaction_rate_mean" in s:
            print(
                f"  ltl_satisfaction_rate: "
                f"{s['ltl_satisfaction_rate_mean']:.4f} ± "
                f"{s['ltl_satisfaction_rate_std']:.4f}"
            )

    # Save
    out = args.output_file or (
        f"baselines_{args.tier}_{'ltl' if args.enable_ltl else 'no_ltl'}_"
        f"{int(time.time())}.json"
    )
    with open(out, "w") as f:
        json.dump(
            {
                "config": {
                    "tier": args.tier,
                    "methods": methods,
                    "num_envs": args.num_envs,
                    "enable_ltl": args.enable_ltl,
                    "time_limit": time_limit,
                    "decide_quantity": args.decide_quantity,
                    "seed_base": args.seed_base,
                },
                "per_method_results": per_method_results,
                "summaries": summaries,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nResults saved to: {out}")


if __name__ == "__main__":
    main()
