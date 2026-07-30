from __future__ import annotations

from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .environment import SafeControllerGrid
from .expert import QLearningExpert
from .solver import directed_search
from .moe import MixtureOfExpertsPolicy

# Each expert is trained on a distribution, not one tiny grid. The distributions
# intentionally cover complementary structural regimes.
EXPERT_SPECS: dict[str, list[tuple[int, int]]] = {
    "expert_wide_small": [(6, 2), (7, 3), (8, 3), (9, 4)],
    "expert_wide_large": [(10, 4), (11, 5), (12, 5), (13, 6)],
    "expert_tall_small": [(3, 6), (3, 7), (4, 8), (4, 9)],
    "expert_tall_large": [(5, 10), (5, 11), (6, 12), (6, 13)],
    # Balanced experts are intentionally narrow. They must not become
    # universal experts that dominate every regime.
    "expert_balanced_small": [(5, 5), (6, 6)],
    "expert_balanced_large": [(8, 8), (9, 9)],
}


def train_experts(output_dir: Path, episodes: int, seed: int) -> list[QLearningExpert]:
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    experts: list[QLearningExpert] = []
    for idx, (name, sizes) in enumerate(EXPERT_SPECS.items()):
        expert = QLearningExpert(name=name, seed=seed + 101 * idx)
        expert.train_distribution(sizes, episodes=episodes)
        expert.save(model_dir / f"{name}.json")
        experts.append(expert)
        print(f"trained {name} on {sizes}")
    return experts


def calibration_points(max_n: int, max_k: int) -> list[tuple[int, int]]:
    # Sparse design-time grid. Final evaluation uses separate sizes and seeds.
    points: list[tuple[int, int]] = []
    for n in range(3, max_n + 1):
        for k in range(2, max_k + 1):
            points.append((n, k))
    return points


def build_prior_maps(experts: list[QLearningExpert], points: list[tuple[int, int]],
                     budget: int, seeds: int = 5) -> list[dict[tuple[int, int], float]]:
    maps: list[dict[tuple[int, int], float]] = [dict() for _ in experts]
    for n, k in points:
        for i, expert in enumerate(experts):
            solved_flags = []
            efficiencies = []
            for s in range(seeds):
                env = SafeControllerGrid(n, k, seed=20_000 + s)
                result = directed_search(env, expert, budget)
                solved_flags.append(float(result.solved))
                efficiencies.append(((budget - result.expansions) / budget) if result.solved else 0.0)
            # Reliability-first prior: solving is primary; efficiency breaks ties.
            maps[i][(n, k)] = float(np.mean(solved_flags) + 0.10 * np.mean(efficiencies))
    return maps


def _policies(experts, prior_maps, n: int, k: int):
    policies = {expert.name: expert for expert in experts}
    policies["moe_sparse_top1"] = MixtureOfExpertsPolicy(
        experts, prior_maps, n=n, k=k, top_k=1, prior_scale=10.0, beta=0.0, gamma=0.0, weight_floor=0.0
    )
    policies["moe_top2"] = MixtureOfExpertsPolicy(
        experts, prior_maps, n=n, k=k, top_k=2, prior_scale=8.0, beta=0.20, gamma=0.30, weight_floor=0.0
    )
    policies["moe_top3"] = MixtureOfExpertsPolicy(
        experts, prior_maps, n=n, k=k, top_k=3, prior_scale=5.0, beta=0.25, gamma=0.35, weight_floor=0.005
    )
    policies["moe_all"] = MixtureOfExpertsPolicy(
        experts, prior_maps, n=n, k=k, top_k=None, prior_scale=2.0
    )
    return policies


def evaluate(experts, prior_maps, test_points: list[tuple[int, int]], budget: int,
             eval_seeds: int = 10) -> pd.DataFrame:
    records = []
    for n, k in test_points:
        for policy_name, policy in _policies(experts, prior_maps, n, k).items():
            solved_flags: list[int] = []
            expansions: list[int] = []
            successful_expansions: list[int] = []
            failed_expansions: list[int] = []
            elapsed: list[float] = []
            for s in range(eval_seeds):
                env = SafeControllerGrid(n, k, seed=50_000 + s)
                start = time.perf_counter()
                result = directed_search(env, policy, budget)
                elapsed.append(time.perf_counter() - start)
                solved_flags.append(int(result.solved))
                expansions.append(result.expansions)
                (successful_expansions if result.solved else failed_expansions).append(result.expansions)
            records.append({
                "n": n,
                "k": k,
                "regime": SafeControllerGrid(n, k).regime,
                "policy": policy_name,
                "success_rate": float(np.mean(solved_flags)),
                "successes": int(np.sum(solved_flags)),
                "trials": eval_seeds,
                "mean_expansions_all": float(np.mean(expansions)),
                "mean_expansions_success": float(np.mean(successful_expansions)) if successful_expansions else np.nan,
                "mean_expansions_failure": float(np.mean(failed_expansions)) if failed_expansions else np.nan,
                "mean_seconds": float(np.mean(elapsed)),
            })
    return pd.DataFrame.from_records(records)


def add_oracle_baseline(df: pd.DataFrame, expert_names: list[str]) -> pd.DataFrame:
    expert_df = df[df.policy.isin(expert_names)]
    oracle_rows = []
    for (n, k), group in expert_df.groupby(["n", "k"]):
        row = group.sort_values(["success_rate", "mean_expansions_success"],
                                ascending=[False, True], na_position="last").iloc[0].copy()
        row["policy"] = "oracle_best_expert"
        oracle_rows.append(row)
    return pd.concat([df, pd.DataFrame(oracle_rows)], ignore_index=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("policy", as_index=False)
            .agg(success_sum=("success_rate", "sum"),
                 success_mean=("success_rate", "mean"),
                 expansions_success=("mean_expansions_success", "mean"),
                 seconds=("mean_seconds", "mean"))
            .sort_values(["success_sum", "expansions_success"], ascending=[False, True]))


def plot_heatmap(df: pd.DataFrame, policy: str, output: Path) -> None:
    subset = df[df.policy == policy]
    pivot = subset.pivot(index="k", columns="n", values="success_rate").sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    image = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_xlabel("n: horizontal complexity")
    ax.set_ylabel("k: vertical complexity")
    ax.set_title(f"Held-out zero-shot solvability: {policy}")
    fig.colorbar(image, ax=ax, label="success rate")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    shown = summary[summary.policy != "oracle_best_expert"]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(shown.policy, shown.success_sum)
    ax.set_ylabel("Equivalent solved held-out instances")
    ax.set_title("Single specialists vs calibrated Mixture-of-Experts")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def default_test_points(test_min: int, test_max: int) -> list[tuple[int, int]]:
    return [(n, k) for n in range(test_min, test_max + 1)
            for k in range(test_min, test_max + 1)]


def run(output_dir: str = "results_v2", episodes: int = 1600,
        calibration_max_n: int = 10, calibration_max_k: int = 10,
        test_min: int = 8, test_max: int = 14, budget: int = 55,
        seed: int = 42, eval_seeds: int = 10) -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    experts = train_experts(output, episodes, seed)

    cal_points = calibration_points(calibration_max_n, calibration_max_k)
    prior_maps = build_prior_maps(experts, cal_points, budget, seeds=5)
    serial_priors = [{f"{n},{k}": v for (n, k), v in m.items()} for m in prior_maps]
    (output / "prior_strengths.json").write_text(json.dumps(serial_priors, indent=2), encoding="utf-8")
    (output / "experiment_config.json").write_text(json.dumps({
        "expert_specs": EXPERT_SPECS,
        "calibration_points": cal_points,
        "test_range": [test_min, test_max],
        "budget": budget,
        "eval_seeds": eval_seeds,
        "seed": seed,
    }, indent=2), encoding="utf-8")

    test_points = default_test_points(test_min, test_max)
    df = evaluate(experts, prior_maps, test_points, budget, eval_seeds)
    df = add_oracle_baseline(df, [e.name for e in experts])
    df.to_csv(output / "evaluation.csv", index=False)
    summary = summarize(df)
    summary.to_csv(output / "summary.csv", index=False)

    for policy in [e.name for e in experts] + ["moe_sparse_top1", "moe_top2", "moe_top3", "moe_all"]:
        plot_heatmap(df, policy, output / f"heatmap_{policy}.png")
    plot_summary(summary, output / "policy_comparison.png")
    print(summary.to_string(index=False))
    return df
