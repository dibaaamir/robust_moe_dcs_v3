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


EXPERT_SPECS = {

    "expert_wide_small": [
        (6, 2),
        (7, 3),
        (8, 3),
        (9, 4),
    ],

    "expert_wide_large": [
        (10, 4),
        (11, 5),
        (12, 5),
        (13, 6),
    ],

    "expert_tall_small": [
        (3, 6),
        (3, 7),
        (4, 8),
        (4, 9),
    ],

    "expert_tall_large": [
        (5, 10),
        (5, 11),
        (6, 12),
        (6, 13),
    ],

    "expert_balanced_small": [
        (5, 5),
        (6, 6),
    ],

    "expert_balanced_large": [
        (8, 8),
        (9, 9),
    ],
}



def train_experts(output_dir, episodes, seed):

    model_dir = output_dir / "models"
    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    experts = []

    for idx, (name, sizes) in enumerate(EXPERT_SPECS.items()):

        expert = QLearningExpert(
            name=name,
            seed=seed + idx * 101
        )

        expert.train_distribution(
            sizes,
            episodes=episodes
        )

        expert.save(
            model_dir / f"{name}.json"
        )

        experts.append(expert)

        print(
            f"trained {name} on {sizes}"
        )

    return experts




def calibration_points(max_n, max_k):

    points = []

    for n in range(3, max_n + 1):

        for k in range(2, max_k + 1):

            points.append(
                (n, k)
            )

    return points




def build_prior_maps(
        experts,
        points,
        budget,
        seeds=5):

    maps = [
        {}
        for _ in experts
    ]


    for n, k in points:

        for i, expert in enumerate(experts):

            solved = []
            efficiency = []


            for s in range(seeds):

                env = SafeControllerGrid(
                    n,
                    k,
                    seed=20000+s
                )


                result = directed_search(
                    env,
                    expert,
                    budget
                )


                solved.append(
                    float(result.solved)
                )


                efficiency.append(
                    (budget-result.expansions)/budget
                    if result.solved
                    else 0.0
                )


            maps[i][(n,k)] = float(
                0.8*np.mean(solved)
                +
                0.2*np.mean(efficiency)
            )


    return maps




def _policies(
        experts,
        prior_maps,
        n,
        k):


    policies = {
        e.name:e
        for e in experts
    }


    policies["moe_sparse_top1"] = MixtureOfExpertsPolicy(
        experts,
        prior_maps,
        n,
        k,
        top_k=1,
        prior_scale=10.0,
        beta=0.0,
        gamma=0.0
    )


    policies["moe_top2"] = MixtureOfExpertsPolicy(
        experts,
        prior_maps,
        n,
        k,
        top_k=2,
        prior_scale=10.0,
        beta=0.35,
        gamma=0.45,
        weight_floor=0.01
    )


    policies["moe_top3"] = MixtureOfExpertsPolicy(
        experts,
        prior_maps,
        n,
        k,
        top_k=3,
        prior_scale=3.0,
        beta=0.20,
        gamma=0.25,
        weight_floor=0.005
    )


    policies["moe_all"] = MixtureOfExpertsPolicy(
        experts,
        prior_maps,
        n,
        k,
        top_k=None,
        prior_scale=2.0
    )


    return policies




def evaluate(
        experts,
        prior_maps,
        test_points,
        budget,
        eval_seeds):


    records = []


    for n,k in test_points:

        policies = _policies(
            experts,
            prior_maps,
            n,
            k
        )


        for policy_name, policy in policies.items():

            success = []
            all_expansions = []
            success_expansions = []
            elapsed = []


            for s in range(eval_seeds):

                env = SafeControllerGrid(
                    n,
                    k,
                    seed=50000+s
                )


                start=time.perf_counter()


                result = directed_search(
                    env,
                    policy,
                    budget
                )


                elapsed.append(
                    time.perf_counter()-start
                )


                success.append(
                    int(result.solved)
                )


                all_expansions.append(
                    result.expansions
                )


                if result.solved:

                    success_expansions.append(
                        result.expansions
                    )



            records.append({

                "n":n,
                "k":k,

                "policy":policy_name,

                "success_rate":
                    float(np.mean(success)),

                "mean_expansions_all":
                    float(np.mean(all_expansions)),

                "mean_expansions_success":
                    float(np.mean(success_expansions))
                    if success_expansions
                    else np.nan,

                "mean_seconds":
                    float(np.mean(elapsed))
            })


    return pd.DataFrame(records)




def add_oracle_baseline(df, expert_names):

    expert_df = df[
        df.policy.isin(expert_names)
    ]


    rows=[]


    for (n,k), group in expert_df.groupby(
        ["n","k"]
    ):

        row = group.sort_values(
            [
                "success_rate",
                "mean_expansions_success"
            ],
            ascending=[
                False,
                True
            ]
        ).iloc[0].copy()


        row["policy"] = "oracle_best_expert"

        rows.append(row)


    return pd.concat(
        [
            df,
            pd.DataFrame(rows)
        ],
        ignore_index=True
    )




def summarize(df):

    return (
        df.groupby(
            "policy",
            as_index=False
        )
        .agg(
            success_mean=(
                "success_rate",
                "mean"
            ),

            expansions_success=(
                "mean_expansions_success",
                "mean"
            ),

            seconds=(
                "mean_seconds",
                "mean"
            )
        )
        .sort_values(
            "success_mean",
            ascending=False
        )
    )




def plot_heatmap(df, policy, output):

    subset=df[
        df.policy==policy
    ]


    if subset.empty:
        return


    pivot=subset.pivot(
        index="k",
        columns="n",
        values="success_rate"
    )


    fig, ax = plt.subplots(
        figsize=(9,6)
    )


    image=ax.imshow(
        pivot.values,
        aspect="auto",
        vmin=0,
        vmax=1
    )


    ax.set_title(policy)


    fig.colorbar(
        image,
        ax=ax
    )


    fig.tight_layout()


    fig.savefig(
        output,
        dpi=170
    )


    plt.close(fig)




def plot_summary(summary, output):

    shown = summary[
        summary.policy!="oracle_best_expert"
    ]


    fig, ax = plt.subplots(
        figsize=(12,5)
    )


    ax.bar(
        shown.policy,
        shown.success_mean
    )


    ax.tick_params(
        axis="x",
        rotation=45
    )


    fig.tight_layout()


    fig.savefig(
        output,
        dpi=170
    )


    plt.close(fig)




def run(
        output_dir="results_v2",
        episodes=1600,
        calibration_max_n=14,
        calibration_max_k=14,
        test_min=3,
        test_max=14,
        budget=25,
        seed=42,
        eval_seeds=10):


    output = Path(output_dir)

    output.mkdir(
        parents=True,
        exist_ok=True
    )


    experts = train_experts(
        output,
        episodes,
        seed
    )


    cal_points = calibration_points(
        calibration_max_n,
        calibration_max_k
    )


    prior_maps = build_prior_maps(
        experts,
        cal_points,
        budget
    )


    serial_priors = [
        {
            f"{n},{k}":float(v)
            for (n,k),v in m.items()
        }
        for m in prior_maps
    ]


    (output/"prior_strengths.json").write_text(
        json.dumps(
            serial_priors,
            indent=2
        ),
        encoding="utf-8"
    )



    (output/"experiment_config.json").write_text(
        json.dumps(
            {
                "expert_order":[
                    e.name
                    for e in experts
                ],
                "expert_specs":EXPERT_SPECS,
                "budget":budget,
                "seed":seed,
                "eval_seeds":eval_seeds
            },
            indent=2
        ),
        encoding="utf-8"
    )



    test_points = [
        (n,k)
        for n in range(test_min,test_max+1)
        for k in range(test_min,test_max+1)
    ]



    df = evaluate(
        experts,
        prior_maps,
        test_points,
        budget,
        eval_seeds
    )


    df = add_oracle_baseline(
        df,
        [
            e.name
            for e in experts
        ]
    )


    df.to_csv(
        output/"evaluation.csv",
        index=False
    )



    summary = summarize(df)


    summary.to_csv(
        output/"summary.csv",
        index=False
    )



    for policy in df.policy.unique():

        plot_heatmap(
            df,
            policy,
            output /
            f"heatmap_{policy}.png"
        )


    plot_summary(
        summary,
        output/"policy_comparison.png"
    )


    print(
        summary.to_string(
            index=False
        )
    )


    return df