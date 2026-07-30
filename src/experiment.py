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


EXPERT_SPECS: dict[str, list[tuple[int, int]]] = {

    "expert_wide_small": [
        (6, 2),
        (7, 3),
        (8, 3),
        (9, 4)
    ],

    "expert_wide_large": [
        (10, 4),
        (11, 5),
        (12, 5),
        (13, 6)
    ],

    "expert_tall_small": [
        (3, 6),
        (3, 7),
        (4, 8),
        (4, 9)
    ],

    "expert_tall_large": [
        (5, 10),
        (5, 11),
        (6, 12),
        (6, 13)
    ],

    "expert_balanced_small": [
        (5, 5),
        (6, 6)
    ],

    "expert_balanced_large": [
        (8, 8),
        (9, 9)
    ],
}



def train_experts(output_dir: Path,
                  episodes: int,
                  seed: int) -> list[QLearningExpert]:

    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    experts = []

    for idx, (name, sizes) in enumerate(EXPERT_SPECS.items()):

        expert = QLearningExpert(
            name=name,
            seed=seed + 101 * idx
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




def calibration_points(max_n: int,
                       max_k: int):

    points = []

    for n in range(3, max_n + 1):

        for k in range(2, max_k + 1):

            points.append((n, k))

    return points




def build_prior_maps(experts,
                     points,
                     budget,
                     seeds=5):

    maps = [
        {}
        for _ in experts
    ]


    for n, k in points:

        for i, expert in enumerate(experts):

            solved_flags = []
            efficiencies = []


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


                solved_flags.append(
                    float(result.solved)
                )


                if result.solved:

                    efficiencies.append(
                        (budget-result.expansions)
                        /
                        budget
                    )

                else:

                    efficiencies.append(0.0)



            maps[i][(n,k)] = float(
                0.8*np.mean(solved_flags)
                +
                0.2*np.mean(efficiencies)
            )


    return maps





def _policies(experts,
              prior_maps,
              n,
              k):


    policies = {
        expert.name: expert
        for expert in experts
    }



    policies["moe_sparse_top1"] = MixtureOfExpertsPolicy(
        experts,
        prior_maps,
        n=n,
        k=k,
        top_k=1,
        prior_scale=10.0,
        beta=0.0,
        gamma=0.0,
        weight_floor=0.0
    )



    policies["moe_top2"] = MixtureOfExpertsPolicy(

        experts,
        prior_maps,

        n=n,
        k=k,

        top_k=2,

        prior_scale=10.0,

        beta=0.35,

        gamma=0.45,

        weight_floor=0.01
    )



    policies["moe_top3"] = MixtureOfExpertsPolicy(

        experts,
        prior_maps,

        n=n,
        k=k,

        top_k=3,

        prior_scale=3.0,

        beta=0.20,

        gamma=0.25,

        weight_floor=0.005
    )



    policies["moe_all"] = MixtureOfExpertsPolicy(

        experts,
        prior_maps,

        n=n,
        k=k,

        top_k=None,

        prior_scale=2.0

    )


    return policies





def evaluate(experts,
             prior_maps,
             test_points,
             budget,
             eval_seeds=10):


    records=[]


    for n,k in test_points:


        policies = _policies(
            experts,
            prior_maps,
            n,
            k
        )


        for policy_name, policy in policies.items():


            solved_flags=[]
            expansions=[]
            success_exp=[]
            fail_exp=[]
            elapsed=[]


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


                solved_flags.append(
                    int(result.solved)
                )


                expansions.append(
                    result.expansions
                )


                if result.solved:

                    success_exp.append(
                        result.expansions
                    )

                else:

                    fail_exp.append(
                        result.expansions
                    )



            records.append({

                "n":n,
                "k":k,

                "policy":policy_name,

                "success_rate":
                    float(np.mean(solved_flags)),

                "mean_expansions_all":
                    float(np.mean(expansions)),

                "mean_expansions_success":
                    float(np.mean(success_exp))
                    if success_exp else np.nan,

                "mean_expansions_failure":
                    float(np.mean(fail_exp))
                    if fail_exp else np.nan,

                "mean_seconds":
                    float(np.mean(elapsed))

            })


    return pd.DataFrame(records)





def add_oracle_baseline(df,
                        expert_names):


    expert_df=df[
        df.policy.isin(expert_names)
    ]


    rows=[]


    for (n,k),group in expert_df.groupby(
        ["n","k"]
    ):


        row=group.sort_values(

            [
                "success_rate",
                "mean_expansions_success"
            ],

            ascending=[
                False,
                True
            ]

        ).iloc[0].copy()


        row["policy"]="oracle_best_expert"

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

            success_mean=
            ("success_rate","mean"),

            expansions_success=
            ("mean_expansions_success","mean"),

            seconds=
            ("mean_seconds","mean")

        )

        .sort_values(
            "success_mean",
            ascending=False
        )
    )





def default_test_points(test_min,
                        test_max):

    return [

        (n,k)

        for n in range(test_min,test_max+1)

        for k in range(test_min,test_max+1)

    ]





def run(output_dir="results_v3_final",
        episodes=1600,
        calibration_max_n=14,
        calibration_max_k=14,
        test_min=3,
        test_max=14,
        budget=25,
        seed=42,
        eval_seeds=10):


    output=Path(output_dir)

    output.mkdir(
        parents=True,
        exist_ok=True
    )


    experts=train_experts(
        output,
        episodes,
        seed
    )


    points=calibration_points(
        calibration_max_n,
        calibration_max_k
    )


    prior_maps=build_prior_maps(
        experts,
        points,
        budget
    )


    test_points=default_test_points(
        test_min,
        test_max
    )


    df=evaluate(
        experts,
        prior_maps,
        test_points,
        budget,
        eval_seeds
    )


    df=add_oracle_baseline(
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


    summary=summarize(df)


    summary.to_csv(
        output/"summary.csv",
        index=False
    )


    print(summary.to_string(index=False))


    return df