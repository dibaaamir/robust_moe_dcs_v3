# src/moe.py

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .environment import SafeControllerGrid, GridState, ACTIONS


@dataclass
class MixtureOfExpertsPolicy:

    experts: list
    prior_maps: list[dict[tuple[int, int], float]]

    n: int
    k: int

    beta: float = 0.35
    gamma: float = 0.55
    prior_scale: float = 5.0

    temperature: float = 1.25

    top_k: int | None = 2

    weight_floor: float = 0.0


    def _nearest_prior(
            self,
            prior_map,
            n,
            k):

        if (n, k) in prior_map:

            return prior_map[(n, k)]


        vals = []

        for (pn, pk), value in prior_map.items():

            distance = abs(pn - n) + abs(pk - k)

            vals.append(
                (
                    distance,
                    value
                )
            )


        vals.sort(
            key=lambda x: x[0]
        )


        vals = vals[:4]


        if not vals:

            return 0.0


        weights = np.array(
            [
                1 / (d + 1)
                for d, _ in vals
            ]
        )


        return float(
            np.dot(
                weights,
                [
                    v
                    for _, v in vals
                ]
            )
            /
            weights.sum()
        )



    def _weights_at(
            self,
            env,
            state):

        scores = []


        for i, expert in enumerate(self.experts):

            probs = expert.action_probabilities(
                env,
                state,
                self.temperature
            )


            p = probs[
                probs > 0
            ]


            if len(p):

                entropy = (
                    -np.sum(
                        p *
                        np.log(
                            p + 1e-12
                        )
                    )
                    /
                    max(
                        np.log(len(p)),
                        1e-12
                    )
                )


                sorted_probs = np.sort(
                    p
                )[::-1]


                margin = (
                    sorted_probs[0]
                    -
                    sorted_probs[1]
                    if len(sorted_probs) > 1
                    else sorted_probs[0]
                )


            else:

                entropy = 1.0

                margin = 0.0



            prior = self._nearest_prior(
                self.prior_maps[i],
                self.n,
                self.k
            )


            score = (
                self.prior_scale * prior
                -
                self.beta * entropy
                +
                self.gamma * margin
            )


            scores.append(
                score
            )


        scores = np.array(
            scores
        )


        if self.top_k and self.top_k < len(scores):

            idx = np.argsort(scores)[-self.top_k:]


            mask = np.full_like(
                scores,
                -np.inf
            )


            mask[idx] = scores[idx]

            scores = mask



        finite = np.isfinite(
            scores
        )


        scores[finite] -= (
            scores[finite].max()
        )


        weights = np.zeros_like(
            scores
        )


        weights[finite] = np.exp(
            scores[finite]
        )


        if self.weight_floor:

            weights[finite] += self.weight_floor



        return (
            weights /
            weights.sum()
        )



    def action_probabilities(
            self,
            env: SafeControllerGrid,
            state: GridState):


        valid = env.valid_actions(
            state
        )


        output = np.zeros(
            len(ACTIONS)
        )


        if not valid:

            return output



        weights = self._weights_at(
            env,
            state
        )


        log_scores = np.zeros(
            len(ACTIONS)
        )


        for weight, expert in zip(
            weights,
            self.experts
        ):

            if weight <= 0:

                continue


            probs = expert.action_probabilities(
                env,
                state,
                self.temperature
            )


            log_scores[valid] += (
                weight *
                np.log(
                    np.maximum(
                        probs[valid],
                        1e-8
                    )
                )
            )



        values = log_scores[valid]


        values -= values.max()


        values = np.exp(
            values
        )


        output[valid] = (
            values /
            values.sum()
        )


        self.weights = weights


        return output