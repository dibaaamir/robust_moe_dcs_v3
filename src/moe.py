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

    weight_floor: float = 0.01


    def _nearest_prior(self, prior_map, n, k):

        if (n, k) in prior_map:
            return prior_map[(n, k)]


        vals=[]

        for (pn,pk),v in prior_map.items():

            d=abs(pn-n)+abs(pk-k)

            vals.append(
                (d,v)
            )


        vals.sort(
            key=lambda x:x[0]
        )


        vals=vals[:4]


        if not vals:
            return 0.0


        w=np.array(
            [
                1/(d+1)
                for d,_ in vals
            ]
        )


        return float(
            np.dot(
                w,
                [
                    v
                    for _,v in vals
                ]
            )
            /
            w.sum()
        )



    def _regime_similarity(self, expert, env):

        ratio = env.n / max(env.k,1)


        if ratio > 1.5:

            current="wide"


        elif ratio < 0.66:

            current="tall"


        else:

            current="balanced"



        if current in expert.name:

            return 1.5


        return 0.6



    def _weights_at(self, env, state):

        scores=[]


        for i,e in enumerate(self.experts):


            probs=e.action_probabilities(
                env,
                state,
                self.temperature
            )


            p=probs[
                probs>0
            ]


            if len(p):

                entropy=(
                    -np.sum(
                        p*np.log(
                            p+1e-12
                        )
                    )
                    /
                    max(
                        np.log(len(p)),
                        1e-12
                    )
                )


                q=np.sort(p)[::-1]


                margin=(
                    q[0]-q[1]
                    if len(q)>1
                    else q[0]
                )


            else:

                entropy=1.0
                margin=0.0



            prior=self._nearest_prior(
                self.prior_maps[i],
                self.n,
                self.k
            )



            regime_bonus=self._regime_similarity(
                e,
                env
            )


            score=(

                self.prior_scale
                *
                prior

                +

                np.log(regime_bonus)

                -

                self.beta*entropy

                +

                self.gamma*margin

            )


            scores.append(score)



        scores=np.array(scores)



        if self.top_k and self.top_k<len(scores):

            idx=np.argsort(scores)[-self.top_k:]


            mask=np.full_like(
                scores,
                -np.inf
            )


            mask[idx]=scores[idx]


            scores=mask




        finite=np.isfinite(scores)


        scores[finite]-=scores[finite].max()


        weights=np.zeros_like(scores)


        weights[finite]=np.exp(
            scores[finite]
        )



        if self.weight_floor:

            weights[finite]+=self.weight_floor



        weights/=weights.sum()


        return weights




    def action_probabilities(
            self,
            env: SafeControllerGrid,
            state: GridState
    ):

        valid=env.valid_actions(state)


        out=np.zeros(
            len(ACTIONS)
        )


        if not valid:

            return out



        weights=self._weights_at(
            env,
            state
        )


        log_scores=np.zeros(
            len(ACTIONS)
        )


        for w,e in zip(
            weights,
            self.experts
        ):


            if w<=0:
                continue



            p=e.action_probabilities(
                env,
                state,
                self.temperature
            )


            log_scores[valid]+=(
                w*
                np.log(
                    np.maximum(
                        p[valid],
                        1e-8
                    )
                )
            )



        vals=log_scores[valid]


        vals-=vals.max()


        vals=np.exp(vals)



        out[valid]=(
            vals/
            vals.sum()
        )


        self.weights=weights


        return out