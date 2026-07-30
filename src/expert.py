from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

from .environment import SafeControllerGrid, GridState, ACTIONS


@dataclass
class QLearningExpert:

    name: str

    alpha: float = 0.18
    gamma: float = 0.96

    epsilon: float = 0.28

    temperature: float = 0.75

    bins: int = 10

    seed: int = 0

    action_prior: tuple[float,float,float] | None = None


    q: dict[
        tuple[int,int,int,int],
        np.ndarray
    ] = field(
        default_factory=lambda:
        defaultdict(
            lambda:
            np.zeros(
                len(ACTIONS),
                dtype=float
            )
        )
    )


    def __post_init__(self):

        if self.action_prior is None:


            # Wide specialist
            if "wide" in self.name:

                self.action_prior = (
                    0.80,
                    0.05,
                    0.15
                )


            # Tall specialist
            elif "tall" in self.name:

                self.action_prior = (
                    0.05,
                    0.80,
                    0.15
                )


            # Balanced specialist
            # reduced dominance compared to v3
            elif "balanced" in self.name:

                self.action_prior = (
                    0.4,
                    0.4,
                    0.2
                )


            else:

                self.action_prior = (
                    0.0,
                    0.0,
                    0.0
                )



    def _rng(self):

        if not hasattr(self,"__rng"):

            self.__rng = np.random.default_rng(
                self.seed
            )

        return self.__rng



    def action_scores(
        self,
        env,
        state
    ):

        learned=np.array(
            self.q[
                env.state_key(
                    state,
                    self.bins
                )
            ],
            dtype=float
        )


        return (
            learned
            +
            np.array(
                self.action_prior,
                dtype=float
            )
        )



    def action_probabilities(
        self,
        env,
        state,
        temperature=None
    ):

        valid=env.valid_actions(state)


        probs=np.zeros(
            len(ACTIONS)
        )


        if not valid:
            return probs


        scores=self.action_scores(
            env,
            state
        )


        t=max(
            1e-6,
            self.temperature
            if temperature is None
            else temperature
        )


        logits=scores[valid]/t

        logits-=logits.max()


        values=np.exp(logits)


        probs[valid]=(
            values /
            values.sum()
        )


        return probs



    def choose_action(
        self,
        env,
        state,
        explore=True
    ):


        valid=env.valid_actions(state)


        if not valid:
            return None


        rng=self._rng()


        if explore and rng.random()<self.epsilon:

            return int(
                rng.choice(valid)
            )


        scores=self.action_scores(
            env,
            state
        )


        best=max(
            scores[a]
            for a in valid
        )


        candidates=[
            a
            for a in valid
            if np.isclose(
                scores[a],
                best
            )
        ]


        return int(
            rng.choice(candidates)
        )



    def train_distribution(
        self,
        training_sizes,
        episodes=1200,
        max_steps=180,
        environment_seeds=40
    ):


        rng=self._rng()

        initial_epsilon=self.epsilon


        for episode in range(episodes):


            n,k = training_sizes[
                int(
                    rng.integers(
                        0,
                        len(training_sizes)
                    )
                )
            ]


            env=SafeControllerGrid(
                n,
                k,
                seed=self.seed*1000 + episode % environment_seeds
            )


            state=env.start


            for _ in range(max_steps):


                action=self.choose_action(
                    env,
                    state,
                    True
                )


                if action is None:
                    break


                nxt,reward,done=env.step(
                    state,
                    action
                )


                key=env.state_key(
                    state,
                    self.bins
                )


                next_key=env.state_key(
                    nxt,
                    self.bins
                )


                next_valid=env.valid_actions(
                    nxt
                )


                bootstrap=max(
                    (
                        self.q[next_key][a]
                        for a in next_valid
                    ),
                    default=0.0
                )


                target = (
                    reward
                    if done
                    else
                    reward+self.gamma*bootstrap
                )


                self.q[key][action]+=(
                    self.alpha *
                    (
                        target -
                        self.q[key][action]
                    )
                )


                state=nxt


                if done:
                    break


            self.epsilon=max(
                0.025,
                initial_epsilon*
                (0.996**episode)
            )



    def save(self,path):

        serial={
            "|".join(map(str,k)):
            v.tolist()
            for k,v in self.q.items()
        }


        payload={

            "name":self.name,

            "bins":self.bins,

            "temperature":self.temperature,

            "seed":self.seed,

            "action_prior":self.action_prior,

            "q":serial
        }


        Path(path).write_text(
            json.dumps(
                payload,
                indent=2
            ),
            encoding="utf-8"
        )



    @classmethod
    def load(cls,path):

        payload=json.loads(
            Path(path).read_text(
                encoding="utf-8"
            )
        )


        expert=cls(
            name=payload["name"],
            bins=payload["bins"],
            temperature=payload["temperature"],
            seed=payload.get("seed",0),
            action_prior=tuple(
                payload.get(
                    "action_prior",
                    (0,0,0)
                )
            )
        )


        for key,value in payload["q"].items():

            expert.q[
                tuple(
                    map(
                        int,
                        key.split("|")
                    )
                )
            ]=np.array(
                value,
                dtype=float
            )


        return expert