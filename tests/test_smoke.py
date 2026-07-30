import numpy as np

from src.environment import SafeControllerGrid
from src.expert import QLearningExpert
from src.moe import MixtureOfExpertsPolicy
from src.solver import directed_search


def test_moe_pipeline():
    wide = QLearningExpert("wide", seed=1)
    tall = QLearningExpert("tall", seed=2)
    wide.train_distribution([(6, 2), (7, 3)], episodes=80)
    tall.train_distribution([(3, 6), (3, 7)], episodes=80)
    priors = [
        {(5, 3): 0.15, (3, 5): 0.03},
        {(5, 3): 0.02, (3, 5): 0.16},
    ]
    moe = MixtureOfExpertsPolicy([wide, tall], priors, n=7, k=3, top_k=2)
    assert np.isclose(moe.weights.sum(), 1.0)
    assert np.all(moe.weights >= 0)
    result = directed_search(SafeControllerGrid(7, 3, seed=4), moe, budget=80)
    assert result.expansions <= 80
