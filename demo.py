from pathlib import Path
import json

from src.environment import SafeControllerGrid
from src.expert import QLearningExpert
from src.moe import MixtureOfExpertsPolicy
from src.solver import directed_search
from src.experiment import EXPERT_SPECS


ROOT = Path(__file__).resolve().parent

RESULTS = ROOT / "results_v2"

MODEL_DIR = RESULTS / "models"

PRIOR_FILE = RESULTS / "prior_strengths.json"



if not PRIOR_FILE.exists():
    raise FileNotFoundError(
        "prior_strengths.json not found. "
        "Run run_experiment.py first."
    )



expert_names = list(EXPERT_SPECS.keys())


experts = []

for name in expert_names:

    path = MODEL_DIR / f"{name}.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing model: {path}"
        )

    experts.append(
        QLearningExpert.load(path)
    )



raw = json.loads(
    PRIOR_FILE.read_text(
        encoding="utf-8"
    )
)



prior_maps = []

for item in raw:

    converted = {}

    for key,value in item.items():

        n,k = key.split(",")

        converted[
            (int(n),int(k))
        ] = float(value)


    prior_maps.append(converted)



env = SafeControllerGrid(
    13,
    9,
    seed=50003
)



policy = MixtureOfExpertsPolicy(

    experts=experts,

    prior_maps=prior_maps,

    n=env.n,

    k=env.k,

    top_k=2,

    prior_scale=10.0,

    beta=0.35,

    gamma=0.45,

    weight_floor=0.01
)



result = directed_search(
    env,
    policy,
    budget=80
)



weights = policy.weights



print(
    "regime:",
    env.regime
)



print(
    "weights:",
    {
        e.name: round(float(w),3)
        for e,w in zip(
            experts,
            weights
        )
    }
)



print(
    "solved:",
    result.solved,
    "expansions:",
    result.expansions
)



print(
    env.render_ascii(
        result.path
    )
)