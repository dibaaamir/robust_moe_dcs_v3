from pathlib import Path
import json

from src.environment import SafeControllerGrid
from src.expert import QLearningExpert
from src.moe import MixtureOfExpertsPolicy
from src.solver import directed_search

RESULTS = Path("results_v2")
model_paths = sorted((RESULTS / "models").glob("*.json"))
if not model_paths:
    raise SystemExit("First run: python run_experiment.py")
experts = [QLearningExpert.load(path) for path in model_paths]
raw = json.loads((RESULTS / "prior_strengths.json").read_text(encoding="utf-8"))
prior_maps = []
for expert_map in raw:
    prior_maps.append({tuple(map(int, key.split(","))): value for key, value in expert_map.items()})

env = SafeControllerGrid(13, 9, seed=50_003)
policy = MixtureOfExpertsPolicy(experts, prior_maps, n=env.n, k=env.k, top_k=3)
result = directed_search(env, policy, budget=80)
print("regime:", env.regime)
print("weights:", {e.name: round(float(w), 3) for e, w in zip(experts, policy.weights)})
print("solved:", result.solved, "expansions:", result.expansions)
print(env.render_ascii(result.path))
