import argparse
from src.experiment import run

parser = argparse.ArgumentParser(description="Robust simplified MoE-DCS experiment")
parser.add_argument("--output", default="results_v2")
parser.add_argument("--episodes", type=int, default=1600)
parser.add_argument("--calibration-max-n", type=int, default=10)
parser.add_argument("--calibration-max-k", type=int, default=10)
parser.add_argument("--test-min", type=int, default=8)
parser.add_argument("--test-max", type=int, default=14)
parser.add_argument("--budget", type=int, default=55)
parser.add_argument("--eval-seeds", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

run(output_dir=args.output, episodes=args.episodes,
    calibration_max_n=args.calibration_max_n,
    calibration_max_k=args.calibration_max_k,
    test_min=args.test_min, test_max=args.test_max,
    budget=args.budget, eval_seeds=args.eval_seeds, seed=args.seed)
