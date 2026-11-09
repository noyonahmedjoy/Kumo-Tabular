import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.benchmarking import load_config, run_experiment  # noqa: E402


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "experiment_config.yaml")
    parser.add_argument("--results-dir", default=PROJECT_ROOT / "results")
    parser.add_argument("--size", choices=["small", "medium", "large"], help="override the Kumo model size")
    parser.add_argument("--device", help="override the Kumo device, e.g. cpu or cuda")
    return parser.parse_args()


def run_tasks(tasks: list[str], args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.size:
        config["kumo"]["size"] = args.size
    if args.device:
        config["kumo"]["device"] = args.device
    with pd.option_context("display.width", 160, "display.max_columns", 20, "display.precision", 4):
        for task in tasks:
            print(f"\n=== {task} ===")
            output = run_experiment(task, config, args.results_dir)
            columns = [c for c in output["summary"].columns if c.endswith("_mean") or c.startswith("runs")]
            print(output["summary"][["model", *columns]].to_string(index=False))
            print()
            print(output["leaderboard"].to_string(index=False))
            print()
            print(output["paired"].to_string(index=False))
            print(f"\nSaved results to {Path(args.results_dir) / task}")


if __name__ == "__main__":
    run_tasks(["classification", "regression"], parse_args("Run both Tabular Intelligence benchmarks."))
