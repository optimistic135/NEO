import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.engine.benchmark import run

DEFAULT = "experiments/balanced/cifar100c_vitb.yaml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT)
    parser.add_argument("--output")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
