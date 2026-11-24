import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.engine.online_benchmark import DESIGN, run
from src.utils.settings import REPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("all", *DESIGN), default="all")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--capacity-policy", choices=("strict", "cap"), default="strict")
    parser.add_argument("--reference", type=Path, default=REPO / "results/ir-benchmark")
    parser.add_argument("--imagenet-clean", type=Path, help="Independent clean ImageNet root with train/ subdirectory")
    parser.add_argument("--smoke-only", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
