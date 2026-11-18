import argparse
from pathlib import Path

from src.utils.settings import HOME, config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("validate", "smoke", "balanced", "online_ir", "ablation", "efficiency", "report"))
    p.add_argument("--datasets", nargs="+", default=list(config("datasets/corruptions.yaml")["datasets"]))
    p.add_argument("--architectures", nargs="+", default=["vit_base", "resnet50"], choices=["vit_base", "resnet50"])
    p.add_argument("--methods", nargs="+")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, default=HOME / "results")
    p.add_argument("--upstream", type=Path, default=HOME / ".upstream")
    p.add_argument("--dart-manifests", type=Path)
    p.add_argument("--online", action="store_true")
    opt = p.parse_args()
    opt.smoke = opt.stage == "smoke"
    if opt.smoke:
        opt.stage, opt.output = "balanced", opt.output / "smoke"
    if opt.stage == "report":
        from src.engine.report import report
        report(opt.output)
    else:
        from src.engine.runner import run
        run(opt)


if __name__ == "__main__":
    main()
