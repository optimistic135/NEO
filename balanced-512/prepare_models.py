"""Populate the local cache with exactly the checkpoints selected by the benchmark."""

import json
from pathlib import Path

from huggingface_hub import snapshot_download
import yaml


def main():
    here = Path(__file__).resolve().parent
    models = yaml.safe_load(
        (here / ".runtime/paper-exp/configs/models/vit_base.yaml").read_text()
    )
    for spec in models.values():
        snapshot_download(spec["repo"], revision=spec["revision"],
                          allow_patterns=[spec["weights"], "config.json", "preprocessor_config.json"])
    snapshot_download("google/vit-base-patch16-224-in21k",
                      revision="b4569560a39a0f1af58e3ddaf17facf20ab919b0",
                      allow_patterns=["preprocessor_config.json"])
    print(json.dumps(models, indent=2))


if __name__ == "__main__":
    main()
