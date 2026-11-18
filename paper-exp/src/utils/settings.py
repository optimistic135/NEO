import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

HOME = Path(__file__).resolve().parents[2]
REPO = HOME.parent
sys.path.insert(0, str(REPO))


def config(path):
    with (HOME / "configs" / path).open() as stream:
        return yaml.safe_load(stream)


def sha256(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
