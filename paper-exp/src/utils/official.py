"""Fetch reviewed upstream files at immutable revisions and verify their hashes."""

from hashlib import sha256
from pathlib import Path
import sys
from types import ModuleType
from urllib.request import urlopen

from src.utils.settings import HOME


def official_file(config, name):
    pin = config["upstream"]
    expected = pin["files"][name]
    root = HOME / "third_party" / config["id"] / pin["commit"]
    path = root / name
    if not path.exists():
        repo = pin["url"].removeprefix("https://github.com/")
        url = f"https://raw.githubusercontent.com/{repo}/{pin['commit']}/{name}"
        with urlopen(url, timeout=30) as response:
            payload = response.read()
        if sha256(payload).hexdigest() != expected:
            raise ValueError(f"Upstream file SHA256 mismatch: {url}")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(payload)
        tmp.replace(path)
    if sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Cached upstream file SHA256 mismatch: {path}")
    return path


def module(config, name, replacements=(), **context):
    path = official_file(config, name)
    key = f"paper_{config['id']}_{name.replace(chr(47), chr(95)).removesuffix('.py')}"
    if key in sys.modules:
        return sys.modules[key]
    source = path.read_text()
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError(f"Upstream adapter anchor changed in {path}: {old}")
        source = source.replace(old, new)
    item = ModuleType(key)
    item.__file__ = str(path)
    item.__dict__.update(context)
    sys.modules[key] = item
    try:
        exec(compile(source, str(path), "exec"), item.__dict__)
    except BaseException:
        sys.modules.pop(key)
        raise
    return item
