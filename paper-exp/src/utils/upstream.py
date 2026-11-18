import importlib.util
from pathlib import Path

from src.utils.settings import config, revision


def load_upstream(root, project, filename):
    path = Path(root) / project
    pin = config("methods/official.yaml")["upstream"][project]["commit"]
    if not path.is_dir() or revision(path) != pin:
        raise FileNotFoundError(f"Need official {project} checkout at {path}, commit {pin}")
    spec = importlib.util.spec_from_file_location(f"paper_{project}_{Path(filename).stem}", path / filename)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
