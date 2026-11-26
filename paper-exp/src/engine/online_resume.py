from copy import deepcopy


_PREVIOUS = "06a0163db02c0a9a2a71c25f1f1a31b8e386781d"


def can_resume_without_cotta(previous, target, prior_roid_success):
    """Allow the known repair only, with all stream and unaffected method settings equal."""
    if previous.get("code_git") != _PREVIOUS or prior_roid_success:
        return False
    old = deepcopy(previous)
    old["code_git"] = target.get("code_git")
    methods = old.get("methods", {})
    if methods.pop("cotta", None) is None:
        return False
    roid = methods.get("roid", {})
    if roid.get("eps") not in ("1e-05", 0.00001):
        return False
    roid["eps"] = 0.00001
    return old == target
