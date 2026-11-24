from src.methods.cotta import CoTTA
from src.methods.deyo import DeYO
from src.methods.eata import EATA
from src.methods.neo import NEO
from src.methods.roid import build_roid
from src.methods.foa import FOA
from src.methods.fozo import FOZO
from src.methods.lame import LAME
from src.methods.no_adapt import NoAdapt
from src.methods.sar import SAR
from src.methods.t3a import T3A
from src.methods.tent import TENT


METHODS = {
    "no_adapt": lambda m, c, x: NoAdapt(m),
    "tent": lambda m, c, x: TENT(m, c, imagenet=x.get("dataset") == "imagenet-c"),
    "t3a": lambda m, c, x: T3A(m, c["filter_k"]),
    "sar": lambda m, c, x: SAR(m, c, x["batch_size"]),
    "cotta": lambda m, c, x: CoTTA(m, c, imagenet=x.get("dataset") == "imagenet-c"),
    "eata": EATA,
    "neo": lambda m, c, x: NEO(m),
    "roid": build_roid,
    "deyo": lambda m, c, x: DeYO(m, c),
    "lame": lambda m, c, x: LAME(m, c),
    "foa": FOA,
    "fozo": FOZO,
}


def setup(model, configuration, context):
    adapter = METHODS[configuration["id"]](model, configuration, context)
    # CUDA work performed to compute source statistics is recorded separately.
    return adapter
