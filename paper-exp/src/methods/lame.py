import torch
from torch.nn import functional as F

from src.methods.base import Adapter
from src.utils.official import module


class LAME(Adapter):
    timing = "stateless-batch-graph"

    def __init__(self, model, cfg):
        super().__init__(model)
        upstream = module(cfg, "src/adaptation/lame.py", (
            ("from .non_adaptive import NonAdaptiveMethod", ""),
            ("from .build import ADAPTER_REGISTRY", ""),
            ("@ADAPTER_REGISTRY.register()", ""),
        ), NonAdaptiveMethod=object)
        self.graph = getattr(upstream, cfg["affinity"] + "_affinity")(knn=cfg["knn"], sigma=cfg["sigma"])
        self.force_symmetry = cfg["force_symmetry"]
        self.optimize = upstream.laplacian_optimization

    @torch.no_grad()
    def predict(self, images):
        features = self.features(images)
        prob = self.model.head(features).softmax(1)
        graph = self.graph(F.normalize(features, dim=-1))
        if self.force_symmetry:
            graph = (graph + graph.T) * .5
        return self.optimize(-(prob + 1e-10).log(), graph)
