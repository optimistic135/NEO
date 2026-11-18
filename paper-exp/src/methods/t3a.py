import torch
from torch.nn import functional as F

from src.methods.base import Adapter, entropy


class T3A(Adapter):
    def __init__(self, model, filter_k):
        super().__init__(model)
        self.k = filter_k
        self.reset()

    @torch.no_grad()
    def reset(self):
        super().reset()
        self.supports = self.model.head.weight.detach()
        logits = self.model.head(self.supports)
        self.labels, self.ent = logits.argmax(1), entropy(logits)

    def classify(self, features):
        templates = features.new_zeros((self.model.head.out_features, features.shape[1]))
        templates.index_add_(0, self.labels, F.normalize(self.supports))
        return features @ F.normalize(templates).T

    def predict(self, images):
        return self.classify(self.features(images))

    @torch.no_grad()
    def update(self, images):
        features = self.features(images)
        logits = self.model.head(features)
        supports = torch.cat([self.supports, features])
        labels = torch.cat([self.labels, logits.argmax(1)])
        ent = torch.cat([self.ent, entropy(logits)])
        # Sort by entropy, then stably group by pseudo-class, retaining the lowest K per class.
        entropy_order = ent.argsort(stable=True)
        order = entropy_order[labels[entropy_order].argsort(stable=True)]
        sorted_labels = labels[order]
        starts = torch.searchsorted(sorted_labels, sorted_labels)
        keep = order[torch.arange(len(order), device=order.device) - starts < self.k]
        self.supports, self.labels, self.ent = supports[keep], labels[keep], ent[keep]
        return self.classify(features)
