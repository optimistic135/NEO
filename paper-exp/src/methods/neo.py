import torch

from src.methods.base import Adapter


class NEO(Adapter):
    def reset(self):
        super().reset()
        self.count = 0
        self.center = self.model.head.weight.new_zeros(self.model.head.in_features)

    def predict(self, images):
        return self.model.head(self.features(images) - self.center)

    @torch.no_grad()
    def update(self, images):
        features = self.features(images)
        total = self.count + len(features)
        self.center = (self.count / total) * self.center + (len(features) / total) * features.mean(0)
        self.count = total
        return self.model.head(features - self.center)
