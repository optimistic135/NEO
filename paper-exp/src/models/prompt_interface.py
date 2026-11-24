import math

import torch
from torch import nn


class PromptInterface(nn.Module):
    """Expose the authors' shallow ViT prompt API on the loaded Hugging Face weights."""

    def __init__(self, model, num_prompts, layer_list):
        super().__init__()
        self.vit = model
        self.num_prompts = num_prompts
        self.prompt_dim = model.head.in_features
        self.layer_list = layer_list
        self.prompts = nn.Parameter(model.head.weight.new_empty(1, num_prompts, self.prompt_dim),
                                    requires_grad=False)
        self.reset()

    @torch.no_grad()
    def reset(self):
        bound = math.sqrt(6 / (3 * 16**2 + self.prompt_dim))
        self.prompts.uniform_(-bound, bound)

    def layers(self, images, prompts=None):
        vit = self.vit.network.vit
        tokens = vit.embeddings(images)
        if prompts is not None:
            tokens = torch.cat((tokens[:, :1], prompts.expand(len(images), -1, -1), tokens[:, 1:]), dim=1)
        features = []
        layers = vit.encoder.layer
        for i, layer in enumerate(layers):
            tokens = layer(tokens)[0]
            norm = layers[i + 1].layernorm_before if i + 1 < len(layers) else vit.layernorm
            features.append(norm(tokens[:, 0]))
        return features

    def layers_cls_features(self, images):
        features = self.layers(images)
        return features if self.layer_list else torch.cat(features, dim=1)

    def layers_cls_features_with_prompts(self, images):
        features = self.layers(images, self.prompts)
        return features if self.layer_list else torch.cat(features, dim=1)
