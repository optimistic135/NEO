import torch

from src.methods.base import Adapter
from src.models.prompt_interface import PromptInterface
from src.utils.official import module
from src.utils.source_stats import source_stats


class PromptSearch(Adapter):
    @property
    def trainable_parameters(self):
        return self.prompt.prompts.numel()

    def __init__(self, model, cfg, context):
        super().__init__(model)
        self.is_fozo = cfg["id"] == "fozo"
        self.prompt = PromptInterface(self.model, cfg["num_prompts"], layer_list=self.is_fozo)
        name = "FOZO" if self.is_fozo else "FOA"
        replacements = [
            ("from models.vpt import PromptViT " if self.is_fozo else "from models.vpt import PromptViT", ""),
            ("from quant_library.quant_layers.matmul import *", ""),
            ("nn.MSELoss(reduction='none').cuda()", "nn.MSELoss(reduction='none')"),
        ]
        if not self.is_fozo:
            replacements.extend([
                ("from utils.cli_utils import accuracy, AverageMeter", ""),
                ("from calibration_library.metrics import ECELoss", ""),
                ("reshape_as(self.model.prompts).cuda()", "reshape_as(self.model.prompts).to(x.device)"),
            ])
        official = module(cfg, f"tta_library/{name if self.is_fozo else name.lower()}.py",
                          replacements, PromptViT=PromptInterface)
        value, self.source_seconds = source_stats(self.prompt, context)
        if self.is_fozo:
            self.core = official.FOZO(self.prompt, cfg["zo_eps"], cfg["lr"], cfg["fitness_lambda"], cfg["n_spsa"], log_dir=None)
            self.core.train_info = dict(first_half=tuple(t[:4608] for t in value),
                                        second_half=tuple(t[4608:] for t in value), combined_stats=value)
            self.timing = "best-current-perturbation-before-prompt-update"
            self.passes = 2 * cfg["n_spsa"]
        else:
            self.core = official.FOA(self.prompt, cfg["fitness_lambda"])
            self.core.train_info = value
            self.core.imagenet_mask = None
            self.timing = "best-current-candidate-before-population-update"
            self.passes = self.core.popsize + 1

    def reset(self):
        super().reset()
        self.core.reset()

    @torch.no_grad()
    def update(self, images):
        logits = self.core(images)
        self.forward_passes += self.passes
        return logits

    @torch.no_grad()
    def predict(self, images):
        prompt = self.prompt.prompts if self.is_fozo else self.core.best_prompts
        features = self.prompt.layers(images, prompt)[-1]
        shift = self.core._get_shift_vector()
        self.forward_passes += 1
        return self.model.head(features if shift is None else features + shift)
