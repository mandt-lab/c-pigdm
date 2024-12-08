import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import DictConfig


# TODO 1: Update this to remove the Classifier guidance part since we dont use that in this work.
# TODO 2: Update the DDPM and VP-SDE modules to have an abstract class Diffusion
class ClassifierGuidanceModel:
    def __init__(self, model: nn.Module, classifier: nn.Module, diffusion: object, cfg: DictConfig):
        self.model = model
        self.classifier = classifier
        self.diffusion = diffusion
        self.cfg = cfg

    def __call__(self, xt, y, t, scale=1.0):
        # Returns both the noise value (score function scaled) and the predicted x0.
        if self.classifier is None:
            y = y if self.cfg.model.class_cond else None
            et = self.model(xt, t, y)[:, :3]
        else:
            # NOTE: Hard coded to work with DDPM only.
            # Don't actually need this in this work.
            alpha_t = self.diffusion.alpha(t).view(-1, 1, 1, 1)
            et = self.model(xt, t, y)[:, :3]
            et = et - (1 - alpha_t).sqrt() * self.cond_fn(xt, y, t, scale=scale)
        # x0_pred = self.diffusion.predict_x_from_eps(xt, et, t)
        # return et, x0_pred
        return et

    def cond_fn(self, xt, y, t, scale=1.0):
        with torch.enable_grad():
            x_in = xt.detach().requires_grad_(True)
            logits = self.classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]

            scale = scale * self.cfg.classifier.classifier_scale
            return torch.autograd.grad(selected.sum(), x_in, create_graph=True)[0] * scale