import abc

from omegaconf import DictConfig
from models.classifier_guidance_model import ClassifierGuidanceModel


class Sampler(abc.ABC):
    """The abstract class for a Sampler algorithm."""

    def __init__(self, model: ClassifierGuidanceModel, cfg: DictConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.sde = self.model.diffusion

    @abc.abstractmethod
    def predictor_update_fn(self):
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self):
        raise NotImplementedError

    def corrector_update_fn(self, x, t, dt):
        # No corrector applied by default
        return x, x
    
    def denoising_fn(self, x, t, dt):
        return x
