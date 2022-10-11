import torch_audiomentations
import torch
from torch import Tensor
from torch import distributions
#from librose.effects import time_stretch, pitch_shift
from torchaudio import transforms

from hw_asr.augmentations.base import AugmentationBase
from hw_asr.augmentations.random_apply import RandomApply


class Gaussian(AugmentationBase):
    def __init__(self, mean=0, std=0.05, *args, **kwargs):
        self.noiser = distributions.Normal(mean, std)

    def __call__(self, data: Tensor):
        return torch.clamp(data + self.noiser.sample(data.size()), min=-1.0, max=1.0)
    
    
class RandomGaussian(AugmentationBase):
    def __init__(self, p, mean=0, std=0.05, *args, **kwargs):
        self._aug = RandomApply(Gaussian(mean, std), p)

    def __call__(self, data: Tensor):
        return self._aug(data)
    
class TimeStretch(AugmentationBase):
    def __init__(self, p, speed=2.0, *args, **kwargs):
        self._aug = RandomApply(transforms.TimeStretch(fixed_rate=speed), p)

    def __call__(self, data: Tensor):
        return self._aug(data)

# class PitchShifting(AugmentationBase):
#     def __init__(self, sr=16000, effort=-5, *args, **kwargs):
#         self.sr = sr
#         self.effort = effort

#     def __call__(self, data: Tensor):
#         augmented = pitch_shift(data.numpy().squeeze(), self.sr, self,effort)
#         return torch.from_numpy(augmented)
    

class Volume(AugmentationBase):
    def __init__(self, gain=2.0, *args, **kwargs):
        self._aug = transforms.Vol(gain=gain, gain_type='amplitude')
        self.effort = effort

    def __call__(self, data: Tensor):
        return self._aug(data)