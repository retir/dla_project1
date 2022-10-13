import torchaudio
import torch
import numpy as np
from torch import Tensor
from torch import distributions
#from librose.effects import time_stretch, pitch_shift
from torchaudio import transforms

from hw_asr.augmentations.base import AugmentationBase
from hw_asr.augmentations.random_apply import RandomApply

class Empty(AugmentationBase):
    def __init__(*args, **kwargs):
        pass
    def __call__(self, data, *args, **kwargs):
        return data

class Gaussian(AugmentationBase):
    def __init__(self, mean=0, std=0.01, *args, **kwargs):
        self.noiser = distributions.Normal(mean, std)

    def __call__(self, data: Tensor):
        return torch.clamp(data + self.noiser.sample(data.size()), min=-1.0, max=1.0)
    
    
class RandomGaussian(AugmentationBase):
    def __init__(self, p, mean=0, std=0.01, *args, **kwargs):
        self.aug = RandomApply(Gaussian(mean, std), p)

    def __call__(self, data: Tensor):
        return self.aug(data)

class PT(AugmentationBase):
    def __init__(self, sr=16000, *args, **kwargs):
        try:
            self.aug = torchaudio.transforms.PitchShift
        except:
            print('Cannot find PitchShift, skipp')
            self.aug = Empty
        self.sr = sr

    def __call__(self, data, *args, **kwargs):
        value = np.random.randint(-5, 5)
        aug = self.aug(self.sr, value)
        res = aug(data)
        aug = None # free memmory
        return res


class PT_class(AugmentationBase):
    def __init__(self, sr=16000, *args, **kwargs):
        self.aug = torchaudio.transforms.PitchShift(16000, -1)

    def __call__(self, data, *args, **kwargs):
        #value = np.random.randint(-5, 5)
        #aug = self.aug(self.sr, value)
        #res = aug(data)
        #aug = None # free memmory
        return self.aug(data)

class RandomPT(AugmentationBase):
    def __init__(self, p, sr=16000, *args, **kwargs):
        self.aug = RandomApply(PT(sr), p)

    def __call__(self, data, *args, **kwargs):
        return self.aug(data)


class Volume(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self.aug = transforms.Vol

    def __call__(self, data, *args, **kwargs):
        value = np.random.uniform(0.5, 2)
        aug = self.aug(value)
        res = aug(data)
        aug = None # free memmory
        return res

class RandomVolume(AugmentationBase):
    def __init__(self, p, *args, **kwargs):
        self.aug = RandomApply(Volume(), p)

    def __call__(self, data, *args, **kwargs):
        return self.aug(data)