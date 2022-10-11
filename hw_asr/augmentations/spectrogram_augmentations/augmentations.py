from hw_asr.augmentations.base import AugmentationBase
from torchaudio import transforms
from hw_asr.augmentations.random_apply import RandomApply
from torch import Tensor

class FreqMask(AugmentationBase):
    def __init__(self, p,  gain=10, *args, **kwargs):
        self._aug = RandomApply(transforms.FrequencyMasking(gain), p)

    def __call__(self, data: Tensor):
        return self._aug(data)
    
class TimeMask(AugmentationBase):
    def __init__(self, p, gain=10, *args, **kwargs):
        self._aug = RandomApply(transforms.TimeMasking(gain), p)

    def __call__(self, data: Tensor):
        return self._aug(data)