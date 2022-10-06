import logging
import torch
import torch.nn.functional as F
from typing import List
from collections import defaultdict

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}
    result_batch['text_encoded_length'] = []
    result_batch['text'] = []
    result_batch['spectrogram'] = []
    result_batch['text_encoded'] = []
    result_batch['spectrogram_length'] = []
    result_batch['audio_path'] = []
    result_batch['duration'] = []
    result_batch['audio'] = []

    #print(dataset_items[0].keys())
    max_spect_lenght = 0
    max_textenc_lenght = 0
    for item in dataset_items:
        max_spect_lenght = max(max_spect_lenght, item['spectrogram'].shape[2])
        max_textenc_lenght = max(max_textenc_lenght, item['text_encoded'].shape[1])
        result_batch['text_encoded_length'].append(item['text_encoded'].shape[1])
        result_batch['text'].append(item['text'])
        result_batch['spectrogram_length'].append(item['spectrogram'].shape[2])
        result_batch['audio_path'].append(item['audio_path'])
        result_batch['duration'].append(item['duration'])
        result_batch['audio'].append(item['audio'])

    for item in dataset_items:
        assert torch.isfinite(item['spectrogram']).all()
        assert torch.isfinite(item['text_encoded']).all()
        padded_text_enc = F.pad(item['text_encoded'], (0, max_textenc_lenght  - item['text_encoded'].shape[1], 0, 0), mode='constant', value=0)
        result_batch['text_encoded'].append(padded_text_enc)

        padded_spec = F.pad(item['spectrogram'], (0, max_spect_lenght - item['spectrogram'].shape[2], 0, 0, 0, 0), mode='constant', value=0)[0]
        result_batch['spectrogram'].append(padded_spec)
    
    result_batch['spectrogram'] = torch.stack(result_batch['spectrogram'], dim=0)
    result_batch['text_encoded'] = torch.vstack(result_batch['text_encoded'])
    result_batch['text_encoded_length'] = torch.tensor(result_batch['text_encoded_length'])
    result_batch['spectrogram_length'] = torch.tensor(result_batch['spectrogram_length'])

    assert torch.isfinite(result_batch['spectrogram']).all()
    assert torch.isfinite(result_batch['text_encoded']).all()
  

    return result_batch
