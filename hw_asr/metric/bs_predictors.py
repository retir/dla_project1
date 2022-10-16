from typing import List

import multiprocessing
import torch
from torch import Tensor

from pyctcdecode import build_ctcdecoder
from hw_asr.base.base_metric import BaseMetric
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.metric.utils import calc_wer

class BSPredicor:
    def __init__(self, text_encoder: BaseTextEncoder, beam_size=100, lm_path=None, **kwargs):
        self.text_encoder = text_encoder
        self.beam_size = beam_size
        vocab = list(self.text_encoder.ind2char.values())
        vocab[0] = ''
        self.decoder = build_ctcdecoder(vocab, kenlm_model_path=lm_path, **kwargs)
        
    def __call__(self, probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        logits_list = [prob[:prob_len] for prob, prob_len in zip(probs.detach().cpu().numpy(), log_probs_length.numpy())]
        with multiprocessing.get_context("fork").Pool(20) as pool:
            predictions = self.decoder.decode_batch(pool, logits_list, beam_width=self.beam_size)
        return predictions