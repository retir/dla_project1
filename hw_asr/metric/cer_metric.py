from typing import List

import multiprocessing
import torch
from torch import Tensor

from pyctcdecode import build_ctcdecoder
from hw_asr.base.base_metric import BaseMetric
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.metric.utils import calc_cer


class ArgmaxCERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        cers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            else:
                pred_text = self.text_encoder.decode(log_prob_vec[:length])
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)
    
    
class BSCERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        

    def __call__(self, probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        cers = []
        
        predictions = [self.text_encoder.ctc_beam_search(prob, prob_len, 10)[0][0] for prob, prob_len in zip(probs.detach().cpu().numpy(), log_probs_length.numpy())]
        lengths = log_probs_length.detach().numpy()
        for pred, length, target_text in zip(predictions, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            cers.append(calc_cer(target_text, pred[:length]))
        return sum(cers) / len(cers)
    

class MCERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, beam_size=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size
        vocab = list(self.text_encoder.ind2char.values())
        vocab[0] = ''
        self.decoder = build_ctcdecoder(vocab)

    def __call__(self, probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        cers = []
        logits_list = [prob[:prob_len] for prob, prob_len in zip(probs.detach().cpu().numpy(), log_probs_length.numpy())]
        with multiprocessing.get_context("fork").Pool(20) as pool:
            predictions = self.decoder.decode_batch(pool, logits_list, beam_width=self.beam_size)
        lengths = log_probs_length.detach().numpy()
        for pred, length, target_text in zip(predictions, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            cers.append(calc_cer(target_text, pred[:length]))
        return sum(cers) / len(cers)


class FastCERMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_bs_pred = True

    def __call__(self, predictions, log_probs_length: Tensor, text: List[str], **kwargs):
        cers = []
        lengths = log_probs_length.detach().numpy()
        for pred, length, target_text in zip(predictions, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            cers.append(calc_cer(target_text, pred[:length]))
        return sum(cers) / len(cers)

