from typing import List, NamedTuple
from collections import defaultdict

import torch
import numpy as np

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        decoded = []
        last_char = self.EMPTY_TOK

        for ind in inds:
            new_char = self.ind2char[ind]
            if new_char != last_char: 
                if new_char != self.EMPTY_TOK:
                    decoded.append(new_char)
                last_char = new_char
        return ''.join(decoded)

    def ctc_beam_search(self, log_probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(log_probs.shape) == 2
        char_length, voc_size = log_probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []
        # TODO: your code here

        hypos = {('', self.EMPTY_TOK) : 1.0}
        for next_char_probs in log_probs[:probs_length]:
            hypos = self.extend_and_merge(next_char_probs, hypos)
            hypos = dict(sorted(hypos.items(), key=lambda x: x[1])[-beam_size:]) #self.truncate_beam(hypos, beam_size)

        to_return = [(prefix, score) for (prefix, _), score in sorted(hypos.items(), key=lambda x: -x[1])]
        assert to_return[0][1] >= to_return[1][1]
        #print()
        #for i in range(3):
        #    print(to_return[i])
       # print()
        return to_return
        #raise NotImplementedError
        #return sorted(hypos, key=lambda x: x.prob, reverse=True)

    def extend_and_merge(self, next_char_log_probs, src_hypos):
        new_hypos = defaultdict(float)
        for next_char_ind, next_char_log_prob in enumerate(next_char_log_probs):
            next_char = self.ind2char[next_char_ind]

            for (text, last_char), hypo_log_prob in src_hypos.items():
                new_prefix = text if next_char == last_char else (text + next_char)
                new_prefix = new_prefix.replace(self.EMPTY_TOK, '')
                new_hypos[(new_prefix, next_char)] += hypo_log_prob * next_char_log_prob #np.log(np.exp(new_hypos[(new_prefix, last_char)]) + np.exp(hypo_log_prob + next_char_log_prob))
        
        return new_hypos
    
    def truncate_beam(hypos, beam_size):
        return dict(sorted(hypos.items(), key=lambda x: x[1])[:beam_size])
