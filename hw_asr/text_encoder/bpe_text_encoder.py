import json
from pathlib import Path
from string import ascii_lowercase
from typing import List, Union

import torch
import numpy as np
from torch import Tensor
from tokenizers import CharBPETokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from string import ascii_lowercase

from hw_asr.base.base_text_encoder import BaseTextEncoder
from .char_text_encoder import CharTextEncoder




class BPETextEncoder(BaseTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, texts_path, vocab_size):
        print('USE BPE TOKENIZER')
        self.tokenizer = CharBPETokenizer()
        self.tokenizer.train([texts_path], vocab_size=vocab_size, initial_alphabet=list(ascii_lowercase + ' '))
        self.vocab = [self.EMPTY_TOK] + list(self.tokenizer.get_vocab().keys())
        
        self.ind2char = {k: v for k, v in enumerate(self.vocab)}
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.ind2char)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> Tensor:
        text = self.normalize_text(text)
        tokenized = self.tokenizer.encode(text)
        return torch.tensor([self.char2ind[token] for token in tokenized.tokens])[None,:]
    

    def decode(self, vector: Union[Tensor, np.ndarray, List[int]]):
        token_ids = [self.tokenizer.token_to_id(self.ind2char[val]) for val in vector if val != 0]
        return self.tokenizer.decode(token_ids)
    
    def ctc_decode(self, inds: List[int]) -> str:
        decoded = []
        last_char = self.EMPTY_TOK

        for ind in inds:
            new_char = self.ind2char[ind]
            if new_char != last_char: 
                if new_char != self.EMPTY_TOK:
                    decoded.append(new_char)
                last_char = new_char
        for ch in decoded:
            assert ch is not None
        return self.tokenizer.decode([self.tokenizer.token_to_id(tok) for tok in decoded])

    def dump(self, file):
        with Path(file).open('w') as f:
            json.dump(self.ind2char, f)

    @classmethod
    def from_file(cls, file):
        with Path(file).open() as f:
            ind2char = json.load(f)
        a = cls([])
        a.ind2char = ind2char
        a.char2ind = {v: k for k, v in ind2char}
        return a

class BPETextEncoder2(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, tokenizer_path):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        alphabet = sorted(list(self.tokenizer.get_vocab().keys()))
        super().__init__(alphabet)
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        print(self.ind2char)

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
                        beam_size: int = 100):
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(log_probs.shape) == 2
        char_length, voc_size = log_probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []

        hypos = {('', self.EMPTY_TOK) : 1.0}
        for next_char_probs in log_probs[:probs_length]:
            hypos = self.extend_and_merge(next_char_probs, hypos)
            hypos = dict(sorted(hypos.items(), key=lambda x: x[1])[-beam_size:]) #self.truncate_beam(hypos, beam_size)

        to_return = [(prefix, score) for (prefix, _), score in sorted(hypos.items(), key=lambda x: -x[1])]
        assert to_return[0][1] >= to_return[1][1]
        return to_return


    def extend_and_merge(self, next_char_log_probs, src_hypos):
        new_hypos = defaultdict(float)
        for next_char_ind, next_char_log_prob in enumerate(next_char_log_probs):
            next_char = self.ind2char[next_char_ind]

            for (text, last_char), hypo_log_prob in src_hypos.items():
                new_prefix = text if next_char == last_char else (text + next_char)
                new_prefix = new_prefix.replace(self.EMPTY_TOK, '')
                new_hypos[(new_prefix, next_char)] += hypo_log_prob * next_char_log_prob
        
        return new_hypos
    
    def truncate_beam(hypos, beam_size):
        return dict(sorted(hypos.items(), key=lambda x: x[1])[:beam_size])