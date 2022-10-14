import json
from pathlib import Path
from string import ascii_lowercase
from typing import List, Union

import torch
import numpy as np
from torch import Tensor
from tokenizers import CharBPETokenizer

from hw_asr.base.base_text_encoder import BaseTextEncoder


class BPETextEncoder(BaseTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, texts_path, vocab_size):
        print('USE BPE TOKENIZER')
        self.tokenizer = CharBPETokenizer()
        self.tokenizer.train([texts_path], vocab_size=vocab_size, initial_alphabet=[' '])
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
        assert 1 == 2, 'DUMP'
        with Path(file).open('w') as f:
            json.dump(self.ind2char, f)

    @classmethod
    def from_file(cls, file):
        assert 1 == 2, 'FROM FILE'
        with Path(file).open() as f:
            ind2char = json.load(f)
        a = cls([])
        a.ind2char = ind2char
        a.char2ind = {v: k for k, v in ind2char}
        return a
