import torch
from torch import Tensor
from torch.nn import CTCLoss


class CTCLossWrapper(CTCLoss):
    def forward(self, log_probs, log_probs_length, text_encoded, text_encoded_length,
                **batch) -> Tensor:
        log_probs_t = torch.transpose(log_probs, 0, 1)

        #print(log_probs_length.shape)
        #print(log_probs_t.shape)
        #print(text_encoded.shape)
        #assert torch.isfinite(log_probs_t).all()
        #assert torch.isfinite(text_encoded).all()
        #print(log_probs_t.shape)
        #print(text_encoded.shape)
        #print(len(log_probs_length))
        #print(len(text_encoded_length))
        #print(log_probs_length.max())
        #print(log_probs.shape)
        tmp = super().forward(
            log_probs=log_probs_t,
            targets=text_encoded,
            input_lengths=log_probs_length,
            target_lengths=text_encoded_length,
        )
        #print(torch.isfinite(tmp))
        return tmp
