from torch import nn
from torch.nn import Sequential
import math

from hw_asr.base import BaseModel


class DeepSpeechV1(BaseModel):
    def __init__(self, n_class, hidden_size=256, rnn_input_size=1024, bidirectional=True, hidden_layers=5, **batch):
        super().__init__(None, n_class, **batch)

        self.bidirectional = bidirectional
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        )


        self.rnns = nn.Sequential(
            nn.LSTM(rnn_input_size, hidden_size, bidirectional=bidirectional),
            *(
                nn.LSTM(hidden_size, hidden_size, bidirectional=bidirectional) for x in range(hidden_layers - 1)
            )
        )

        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, n_class, bias=False)
        )


    def forward(self, spectrogram, spectrogram_length, **batch):
        output_lenghts = self.transform_input_lengths(spectrogram_length)
        
        x = spectrogram[:, None, :, :]
        x = self.conv(x)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()

        for rnn in self.rnns:
            x = nn.utils.rnn.pack_padded_sequence(x, output_lenghts, enforce_sorted=False)
            x, h = rnn(x)
            x, _ = nn.utils.rnn.pad_packed_sequence(x)
            if self.bidirectional:
                x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)
        

        s0, s1 = x.size(0), x.size(1)
        x = self.fc(x.view(s0 * s1, -1))
        x = x.view(s0, s1, -1)
        x = x.transpose(0, 1)

        return {"logits": x}

    def transform_input_lengths(self, input_lengths):
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) // m.stride[1] + 1)
        return seq_len.int()