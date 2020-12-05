import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=1,
                          bidirectional=False, batch_first=True)

    def forward(self, input, hidden=None):
        # input, 1*seq_len
        embedded = self.embedding(input)  # 1*seq_len*hidden_size
        # output : 1*seq_len*hidden_size
        # hidden : (num_layers*bi)*hidden_size
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    # def initHidden(self):
    #     return torch.zeros(1, 1, self.hidden_size)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, dropout_p: float = 0.1, max_length: int = 10):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        # encoder_outputs max_len*hidden_size
        # embedded 1*1*hidden_size
        # hidden 1*1*hidden_size
        embedded = self.dropout(self.embedding(input))
        # attn_weights 1*max_len
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[-1]), 1)), dim=1)

        # attn_applied 1*hidden_size
        attn_applied = torch.mm(attn_weights, encoder_outputs)

        output = torch.cat((embedded[0], attn_applied), 1)
        output = self.attn_combine(output).unsqueeze(0)  # 1*1*hidden_size

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


if __name__ == "__main__":
    gru = nn.GRU(input_size=128, hidden_size=128, num_layers=5, bidirectional=True,
                 batch_first=True)
    x = torch.randn((16, 11, 128))
    o, h = gru(x)
    print(o.shape, h.shape)
