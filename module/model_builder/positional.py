import torch
import math

class Positional_Encoding(torch.nn.Module):
    def __init__(self, seq_length, n_dim):
        super(Positional_Encoding, self).__init__()
        self.seq_length = seq_length
        self.n_dim = n_dim

    def forward(self):
        # positional vector
        position_encode = torch.zeros((self.seq_length, self.n_dim))
        for pos in range(self.seq_length):
            for i in range(0, self.n_dim, 2):
                position_encode[pos, i] = math.sin(pos / (10000 ** (2 * i / self.n_dim)))
                position_encode[pos, i+1] = math.cos(pos / (10000 ** (2 * i / self.n_dim)))
        return position_encode