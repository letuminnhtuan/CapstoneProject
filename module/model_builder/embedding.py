import torch

class Embedding(torch.nn.Module):
    def __init__(self, vocab_size, n_dim):
        super(Embedding, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_dim)

    def forward(self, x):
        out = self.embedding(x)
        return out