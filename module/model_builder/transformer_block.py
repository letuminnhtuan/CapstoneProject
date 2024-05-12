import torch
from module.model_builder.multihead import MultiHeadAttention

class TransformerBlock(torch.nn.Module):
    def __init__(self, n_head, n_dim, n_expansion):
        super(TransformerBlock, self).__init__()
        # parameters
        self.n_head = n_head
        self.n_dim = n_dim
        self.n_expansion = n_expansion
        # instances
        self.multihead = MultiHeadAttention(n_head=self.n_head, n_dim=self.n_dim)
        self.norm_attention = torch.nn.LayerNorm(self.n_dim)
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(self.n_dim, self.n_expansion * self.n_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.n_expansion * self.n_dim, self.n_dim),
            # torch.nn.ReLU(),
        )
        self.norm_feedforward = torch.nn.LayerNorm(self.n_dim)

    def forward(self, query, key, value):
        multihead_vector = self.multihead(query, key, value)
        add_norm_vector = self.norm_attention(multihead_vector + query)
        feed_forward_vector = self.feedforward(add_norm_vector)
        output = self.norm_feedforward(feed_forward_vector + add_norm_vector)
        return output