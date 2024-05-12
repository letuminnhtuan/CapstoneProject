import torch
import math
import torch.nn.functional as F


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_head, n_dim):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.n_dim = n_dim
        self.n_dim_each_head = int(self.n_dim / self.n_head)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # init query, key, value
        self.flat = torch.nn.Flatten(-2)
        self.query_matrix = torch.nn.Linear(self.n_dim_each_head, self.n_dim_each_head, bias=False)
        self.key_matrix = torch.nn.Linear(self.n_dim_each_head, self.n_dim_each_head, bias=False)
        self.value_matrix = torch.nn.Linear(self.n_dim_each_head, self.n_dim_each_head, bias=False)
        self.output_matrix = torch.nn.Linear(self.n_dim_each_head * self.n_head, self.n_dim_each_head * self.n_head, bias=False)

    def forward(self, query, key, value, mask=None):  # (batch_size, seq_length, n_dim)
        batch_size = key.size(0)
        seq_length = key.size(1)
        seq_length_query = query.size(1)
        # divide head => (batch_size, seq_length, n_head, n_dim_each_head)
        query = query.view(batch_size, seq_length_query, self.n_head, self.n_dim_each_head)
        key = key.view(batch_size, seq_length, self.n_head, self.n_dim_each_head)
        value = value.view(batch_size, seq_length, self.n_head, self.n_dim_each_head)
        q = self.query_matrix(query)
        k = self.key_matrix(key)
        v = self.value_matrix(value)
        # transpose => (batch_size, n_head, seq_length, n_dim_each_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # -------------------------- Compute MultiHead-Attention --------------------------
        """
        - Step 1: compute matmul(q, k^T)
        - Step 2: scale with sqrt(n_dim)
        - Step 3: compute softmax => matrix A
        - Step 4: compute matmul of matrix A and value matrix
        - Step 5: concatenate matrix => matrix Z
        - Step 4: compute matmul of matrix Z and matrix W0
        """
        k_T = k.transpose(-1, -2)  # => (batch_size, n_head, n_dim_each_head, seq_length)
        product = torch.matmul(q, k_T)  # => (batch_size, n_head, seq_length_query, seq_length)
        product = product / math.sqrt(self.n_dim_each_head)
        if mask is not None:
            product = product.masked_fill(mask == 0, float("-1e20"))
        product = product.to(self.device)
        scores = F.softmax(product, dim=-1)  # => (batch_size, n_head, seq_length_query, seq_length)
        scores = torch.matmul(scores, v)  # => (batch_size, n_head, seq_length_query, n_dim_each_head)
        scores = scores.transpose(1, 2)  # => (batch_size, seq_length_query, n_head, n_dim_each_head)
        scores = self.flat(scores)
        output = self.output_matrix(scores)
        return output
