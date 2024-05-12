import torch
from module.model_builder.transformer_block import TransformerBlock
from module.model_builder.multihead import MultiHeadAttention
from module.model_builder.embedding import Embedding
from module.model_builder.positional import Positional_Encoding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class DecoderLayer(torch.nn.Module):
    def __init__(self, n_head, n_dim, seq_length, vocab_size, n_expansion):
        super(DecoderLayer, self).__init__()
        # parameters
        self.n_head = n_head
        self.n_dim = n_dim
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.n_expansion = n_expansion
        # instance
        self.mask_attention = MultiHeadAttention(n_head=self.n_head, n_dim=self.n_dim)
        self.norm_mask_attention = torch.nn.LayerNorm(self.n_dim)
        self.transformer_block = TransformerBlock(self.n_head, self.n_dim, self.n_expansion)

    def forward(self, x, key, value, mask):
        masked_output = self.mask_attention(query=x, key=x, value=x, mask=mask)
        masked_output = self.norm_mask_attention(x + masked_output)
        output = self.transformer_block(query=masked_output, key=key, value=value)
        return output

class DecoderBlock(torch.nn.Module):
    def __init__(self, n_head, n_dim, seq_length, vocab_size, n_expansion, n_layer):
        super(DecoderBlock, self).__init__()
        # parameters
        self.n_head = n_head
        self.n_dim = n_dim
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.n_expansion = n_expansion
        self.n_layer = n_layer
        # instance
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding = Embedding(vocab_size=self.vocab_size, n_dim=self.n_dim)
        self.decoder_layers = torch.nn.ModuleList(
            [DecoderLayer(self.n_head, self.n_dim, self.seq_length, self.vocab_size, self.n_expansion).to(self.device) for _ in range(self.n_layer)]
        )
        self.fc_output = torch.nn.Linear(self.n_dim, self.vocab_size)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, output_encoder, input_decoder, mask):
        embedding_vector = self.embedding(input_decoder)
        position = Positional_Encoding(seq_length=embedding_vector.shape[1], n_dim=self.n_dim)
        embedding_vector = embedding_vector + position().requires_grad_(False).to(device)
        output = self.decoder_layers[0](x=embedding_vector, key=output_encoder, value=output_encoder, mask=mask)
        for decoder_layer in self.decoder_layers[1:]:
            output = decoder_layer(x=output, key=output_encoder, value=output_encoder, mask=mask)
            output = self.dropout(output)
        return self.fc_output(output)