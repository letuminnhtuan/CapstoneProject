import torch
from module.model_builder.encoder import EncoderBlock
from module.model_builder.decoder import DecoderBlock
class Model(torch.nn.Module):
    def __init__(self, n_dim_model,
        input_channel_encoder, hidden_dim_encoder, n_head_encoder, n_expansion_encoder, n_layer_encoder,
        n_head_decoder, seq_length_decoder, vocab_size_decoder, n_expansion_decoder, n_layer_decoder,
    ):
        super(Model, self).__init__()
        # Parameters
        self.n_dim_model = n_dim_model
        self.input_chanel_encoder = input_channel_encoder
        self.hidden_dim_encoder = hidden_dim_encoder
        self.n_head_encoder = n_head_encoder
        self.n_expansion_encoder = n_expansion_encoder
        self.n_layer_encoder = n_layer_encoder
        self.n_head_decoder = n_head_decoder
        self.seq_length_decoder = seq_length_decoder
        self.vocab_size_decoder = vocab_size_decoder
        self.n_expansion_decoder = n_expansion_decoder
        self.n_layer_decoder = n_layer_decoder
        # Instances
        self.encoder = EncoderBlock(self.input_chanel_encoder, self.hidden_dim_encoder, self.n_dim_model,  self.n_head_encoder, self.n_expansion_encoder, self.n_layer_encoder)
        self.decoder = DecoderBlock(self.n_head_decoder, self.n_dim_model, self.seq_length_decoder, self.vocab_size_decoder, self.n_expansion_decoder, self.n_layer_decoder)

    @staticmethod
    def create_mask(seq_input):
        batch_size, seq_len = seq_input.shape
        mask = torch.tril(torch.ones((seq_len, seq_len)))
        return mask.expand(batch_size, 1, seq_len, seq_len)

    def forward(self, input_encoder, input_decoder):
        output_encoder = self.encoder(input_encoder)
        mask = self.create_mask(input_decoder).to('cuda' if torch.cuda.is_available() else 'cpu')
        output_decoder = self.decoder(output_encoder, input_decoder, mask)
        return output_decoder