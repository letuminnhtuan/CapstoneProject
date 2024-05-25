import torch
# from module.model_builder.vit import ViT
from module.model_builder.vit_v2 import ViT

class EncoderBlock(torch.nn.Module):
    def __init__(self, input_chanel, hidden_dim, output_chanel, n_head, n_expansion, n_layer):
        super(EncoderBlock, self).__init__()
        # Parameters
        self.input_chanel = input_chanel
        self.hidden_dim = hidden_dim
        self.output_chanel = output_chanel
        self.n_head = n_head
        self.n_expansion = n_expansion
        self.n_layer = n_layer
        # Instances
        self.vit = ViT(self.input_chanel, self.hidden_dim, self.n_head, self.n_expansion, self.n_layer)
        self.fc_out = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.output_chanel),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )

    def forward(self, x):
        x = self.vit(x)
        x = self.fc_out(x)
        return x