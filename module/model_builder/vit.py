import torch
from module.model_builder.positional import Positional_Encoding
from module.model_builder.transformer_block import TransformerBlock

class ViT(torch.nn.Module):
    def __init__(self, input_chanel, output_chanel, n_head, n_expansion, n_layer):
        super(ViT, self).__init__()
        # Parameters
        self.input_chanel = input_chanel
        self.output_chanel = output_chanel
        self.n_head = n_head
        self.n_expansion = n_expansion
        self.n_layer = n_layer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Instance
        self.patch_embedding = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.input_chanel, out_channels=self.output_chanel, kernel_size=32, stride=32, padding=0),
            torch.nn.BatchNorm2d(self.output_chanel),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5)
        )
        self.skip_connection = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.output_chanel, out_channels=self.output_chanel, kernel_size=1, stride=1, padding=0),
            torch.nn.BatchNorm2d(self.output_chanel),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.15)
        )
        self.dropout = torch.nn.Dropout(0.2)
        self.flatten = torch.nn.Flatten(2)
        self.transformer_block = TransformerBlock(self.n_head, self.output_chanel, self.n_expansion)

    def add_cls_token(self, x):
        batch_size = x.shape[0]
        cls_token = torch.nn.Parameter(data=torch.zeros(batch_size, 1, self.output_chanel), requires_grad=True).to(self.device)
        return torch.concat([cls_token, x], dim=1)

    def forward(self, x):
        """ Input shape: (batch_size, chanel, height, width) """
        x = self.patch_embedding(x)     # => (batch_size, seq_len, output_chanel)
        x = self.flatten(x)
        x = x.transpose(-1, -2)
        x = self.add_cls_token(x)       # => (batch_size, seq_len+1, output_chanel)
        position = Positional_Encoding(seq_length=x.shape[1], n_dim=self.output_chanel)
        x = x + position().requires_grad_(False).to(self.device)
        for _ in range(self.n_layer):
            x = self.transformer_block(x, x, x)
            # x = self.dropout(x)
        return x