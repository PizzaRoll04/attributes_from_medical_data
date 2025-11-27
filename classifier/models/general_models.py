import torch.nn as nn

activations = {
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "silu": nn.SiLU(),
    "tanh": nn.Tanh(),
    "leakyrelu": nn.LeakyReLU(),
}

class MLP(nn.Modules):
    def __init__(
            self, 
            dim_input, 
            dim_output, 
            dims_hidden=[256, 128, 64], 
            activation="relu", 
            dropout = 0.0,
            use_batchnorm=False,
            use_layernorm=False
        ):
        super().__init__()

        activation = activations[activation.lower()]
        layers = []
        prev = dim_input

        for dim_hidden in dims_hidden:
            layers.append(nn.Linear(prev, dim_hidden))

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(dim_hidden))
            if use_layernorm:
                layers.append(nn.LayerNorm(dim_hidden))
        
            layers.append(activation)

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev = dim_hidden

        layers.append(nn.Linear(prev, dim_output))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)