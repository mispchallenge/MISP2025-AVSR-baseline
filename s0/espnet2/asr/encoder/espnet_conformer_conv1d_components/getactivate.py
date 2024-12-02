import torch
def get_activation(act):
    """Return activation function."""
    # Lazy load to avoid unused import
    

    activation_funcs = {
        "hardtanh": torch.nn.Hardtanh,
        "tanh": torch.nn.Tanh,
        "relu": torch.nn.ReLU,
        "selu": torch.nn.SELU,
        "swish": Swish,
    }

    return activation_funcs[act]()

class Swish(torch.nn.Module):
    """Construct an Swish object."""

    def forward(self, x):
        """Return Swich activation function."""
        return x * torch.sigmoid(x)