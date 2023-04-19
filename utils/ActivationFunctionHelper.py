import torch.nn as nn

def get_activation(name, *args, **kwargs):
    activation_constructors = {
        'tanh': nn.Tanh(),
        'relu': nn.ReLU(inplace=False),
        'sigmoid': nn.Sigmoid(),
        'celu': nn.CELU(alpha=1.0, inplace=False),
        'softmax': nn.Softmax(dim=None), # It's recommended to explicitly set the `dim` parameter
        'leakyrelu': nn.LeakyReLU(negative_slope=0.01, inplace=False),
        'prelu': nn.PReLU(num_parameters=1, init=0.25),
        'elu': nn.ELU(alpha=1.0, inplace=False),
        'gelu': nn.GELU(),
        'swish': nn.SiLU(inplace=False),
        'relu6': nn.ReLU6(inplace=False),
        'hardshrink': nn.Hardshrink(lambd=0.5),
        'hardtanh': nn.Hardtanh(min_val=-1.0, max_val=1.0, inplace=False),
        'softplus': nn.Softplus(beta=1, threshold=20),
        'logsigmoid': nn.LogSigmoid(),
        'tanhshrink': nn.Tanhshrink(),
        'softshrink': nn.Softshrink(lambd=0.5),
        'softsign': nn.Softsign(),
        'rrelu': nn.RReLU(lower=1/8, upper=1/3, inplace=False),
        'hardswish': nn.Hardswish(inplace=False),
        'logsoftmax': nn.LogSoftmax(dim=None), # It's recommended to explicitly set the `dim` parameter
        'softmin': nn.Softmin(dim=None), # It's recommended to explicitly set the `dim` parameter
    }

    constructor = activation_constructors.get(name)
    if constructor is None:
        raise ValueError(f"Unsupported activation function: {name}")

    return constructor
