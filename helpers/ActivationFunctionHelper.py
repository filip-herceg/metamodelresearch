from torch import nn

def get_activation(name):
    activations = {
        'tanh': nn.Tanh(),
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'celu': nn.CELU(),
        'softmax': nn.Softmax(dim=1),
        'leakyrelu': nn.LeakyReLU(negative_slope=0.01),
        'prelu': nn.PReLU(num_parameters=1),
        'elu': nn.ELU(alpha=1.0),
        'gelu': nn.GELU(),
        'swish': nn.SiLU(),
        'relu6': nn.ReLU6(),
        'hardshrink': nn.Hardshrink(lambd=0.5),
        'hardtanh': nn.Hardtanh(min_val=-1.0, max_val=1.0),
        'softplus': nn.Softplus(beta=1, threshold=20),
        'logsigmoid': nn.LogSigmoid(),
        'tanhshrink': nn.Tanhshrink(),
        'softshrink': nn.Softshrink(lambd=0.5),
        'softsign': nn.Softsign(),
        'rrelu': nn.RReLU(lower=0.125, upper=0.3333333333333333),
        'hardswish': nn.Hardswish(),
        'logsoftmax': nn.LogSoftmax(dim=1),
        'softmin': nn.Softmin(dim=1)
    }

    activation = activations.get(name)
    if activation is None:
        raise ValueError(f"Unsupported activation function: {name}")

    return activation