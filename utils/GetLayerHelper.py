import torch.nn as nn

def get_layer(name, *args, **kwargs):
    layers = {
        'linear': nn.Linear,
        'conv1d': nn.Conv1d,
        'conv2d': nn.Conv2d,
        'conv3d': nn.Conv3d,
        'convtranspose1d': nn.ConvTranspose1d,
        'convtranspose2d': nn.ConvTranspose2d,
        'convtranspose3d': nn.ConvTranspose3d,
        'maxpool1d': nn.MaxPool1d,
        'maxpool2d': nn.MaxPool2d,
        'maxpool3d': nn.MaxPool3d,
        'avgpool1d': nn.AvgPool1d,
        'avgpool2d': nn.AvgPool2d,
        'avgpool3d': nn.AvgPool3d,
        'batchnorm1d': nn.BatchNorm1d,
        'batchnorm2d': nn.BatchNorm2d,
        'batchnorm3d': nn.BatchNorm3d,
        'instancenorm1d': nn.InstanceNorm1d,
        'instancenorm2d': nn.InstanceNorm2d,
        'instancenorm3d': nn.InstanceNorm3d,
        'layernorm': nn.LayerNorm,
        'groupnorm': nn.GroupNorm,
        'dropout': nn.Dropout,
        'dropout2d': nn.Dropout2d,
        'dropout3d': nn.Dropout3d,
        'embedding': nn.Embedding,
        'rnn': nn.RNN,
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'transformer': nn.Transformer,
        'transformerencoder': nn.TransformerEncoder,
        'transformerdecoder': nn.TransformerDecoder,
        'identity': nn.Identity,
        'pixelshuffle': nn.PixelShuffle,
        'upsample': nn.Upsample,
        'pad': nn.ZeroPad2d,
        'adaptiveavgpool1d': nn.AdaptiveAvgPool1d,
        'adaptiveavgpool2d': nn.AdaptiveAvgPool2d,
        'adaptiveavgpool3d': nn.AdaptiveAvgPool3d,
        'adaptivemaxpool1d': nn.AdaptiveMaxPool1d,
        'adaptivemaxpool2d': nn.AdaptiveMaxPool2d,
        'adaptivemaxpool3d': nn.AdaptiveMaxPool3d,
    }

    layer = layers.get(name)
    if layer is None:
        raise ValueError(f"Unsupported layer type: {name}")

    return layer(*args, **kwargs)
