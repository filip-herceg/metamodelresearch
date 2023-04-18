import torch.nn as nn

from utils.ActivationFunctionHelper import get_activation


class CustomModel(nn.Module):
    def __init__(self, input_shape, output_shape, layers_config=None, dropout_probability=0.2):
        super(CustomModel, self).__init__()

        # THIS SHOULD IN THEORY NEVER HAPPEN
        if layers_config is None:
            layers_config = [('Dense', 64, 'relu'), ('Dropout', dropout_probability), ('Dense', 32, 'relu')]
        # UNTIL HERE

        layers = []
        prev_layer_size = input_shape

        for layer_type, *args in layers_config:
            if layer_type == 'Dense':
                layer_size = args[0] if args else 64  # default value of 64
                activation_function = args[1] if len(args) > 1 else 'relu'  # default value of 'relu'
                layers.append(nn.Linear(prev_layer_size, layer_size))
                layers.append(get_activation(activation_function))
                prev_layer_size = layer_size

            elif layer_type == 'Dropout':
                dropout_probability = args[0] if args else dropout_probability  # default value of 0.2
                layers.append(nn.Dropout(dropout_probability))

        layers.append(nn.Linear(prev_layer_size, output_shape))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
