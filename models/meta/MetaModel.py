from skopt.space import Categorical, Integer, Real
from torch import nn


class MetaModel(nn.Module):
    def __init__(self, input_shape, output_shape, train_loader, val_loader, max_layers=5, n_calls=20,
                 search_space=None):
        super().__init__()
        if search_space is None:
            self.search_space = self.create_layer_space(max_layers)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_layers = max_layers
        self.n_calls = n_calls
        self.space = search_space

    def _forward(self, x):
        # TODO: Implement this method
        pass

    def _trainAndEvaluate(self, layers_config, dropout_probability):
        # TODO: Implement this method
        pass

    def create_layer_space(self, num_layers, activation_functions=None, layer_types=None):

        if activation_functions == None:
            self.activation_functions = ['relu', 'sigmoid', 'tanh']

        if layer_types == None:
            self.max_layers =['Dense', 'Dropout']

        space = []
        for i in range(num_layers):
            space += [
                Categorical(layer_types, name=f'layer_type_{i + 1}'),
                Integer(16, 512, name=f'layer_size_{i + 1}'),
                Categorical(activation_functions + [None], name=f'activation_function_{i + 1}')
            ]
        space += [Real(0, 1, name='dropout_probability')]
        return space

    @staticmethod
    def get_activation(name):
        if name == 'relu':
            return nn.ReLU()
        elif name == 'sigmoid':
            return nn.Sigmoid()
        elif name == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {name}")


