from skopt.space import Categorical, Integer, Real
import torch
import torch.nn as nn

"""
    This class is used to create a MetaModel that will be used to generate FeedforwardModels.
    The MetaModel will be trained on the performance of the FeedforwardModels it generates.
    The MetaModel will then be used to generate new FeedforwardModels that will be trained
    on the data. The performance of the new FeedforwardModels will be used to update the
    MetaModel. This process will continue until the MetaModel is able to generate
    FeedforwardModels that perform well on the data.
"""


class MetaModel(nn.Module):
    """
        This method initializes the MetaModel class.

        :param input_shape: The shape of the input data.
        :param output_shape: The shape of the output data.
        :param train_loader: The training data loader.
        :param val_loader: The validation data loader.
        :param max_layers: The maximum number of layers that the FeedforwardModels can have.
        :param n_calls: The number of times to call the optimizer.
        :param search_space: The search space for the optimizer.
    """

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

    def save(self, filename):
        # save the MetaModel and FeedforwardModel state dicts to a file
        state = {
            'meta_model_state_dict': self.meta_model.state_dict(),
            'feedforward_model_state_dict': self.feedforward_model.state_dict()
        }
        torch.save(state, filename)

    def forward(self):
        # TODO: Implement this method
        pass

    def backward(self):
        # TODO: Implement this method
        pass

    def parameters(sel):
        # TODO: Implement this method
        pass

    def zero_grad(self):
        # TODO: Implement this method
        pass


    def loss(self):
        # TODO: Implement this method
        pass

    def optim(self):
        # TODO: Implement this method
        pass

    def training_step(self):
        # TODO: Implement this method
        pass

    def eval(self):
        # TODO: Implement this method
        pass

    def to(self):
        # TODO: Implement this method
        pass

    def __create_layer_space(self, num_layers, activation_functions=None, layer_types=None):

        if activation_functions == None:
            self.activation_functions = ['relu', 'sigmoid', 'tanh']

        if layer_types == None:
            self.max_layers = ['Dense', 'Dropout']

        space = []
        for i in range(num_layers):
            space += [
                Categorical(layer_types, name=f'layer_type_{i + 1}'),
                Integer(16, 512, name=f'layer_size_{i + 1}'),
                Categorical(activation_functions + [None], name=f'activation_function_{i + 1}')
            ]
        space += [Real(0, 1, name='dropout_probability')]
        return space
