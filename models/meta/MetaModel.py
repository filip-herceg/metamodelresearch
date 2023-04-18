import torch
import torch.nn as nn
from torch.optim import Adam
from skopt import Optimizer
from skopt.space import Categorical, Integer
import pytorch_lightning as pl

from utils.ActivationFunctionHelper import get_activation
from utils.GetLayerHelper import get_layer

class MetaModel(pl.LightningModule):
    """
    MetaModel is a class for creating a metamodel that generates and optimizes
    neural network architectures. The metamodel learns to optimize the generated
    model's architecture by testing new arrangements of layer types, activation
    function types, layer shapes, and number of layers.
    """

    def __init__(self, input_shape, output_shape, train_loader, val_loader, max_layers=5, n_calls=20,
                 search_space=None):
        """
        Initialize the MetaModel class.

        Args:
            input_shape (int): The shape of the input data.
            output_shape (int): The shape of the output data.
            train_loader (torch.utils.data.DataLoader): The training data loader.
            val_loader (torch.utils.data.DataLoader): The validation data loader.
            max_layers (int, optional): The maximum number of layers the MetaModel can generate. Defaults to 5.
            n_calls (int, optional): The number of times the MetaModel will be trained. Defaults to 20.
            search_space (list, optional): The search space for the MetaModel. Defaults to None.
        """
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_layers = max_layers
        self.n_calls = n_calls

        if search_space is None:
            self.search_space = self.create_layer_space(max_layers)
        else:
            self.search_space = search_space

    def save(self, filename):
        """
        Save the MetaModel state dict to a file.

        Args:
            filename (str): The name of the file to save the state dict.
        """
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        """
        Load the MetaModel state dict from a file.

        Args:
            filename (str): The name of the file to load the state dict.
        """
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict)

    def create_layer_space(self, num_layers, activation_functions=None, layer_types=None):
        """
        Create the search space for layers and activation functions.

        Args:
            num_layers (int): The number of layers in the search space.
            activation_functions (list, optional): A list of available activation functions. Defaults to None.
            layer_types (list, optional): A list of available layer types. Defaults to None.

        Returns:
            list: A list of search space dimensions.
        """
        if activation_functions is None:
            activation_functions = ['relu', 'sigmoid', 'tanh']
        if layer_types is None:
            layer_types = ['linear', 'conv2d', 'maxpool2d', 'avgpool2d']

        space = []
        for i in range(num_layers):
            space.append(Categorical(layer_types, name=f'layer_type_{i}'))
            space.append(Integer(1, 512, name=f'layer_units_{i}'))
            space.append(Categorical(activation_functions, name=f'activation{i}'))

        return space

    def generate_model(self, params):
        """
        Generate a model based on the provided parameters.

        Args:
            params (list): A list of parameters specifying layer types, units, and activation functions.

        Returns:
            torch.nn.Sequential: A generated neural network model.
        """
        layers = []
        for i in range(0, len(params), 3):
            layer_type = params[i]
            layer_units = params[i + 1]
            activation_name = params[i + 2]

            layer = get_layer(layer_type, layer_units)
            activation = get_activation(activation_name)

            layers.append(layer)
            layers.append(activation)

        self.model = nn.Sequential(*layers)
        return self.model

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        model = self.model
        outputs = model(inputs)
        loss = self.loss(outputs, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        model = self.model
        outputs = model(inputs)
        loss = self.loss(outputs, targets)
        self.log('val_loss', loss)

    def optimize(self):
        """
        Optimize the model's architecture.

        Returns:
            list: The best parameters found during optimization.
        """
        optimizer = Optimizer(self.search_space)
        best_loss = float('inf')
        best_params = None

        for _ in range(self.n_calls):
            params = optimizer.ask()
            model = self.generate_model(params)

            # Train and validate the model
            trainer = pl.Trainer(max_epochs=10)
            trainer.fit(self, self.train_loader, self.val_loader)

            val_loss = trainer.logged_metrics['val_loss'].item()

            if val_loss < best_loss:
                best_loss = val_loss
                best_params = params

            optimizer.tell(params, val_loss)

        return best_params

    def backward(self, loss, **kwargs):
        """
        Backward pass through the model.

        Args:
            loss (torch.Tensor): The loss tensor.

        Returns:
            None
            :param **kwargs:
        """
        loss.backward()

    def parameters(self, **kwargs):
        """
        Get the model parameters.

        Returns:
            Iterator[nn.Parameter]: Model parameters.
            :param **kwargs:
        """
        return self.model.parameters()

    def zero_grad(self, **kwargs):
        """
        Zero the gradients of all parameters.

        Returns:
            None
            :param **kwargs:
        """
        for param in self.model.parameters():
            param.grad = None

    def loss(self, outputs, targets):
        """
        Calculate the loss between outputs and targets.

        Args:
            outputs (torch.Tensor): Model outputs.
            targets (torch.Tensor): Ground truth targets.

        Returns:
            torch.Tensor: Loss tensor.
        """
        criterion = nn.MSELoss()  # Change this to the appropriate loss function for your problem
        return criterion(outputs, targets)