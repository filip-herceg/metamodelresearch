import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.optim import Adam
import optuna
import pytorch_lightning as pl

from utils.ActivationFunctionHelper import get_activation, get_available_activation_types
from utils.GetLayerHelper import get_layer, get_available_layer_types


class MetaModel(pl.LightningModule):
    """
    MetaModel is a class for creating a metamodel that generates and optimizes
    neural network architectures. The metamodel learns to optimize the generated
    model's architecture by testing new arrangements of layer types, activation
    function types, layer shapes, and number of layers.
    """

    def __init__(self, input_shape, output_shape, train_loader, val_loader, max_layers=5, n_calls=20,
                 search_space=None, verbosity=1):
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
        self.verbosity = verbosity

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

    def create_layer_space(self, num_layers, layer_types=None, activation_functions=None):
        """
        Create the search space for layers and activation functions.

        Args:
            num_layers (int): The number of layers in the search space.
            activation_functions (list, optional): A list of available activation functions. Defaults to None.

        Returns:
            list: A list of search space dimensions.
        """
        if activation_functions is None:
            activation_functions = get_available_activation_types()

        if layer_types is None:
            layer_types = get_available_layer_types()

        space = {
            'layer_types': layer_types,
            'activation_functions': activation_functions
        }

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

            layer = get_layer(layer_type, self.input_shape, layer_units)  # use input_shape as the first argument
            activation = get_activation(activation_name)  # remove input argument

            layers.append(layer)
            layers.append(activation)

        self.model = nn.Sequential(*layers)
        return self.model

    def configure_optimizers(self):
        """
        Configure the optimizer for the MetaModel.

        Returns:
            torch.optim.Optimizer: The configured optimizer.
        """
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        """
        Perform a training step for the MetaModel.

        Args:
            batch (tuple): A tuple containing input data and corresponding targets.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The loss for the current training step.
        """
        inputs, targets = batch
        model = self.model
        outputs = model(inputs)
        loss = self.loss(outputs, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a validation step for the MetaModel.

        Args:
            batch (tuple): A tuple containing input data and corresponding targets.
            batch_idx (int): The index of the current batch.

        Returns:
            None
        """
        inputs, targets = batch
        model = self.model
        outputs = model(inputs)
        loss = self.loss(outputs, targets)
        self.log('val_loss', loss)

    def objective(self, trial):
        params = []
        for i in range(self.max_layers):
            layer_type = trial.suggest_categorical(f"layer_type_{i}", self.search_space['layer_types'])
            layer_units = trial.suggest_int(f"layer_units_{i}", 1, 512)
            activation_name = trial.suggest_categorical(f"activation_{i}", self.search_space['activation_functions'])

            params.extend([layer_type, layer_units, activation_name])

        model = self.generate_model(params)

        if self.verbosity >= 1:
            print(f"Generated model: {model}")

        # Train and validate the model
        trainer = pl.Trainer(max_epochs=10, progress_bar_refresh_rate=(100 if self.verbosity >= 2 else 0))
        trainer.fit(self, self.train_loader, self.val_loader)

        val_loss = trainer.callback_metrics['val_loss'].item()  # use callback_metrics instead of logged_metrics
        if self.verbosity >= 1:
            print(f"Validation loss: {val_loss}")

        return val_loss

    def optimize(self):
        """
        Optimize the model's architecture using Optuna.

        Returns:
            list: The best parameters found during optimization.
        """
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.n_calls)

        best_params = study.best_params.values()

        return list(best_params)


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

    def test_and_plot(self, test_loader):
        best_params = self.optimize()
        best_model = self.generate_model(best_params)

        # Evaluate the best model on test data
        test_inputs, test_targets = next(iter(test_loader))
        test_outputs = best_model(test_inputs).detach().numpy()

        # Plot test data
        plt.scatter(test_inputs.numpy(), test_targets.numpy(), label='True')
        plt.scatter(test_inputs.numpy(), test_outputs, label='Predicted')
        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.legend()
        plt.show()
