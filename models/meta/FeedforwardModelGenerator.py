import torch
import torch.nn as nn
import torch.optim as optim
from skopt import Optimizer
from skopt.space import Categorical, Integer, Real
from skopt.utils import use_named_args
import numpy as np
import matplotlib.pyplot as plt


# Define the search space for Bayesian optimization
space = [
    Categorical(['Dense', 'Dropout'], name='layer_type_1'),
    Integer(16, 512, name='layer_size_1'),
    Categorical(['relu', 'sigmoid', 'tanh', None], name='activation_function_1'),

    Categorical(['Dense', 'Dropout'], name='layer_type_2'),
    Integer(16, 512, name='layer_size_2'),
    Categorical(['relu', 'sigmoid', 'tanh', None], name='activation_function_2'),

    Categorical(['Dense', 'Dropout'], name='layer_type_3'),
    Integer(16, 512, name='layer_size_3'),
    Categorical(['relu', 'sigmoid', 'tanh', None], name='activation_function_3'),

    Categorical(['Dense', 'Dropout'], name='layer_type_4'),
    Integer(16, 512, name='layer_size_4'),
    Categorical(['relu', 'sigmoid', 'tanh', None], name='activation_function_4'),


    Categorical(['Dense', 'Dropout'], name='layer_type_5'),
    Integer(16, 512, name='layer_size_5'),
    Categorical(['relu', 'sigmoid', 'tanh', None], name='activation_function_5'),

    Real(0, 1, name='dropout_probability'),  # Add a new variable for dropout probability
]


class FeedForwardModelGenerator:
    def __init__(self, input_shape, output_shape, train_loader, val_loader, max_layers=5, n_calls=20,
                 search_space=space):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_layers = max_layers
        self.n_calls = n_calls
        self.space = search_space

    def _create_model(self, layers_config, dropout_probability):
        class CustomModel(nn.Module):
            def __init__(self, input_shape, output_shape):
                super(CustomModel, self).__init__()
                layers = []
                prev_layer_size = input_shape

                for layer_type, layer_size, activation_function in layers_config:
                    if layer_type == 'Dense':
                        layers.append(nn.Linear(prev_layer_size, layer_size))
                        prev_layer_size = layer_size

                    elif layer_type == 'Dropout':
                        layers.append(nn.Dropout(dropout_probability))  # Use the dropout_probability variable

                layers.append(nn.Linear(prev_layer_size, output_shape))
                self.model = nn.Sequential(*layers)


            def forward(self, x):
                return self.model(x)

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

        return CustomModel(self.input_shape, self.output_shape)



    def _train_and_evaluate(self, model, epochs=10):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for i, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            print(f"Epoch: {epoch+1}, Loss: {running_loss / (i+1)}")

        # Evaluate the model on the validation set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                # Display images from the validation dataset
                inputs = inputs.cpu().numpy()
                for idx in range(inputs.shape[0]):
                    plt.imshow(inputs[idx, 0], cmap='gray')
                    plt.title(f"True Label: {targets[idx]}, Predicted Label: {predicted[idx]}")
                    plt.show()

        val_accuracy = correct / total
        return val_accuracy

    def _objective(self, x):
        layers_config = []
        n_layers = len(x) // 3

        for i in range(n_layers):
            layer_config = (x[3 * i], x[3 * i + 1], x[3 * i + 2])
            layers_config.append(layer_config)

        dropout_probability = x[-1]  # Get the dropout_probability from the search space
        model = self._create_model(layers_config, dropout_probability)  # Pass dropout_probability to _create_model
        val_accuracy = self._train_and_evaluate(model)

        # Since scikit-optimize minimizes the objective, we return the negative accuracy
        return -val_accuracy

    def generate_optimal_model(self):
        optimizer = Optimizer(self.space)
        for _ in range(self.n_calls):
            x = optimizer.ask()
            f = self._objective(x)  # Call the modified _objective function
            optimizer.tell(x, f)

        best_params = optimizer.Xi[np.argmin(optimizer.yi)]
        best_layers_config = []
        n_layers = len(best_params) // 3

        for i in range(n_layers):
            layer_config = (best_params[3 * i], best_params[3 * i + 1], best_params[3 * i + 2])
            best_layers_config.append(layer_config)

        best_model = self._create_model(best_layers_config)
        return best_model

