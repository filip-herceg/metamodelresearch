import os

import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.core import LightningModule
from ray.tune import Trainable, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from skopt.space import Categorical, Integer, Real
from torch.optim import Adam

space = [
    Categorical(['Dense', 'Dropout', None], name='layer_type_1'),
    Integer(16, 512, name='layer_size_1'),
    Categorical(['relu', 'sigmoid', 'tanh', None], name='activation_function_1'),

    Categorical(['Dense', 'Dropout', None], name='layer_type_2'),
    Integer(16, 512, name='layer_size_2'),
    Categorical(['relu', 'sigmoid', 'tanh', None], name='activation_function_2'),

    Categorical(['Dense', 'Dropout', None], name='layer_type_3'),
    Integer(16, 512, name='layer_size_3'),
    Categorical(['relu', 'sigmoid', 'tanh', None], name='activation_function_3'),

    Categorical(['Dense', 'Dropout', None], name='layer_type_4'),
    Integer(16, 512, name='layer_size_4'),
    Categorical(['relu', 'sigmoid', 'tanh', None], name='activation_function_4'),

    Categorical(['Dense', 'Dropout', None], name='layer_type_5'),
    Integer(16, 512, name='layer_size_5'),
    Categorical(['relu', 'sigmoid', 'tanh', None], name='activation_function_5'),

    Real(0, 1, name='dropout_probability'),
]


search_space = {
    'layer_type_1': space[0],
    'layer_size_1': space[1],
    'activation_function_1': space[2],

    'layer_type_2': space[3],
    'layer_size_2': space[4],
    'activation_function_2': space[5],

    'layer_type_3': space[6],
    'layer_size_3': space[7],
    'activation_function_3': space[8],

    'layer_type_4': space[9],
    'layer_size_4': space[10],
    'activation_function_4': space[11],

    'layer_type_5': space[12],
    'layer_size_5': space[13],
    'activation_function_5': space[14],

    'dropout_probability': space[15],
}


class FeedforwardModel(nn.Module):
    def __init__(self, layer_type_1, layer_size_1, activation_function_1,
                 layer_type_2, layer_size_2, activation_function_2,
                 layer_type_3, layer_size_3, activation_function_3,
                 layer_type_4, layer_size_4, activation_function_4,
                 layer_type_5, layer_size_5, activation_function_5,
                 dropout_probability):
        super().__init__()

        self.layers = nn.Sequential(
            self._get_layer(layer_type_1, layer_size_1, activation_function_1),
            self._get_layer(layer_type_2, layer_size_2, activation_function_2),
            self._get_layer(layer_type_3, layer_size_3, activation_function_3),
            self._get_layer(layer_type_4, layer_size_4, activation_function_4),
            self._get_layer(layer_type_5, layer_size_5, activation_function_5)
        )

        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, x):
        x = self.layers(x)
        x = self.dropout(x)
        return x

    def _get_layer(self, layer_type, layer_size, activation_function):
        if layer_type == 'Dense':
            layer = nn.Linear(layer_size, layer_size)
        elif layer_type == 'Dropout':
            layer = nn.Dropout(layer_size)
        else:
            raise ValueError(f'Invalid layer type: {layer_type}')

        if activation_function is not None:
            if activation_function == 'relu':
                layer = nn.Sequential(layer, nn.ReLU())
            elif activation_function == 'sigmoid':
                layer = nn.Sequential(layer, nn.Sigmoid())
            elif activation_function == 'tanh':
                layer = nn.Sequential(layer, nn.Tanh())
            else:
                raise ValueError(f'Invalid activation function: {activation_function}')

        return layer


class MetaModel(Trainable, LightningModule):
    def __init__(self):
        super().__init__()
        self.config = {
            'layer_type_1': 'Dense',
            'layer_size_1': 16,
            'activation_function_1': 'relu',

            'layer_type_2': 'Dense',
            'layer_size_2': 16,
            'activation_function_2': 'relu',

            'layer_type_3': 'Dense',
            'layer_size_3': 16,
            'activation_function_3': 'relu',

            'layer_type_4': 'Dense',
            'layer_size_4': 16,
            'activation_function_4': 'relu',

            'layer_type_5': 'Dense',
            'layer_size_5': 16,
            'activation_function_5': 'relu',

            'dropout_probability': 0.0
        }

    def setup(self, config):
        self.config.update(config)

        for key in ['layer_type_1', 'layer_size_1', 'activation_function_1',
                    'layer_type_2', 'layer_size_2', 'activation_function_2',
                    'layer_type_3', 'layer_size_3', 'activation_function_3',
                    'layer_type_4', 'layer_size_4', 'activation_function_4',
                    'layer_type_5', 'layer_size_5', 'activation_function_5',
                    'dropout_probability']:
            if key not in self.config:
                self.config[key] = None

        self.feedforward_model = FeedforwardModel(
            self.config['layer_type_1'], self.config['layer_size_1'], self.config['activation_function_1'],
            self.config['layer_type_2'], self.config['layer_size_2'], self.config['activation_function_2'],
            self.config['layer_type_3'], self.config['layer_size_3'], self.config['activation_function_3'],
            self.config['layer_type_4'], self.config['layer_size_4'], self.config['activation_function_4'],
            self.config['layer_type_5'], self.config['layer_size_5'], self.config['activation_function_5'],
            self.config['dropout_probability']
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.feedforward_model.parameters())

    def forward(self, x):
        return self.feedforward_model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return self.optimizer

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, 'model.pth')
        torch.save(self.feedforward_model.state_dict(), checkpoint_path)

    def load_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, 'model.pth')
        self.feedforward_model.load_state_dict(torch.load(checkpoint_path))

    def _run(self, epoch_amount, train_loader, val_loader, test_loader, search_alg, scheduler):
        analysis = tune.run(
            self,
            resources_per_trial={
                'cpu': 1},
            config={
                'layer_type_1': Categorical(['Dense', 'Dropout']),
                'layer_size_1': Integer(16, 512),
                'activation_function_1': Categorical(['relu', 'sigmoid', 'tanh', None]),

                'layer_type_2': Categorical(['Dense', 'Dropout']),
                'layer_size_2': Integer(16, 512),
                'activation_function_2': Categorical(['relu', 'sigmoid', 'tanh', None]),

                'layer_type_3': Categorical(['Dense', 'Dropout']),
                'layer_size_3': Integer(16, 512),
                'activation_function_3': Categorical(['relu', 'sigmoid', 'tanh', None]),

                'layer_type_4': Categorical(['Dense', 'Dropout']),
                'layer_size_4': Integer(16, 512),
                'activation_function_4': Categorical(['relu', 'sigmoid', 'tanh', None]),

                'layer_type_5': Categorical(['Dense', 'Dropout']),
                'layer_size_5': Integer(16, 512),
                'activation_function_5': Categorical(['relu', 'sigmoid', 'tanh', None]),

                'dropout_probability': Real(0, 1),
            },
            num_samples=10,
            scheduler=scheduler,
            search_alg=search_alg,
            stop={'training_iteration': epoch_amount},
            verbose=1,
        )

        best_trial = analysis.get_best_trial('val_loss', 'min', 'last')
        best_config = best_trial.config
        best_loss = best_trial.last_result['val_loss']

        print(f'Best config: {best_config}')
        print(f'Best validation loss: {best_loss:.4f}')

        self.setup(best_config)

        # Create an instance of Trainer
        trainer = Trainer(max_epochs=epoch_amount)

        # Call the fit method to train the model with the best configuration
        trainer.fit(self, train_loader, val_loader)

        # Evaluate the model on the test set
        result = trainer.test(self, test_loader)[0]
        test_loss = result['test_loss']
        print(f'Test loss: {test_loss:.4f}')
        return test_loss


def main(epoch_amount, train_loader, val_loader, test_loader):
    meta_model = MetaModel()
    search_alg = HyperOptSearch(search_space, metric='val_loss', mode='min')
    scheduler = ASHAScheduler(max_t=epoch_amount, grace_period=1)
    for i in range(10):
        test_loss = meta_model._run(epoch_amount=epoch_amount, train_loader=train_loader,
                                    val_loader=val_loader, test_loader=test_loader,
                                    search_alg=search_alg, scheduler=scheduler)
        print(f'Test loss for trial {i + 1}: {test_loss:.4f}')
