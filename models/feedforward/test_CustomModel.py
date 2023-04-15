import torch
import torch.nn as nn
import unittest

from CustomModel import CustomModel


class TestCustomModel(unittest.TestCase):
    def setUp(self):
        self.input_shape = 10
        self.output_shape = 2
        self.layers_config = [('Dense', 32, 'relu'), ('Dropout', 0.3), ('Dense', 16, 'sigmoid')]
        self.dropout_probability = 0.2

        self.model = CustomModel(self.input_shape, self.output_shape, self.layers_config, self.dropout_probability)

    def test_forward(self):
        batch_size = 5
        input_tensor = torch.randn(batch_size, self.input_shape)
        output_tensor = self.model(input_tensor)
        self.assertEqual(output_tensor.shape, (batch_size, self.output_shape))

    def test_default_layers_config(self):
        default_model = CustomModel(self.input_shape, self.output_shape, dropout_probability=self.dropout_probability)

        # Check that the default model has the expected layers
        expected_layers = [
            nn.Linear(self.input_shape, 64), nn.ReLU(),
            nn.Dropout(self.dropout_probability),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, self.output_shape)
        ]
        self.assertListEqual([str(l) for l in default_model.model], [str(l) for l in expected_layers])


    def test_dropout(self):
        # Check that dropout is applied during training but not during evaluation
        input_tensor = torch.randn(1, self.input_shape)
        self.model.train()
        output_tensor_train = self.model(input_tensor)
        self.assertGreater(torch.abs(output_tensor_train).mean(), 0)

        self.model.eval()
        with torch.no_grad():
            output_tensor_eval = self.model(input_tensor)
        self.assertEqual(output_tensor_train, output_tensor_eval)
