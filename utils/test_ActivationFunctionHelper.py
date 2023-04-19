import unittest
from ActivationFunctionHelper import get_activation, get_available_activation_types
import torch.nn as nn

class TestActivationFunctionHelper(unittest.TestCase):

    def test_get_activation(self):
        # Test valid activation type (ReLU)
        layer = get_activation('ReLU')
        self.assertIsInstance(layer, nn.ReLU)

        # Test valid activation type (Sigmoid)
        layer = get_activation('Sigmoid')
        self.assertIsInstance(layer, nn.Sigmoid)

        # Test valid activation type (Tanh)
        layer = get_activation('Tanh')
        self.assertIsInstance(layer, nn.Tanh)

        # Test invalid activation type (neural network layer)
        with self.assertRaises(ValueError):
            layer = get_activation('Linear')

        # Test invalid activation type
        with self.assertRaises(ValueError):
            layer = get_activation('foo')


    def test_get_available_layer_types(self):
        layers = get_available_activation_types()
        print(layers)
        self.assertIn('ReLU', layers)
        self.assertIn('Sigmoid', layers)
        self.assertIn('Tanh', layers)

