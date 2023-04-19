import unittest
from GetLayerHelper import get_layer, get_available_layer_types
import torch.nn as nn

class TestGetLayerHelper(unittest.TestCase):

    def test_get_layer(self):
        # Test valid layer type (linear)
        layer = get_layer('Linear', 5, 10)
        self.assertIsInstance(layer, nn.Linear)

        # Test valid layer type (conv2d)
        layer = get_layer('Conv2d', 3, 16, kernel_size=3, padding=1)
        self.assertIsInstance(layer, nn.Conv2d)

        # Test invalid layer type (activation function)
        with self.assertRaises(ValueError):
            layer = get_layer('ReLU')

        # Test invalid layer type
        with self.assertRaises(ValueError):
            layer = get_layer('foo')


    def test_get_available_layer_types(self):
        layers = get_available_layer_types()
        print(layers)
        self.assertIn('Linear', layers)
        self.assertIn('Conv2d', layers)

