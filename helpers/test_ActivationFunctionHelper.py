from unittest import TestCase

from torch import nn

from helpers.ActivationFunctionHelper import get_activation


class Test(TestCase):
    def test_get_activation(self):
        # Test supported activation functions
        self.assertIsInstance(get_activation('tanh'), nn.Tanh)
        self.assertIsInstance(get_activation('relu'), nn.ReLU)
        self.assertIsInstance(get_activation('sigmoid'), nn.Sigmoid)
        self.assertIsInstance(get_activation('celu'), nn.CELU)
        self.assertIsInstance(get_activation('softmax'), nn.Softmax)
        self.assertIsInstance(get_activation('leakyrelu'), nn.LeakyReLU)
        self.assertIsInstance(get_activation('prelu'), nn.PReLU)
        self.assertIsInstance(get_activation('elu'), nn.ELU)
        self.assertIsInstance(get_activation('gelu'), nn.GELU)
        self.assertIsInstance(get_activation('swish'), nn.SiLU)
        self.assertIsInstance(get_activation('relu6'), nn.ReLU6)
        self.assertIsInstance(get_activation('hardshrink'), nn.Hardshrink)
        self.assertIsInstance(get_activation('hardtanh'), nn.Hardtanh)
        self.assertIsInstance(get_activation('softplus'), nn.Softplus)
        self.assertIsInstance(get_activation('logsigmoid'), nn.LogSigmoid)
        self.assertIsInstance(get_activation('tanhshrink'), nn.Tanhshrink)
        self.assertIsInstance(get_activation('softshrink'), nn.Softshrink)
        self.assertIsInstance(get_activation('softsign'), nn.Softsign)
        self.assertIsInstance(get_activation('rrelu'), nn.RReLU)
        self.assertIsInstance(get_activation('hardswish'), nn.Hardswish)
        self.assertIsInstance(get_activation('logsoftmax'), nn.LogSoftmax)
        self.assertIsInstance(get_activation('softmin'), nn.Softmin)

        # Test unsupported activation function
        with self.assertRaises(ValueError):
            get_activation('nonesense_gibberish')
