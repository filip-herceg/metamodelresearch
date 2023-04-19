import torch.nn as nn
import inspect

from torch.nn.modules import activation


def get_layer(name, input_shape, layer_units, **kwargs):
    """
    Returns a layer module given its name, input shape, number of units, and any additional arguments.

    Args:
        name (str): The name of the layer module.
        input_shape (int or tuple[int]): The shape of the input data.
        layer_units (int): The number of units in the layer.
        **kwargs: Any additional keyword arguments needed to construct the layer module.

    Returns:
        nn.Module: An instance of the specified layer type.

    Raises:
        ValueError: If the specified layer type is not supported, or if any required arguments are missing.
    """
    if name in get_available_layer_types():
        layer_class = getattr(nn, name)

        # Get the expected arguments for the layer class
        layer_args = get_layer_args(layer_class)

        # Check if all required arguments are present in kwargs
        missing_args = set(layer_args['required']) - set(kwargs.keys())
        if missing_args:
            raise ValueError(f"Layer type '{name}' is missing required arguments: {missing_args}")

        # Use the default values for any missing optional arguments
        for arg, default_value in layer_args['optional'].items():
            if arg not in kwargs:
                kwargs[arg] = default_value

        # Call the layer constructor with the specified arguments
        return layer_class(input_shape, layer_units, **kwargs)
    else:
        raise ValueError(f"Layer type '{name}' not supported.")


def get_available_layer_types():
    """
    Returns a list of the available layer types that can be constructed.

    Returns:
        list[str]: A list of strings representing the names of the available layer types.
    """
    layer_types = []
    for name, obj in inspect.getmembers(nn):
        if inspect.isclass(obj) and issubclass(obj, nn.Module) and not name.startswith('_') and not name.endswith('Base'):
            if not hasattr(activation, name):
                layer_types.append(name)
    return layer_types


def get_layer_args(layer_class):
    """
    Returns a dictionary of the expected arguments for a given layer class.

    Args:
        layer_class (nn.Module): The PyTorch layer class.

    Returns:
        dict: A dictionary with two keys: 'required' and 'optional'. The value of 'required' is a list of required
        argument names for the layer, and the value of 'optional' is a dictionary of optional argument names and
        their default values.
    """
    signature = inspect.signature(layer_class.__init__)
    required_args = []
    optional_args = {}
    for param in signature.parameters.values():
        if param.default == inspect.Parameter.empty:
            required_args.append(param.name)
        else:
            optional_args[param.name] = param.default
    return {'required': required_args, 'optional': optional_args}
