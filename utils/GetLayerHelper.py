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

        # Check if the layer class requires any additional arguments beyond input_shape and layer_units
        signature = inspect.signature(layer_class.__init__)
        if len(signature.parameters) > 2:
            # The layer class has additional arguments beyond input_shape and layer_units
            # Get the names and default values of the required arguments
            required_args = list(signature.parameters.keys())[2:]
            default_values = list(signature.parameters.values())[2:]
            default_kwargs = {k: v.default for k, v in zip(required_args, default_values) if v.default is not inspect.Parameter.empty}

            # Make sure all required arguments are present in kwargs
            missing_args = set(required_args) - set(kwargs.keys())
            if missing_args:
                raise ValueError(f"Layer type '{name}' is missing required arguments: {missing_args}")

            # Use the default values for any missing optional arguments
            for arg, default_value in default_kwargs.items():
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

