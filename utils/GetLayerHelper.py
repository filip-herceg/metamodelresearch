import torch.nn as nn
import inspect

def get_layer(name, *args, **kwargs):
    """
    Returns a layer module given its name and arguments.

    Args:
        name (str): The name of the layer module.
        *args: Any additional positional arguments needed to construct the layer module.
        **kwargs: Any additional keyword arguments needed to construct the layer module.

    Returns:
        nn.Module: An instance of the specified layer type.

    Raises:
        ValueError: If the specified layer type is not supported.
    """
    if name in get_available_layer_types():
        layer_class = getattr(nn, name)
        return layer_class(*args, **kwargs)
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
            layer_types.append(name)
    return layer_types
