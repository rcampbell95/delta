"""
Construct sequential neural networks using the Tensorflow-Keras API from
dictionaries.  Assumes that the names of the parameters for the layer constructor functions are as
given in the Tensorflow API documentation.
"""
import functools
from typing import Callable

import tensorflow
import tensorflow.keras.models
import tensorflow.keras.layers

from delta.config import config

def _layer_func(layer_type):
    """
    Gets the class object from the keras layers for the specified layer type.
    """
    return getattr(tensorflow.keras.layers, layer_type)

def _make_layer(layer_dict, param_dict):
    """
    Constructs a layer specified in layer_dict, possibly using parameters specified in
    param_dict.  Assumes layer_dict only contains the properly named parameters for constructing
    the layer, and the additional field 'type', that specifies the type of keras layer.
    """
    if len(layer_dict.keys()) > 1:
        raise ValueError('Layer with multiple types.')
    layer_type = next(layer_dict.keys().__iter__())
    l = layer_dict[layer_type]
    if l is None:
        l = {}

    for (k, v) in l.items():
        if isinstance(v, str) and v in param_dict.keys():
            l[k] = param_dict[v]

    return _layer_func(layer_type)(**l)

def _make_model(model_dict, exposed_params):
    layer_list = model_dict['layers']
    defined_params = {}
    if 'params' in model_dict and model_dict['params'] is not None:
        defined_params = model_dict['params']

    params = {**exposed_params, **defined_params}
    layer_objs = [_make_layer(l, params) for l in layer_list]
    return tensorflow.keras.models.Sequential(layer_objs)

def model_from_dict(model_dict, exposed_params) -> Callable[[], tensorflow.keras.models.Sequential]:
    """
    Creates a function that returns a sequential model from a dictionary.
    """
    return functools.partial(_make_model, model_dict, exposed_params)

def config_model(num_bands: int) -> Callable[[], tensorflow.keras.models.Sequential]:
    """
    Creates the model specified in the configuration.
    """
    in_data_shape = (config.chunk_size(), config.chunk_size(), num_bands)
    out_data_shape = (config.output_size(), config.output_size(), config.classes())

    params_exposed = {'out_shape' : out_data_shape,
                      'out_dims' : out_data_shape[0] * out_data_shape[1] * out_data_shape[2],
                      'in_shape' : in_data_shape,
                      'in_dims'  : in_data_shape[0] * in_data_shape[1] * in_data_shape[2],
                      'num_bands' : in_data_shape[2]}

    return model_from_dict(config.model_dict(), params_exposed)
