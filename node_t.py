from .layer import *
from .create_any import *
from .model_forward import *
from .unit import *

NODE_CLASS_MAPPINGS = {
    "linear_layer": linear_layer,
    'activation_function': activation_function,
    'Conv_layer': Conv_layer,
    'create_model': create_model,
    'create_intput': create_intput,
    "forward_test": forward_test,
    'create_dataset': create_dataset,
    'create_training_task': create_training_task,
    'show_dimensions': show_dimensions,
    'view_layer': view_layer,
    'res_connect': res_connect,
    'Normalization_layer': Normalization_layer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "linear_layer": "Fully connected layer",
    "Conv_layer": "Conv layer",
    "activation_function": "Activation Function",
    "create_model": 'create model',
    'create_intput': 'create intput',
    "forward_test": 'forward_test',
    'create_dataset': 'create dataset',
    'create_training_task': 'training',
    'show_dimensions': 'show dimensions',
    'view_layer': 'view layer',
    'res_connect': 'res connect',
    'Normalization_layer': 'Normalization layer'
}
