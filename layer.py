import torch
from torch import nn
import copy
import time
import os


class linear_layer():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "in_features": ("INT", {"default": 10., "min": 1, "max": 0xffffffffffffffff, "step": 1, }),
                "out_features": ("INT", {"default": 10., "min": 1, "max": 0xffffffffffffffff, "step": 1, }),
            },
            "optional": {
                "layer": ("LAYER",),
                'res': ("RES",),
                "res_type": (
                    ['add', 'concat-dim=0', 'concat-dim=1', 'concat-dim=2', 'concat-dim=3', 'Mul', 'div', 'sub'],),
            }
        }

    RETURN_TYPES = ("LAYER", 'RES',)
    RETURN_NAMES = ('LAYER', 'res',)
    FUNCTION = "init_layer"
    OUTPUT_NODE = True
    CATEGORY = "Build and train your network"

    def init_layer(self, layer=None, in_features=10, out_features=10, res=None, res_type=None):
        if layer is None and res is None:
            layer_ = nn.ModuleList()
            layer_.append(nn.Linear(in_features=in_features, out_features=out_features))
            rt_res = []
            layer = [layer_, rt_res]
            return (layer, [len(layer[0]) - 1],)

        if layer is not None and res is None:
            layer_a = copy.deepcopy(layer)
            del layer
            layer_a[0].append(nn.Linear(in_features=in_features, out_features=out_features))
            layer = copy.deepcopy(layer_a)
            del layer_a
            return (layer, [len(layer[0]) - 1],)

        if res is not None and layer is not None:
            layer_a = copy.deepcopy(layer)
            del layer
            layer_a[0].append(nn.Linear(in_features=in_features, out_features=out_features))

            if len(res) <= 1:
                layer_a[1].append((res[0], len(layer_a[0]) - 1, res_type))
            else:
                if isinstance(res, tuple):
                    for i in res:
                        layer_a[1].append((i[0], len(layer_a[0]) - 1, i[2], i[1]))
                else:
                    layer_a[1].append((res[0], len(layer_a[0]) - 1, res_type, res[1]))
            layer = copy.deepcopy(layer_a)
            del layer_a
            return (layer, [(len(layer[0]) - 1), ],)

        if layer is None and res is not None:
            if len(res) <= 1:
                res_layer = nn.ModuleList()
                res_layer.append(nn.Linear(in_features=in_features, out_features=out_features))
                res_a = copy.deepcopy(res)
                del res
                res_a.append(res_layer)
                res_a.append(res_type)
                res = copy.deepcopy(res_a)
                del res_a
            else:
                res_a = copy.deepcopy(res)
                del res
                res_a[1].append(nn.Linear(in_features=in_features, out_features=out_features))
                res_a[-1] = res_type
                res = copy.deepcopy(res_a)
                del res_a
            return (layer, res,)


class activation_function():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "act_func_type": (
                    ['ReLU', 'Sigmoid', 'Tanh',
                     'Softmax', 'LeakyReLU'],),
            },
            "optional": {
                "layer": ("LAYER",),
                'res': ("RES",),
                "res_type": (
                    ['add', 'concat-dim=0', 'concat-dim=1', 'concat-dim=2', 'concat-dim=3', 'Mul', 'div', 'sub'],),
            }
        }

    RETURN_TYPES = ("LAYER", 'RES',)
    RETURN_NAMES = ('LAYER', 'res',)
    FUNCTION = "init_layer"
    OUTPUT_NODE = True
    CATEGORY = "Build and train your network"

    def __init__(self):
        self.Chart = {
            'ReLU': nn.ReLU(),
            'Sigmoid': nn.Sigmoid(),
            'Tanh': nn.Tanh(),
            'Softmax': nn.Softmax(dim=-1),
            'LeakyReLU': nn.LeakyReLU(),
        }

    def init_layer(self, act_func_type, layer=None, res=None, res_type=None):
        if layer is None and res is None:
            layer_ = nn.ModuleList()
            layer_.append(self.Chart[act_func_type])
            rt_res = []
            layer = [layer_, rt_res]
            return (layer, [len(layer[0]) - 1],)

        if layer is not None and res is None:
            layer_a = copy.deepcopy(layer)
            del layer
            layer_a[0].append(self.Chart[act_func_type])
            layer = copy.deepcopy(layer_a)
            del layer_a
            return (layer, [len(layer[0]) - 1],)

        if res is not None and layer is not None:
            layer_a = copy.deepcopy(layer)
            del layer
            layer_a[0].append(self.Chart[act_func_type])

            if len(res) <= 1:
                layer_a[1].append((res[0], len(layer_a[0]) - 1, res_type))
            else:
                if isinstance(res, tuple):
                    for i in res:
                        layer_a[1].append((i[0], len(layer_a[0]) - 1, i[2], i[1]))
                else:
                    layer_a[1].append((res[0], len(layer_a[0]) - 1, res_type, res[1]))
            layer = copy.deepcopy(layer_a)
            del layer_a
            return (layer, [(len(layer[0]) - 1), ],)

        if layer is None and res is not None:
            if len(res) <= 1:
                res_layer = nn.ModuleList()
                res_layer.append(self.Chart[act_func_type])
                res_a = copy.deepcopy(res)
                del res
                res_a.append(res_layer)
                res = copy.deepcopy(res_a)
                del res_a
            else:
                res_a = copy.deepcopy(res)
                del res
                res_a[1].append(self.Chart[act_func_type])
                res_a[-1] = res_type
                res = copy.deepcopy(res_a)
                del res_a
            return (layer, res,)


class view(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, tensor):
        return tensor.view(tensor.size(0), self.d)


class view_layer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "flatten_dimension": (
                    "INT", {"default": -1, "min": -99, "max": 0xffffffffffffffff, "step": 1, }),
            },
            "optional": {
                "layer": ("LAYER",),
            }
        }

    RETURN_TYPES = ("LAYER",)
    RETURN_NAMES = ('LAYER',)
    FUNCTION = "init_layer"
    OUTPUT_NODE = True
    CATEGORY = "Build and train your network"

    def init_layer(self, layer=None, flatten_dimension=-1):
        if isinstance(layer, nn.ModuleList):
            layer_a = copy.deepcopy(layer)
            del layer
            layer_a.append(view(flatten_dimension))
            layer = copy.deepcopy(layer_a)
        else:
            layer = nn.ModuleList()
            layer.append(view(flatten_dimension))
        return (layer,)


class Conv_layer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "in_channels": ("INT", {"default": 3, "min": 1, "max": 0xffffffffffffffff, "step": 1, }),
                "out_channels": ("INT", {"default": 32, "min": 1, "max": 0xffffffffffffffff, "step": 1, }),
                "kernel_size": ("INT", {"default": 3, "min": 1, "max": 0xffffffffffffffff, "step": 1, }),
                "padding": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1, }),
                "stride": ("INT", {"default": 1, "min": 1, "max": 0xffffffffffffffff, "step": 1, }),
            },
            "optional": {
                "layer": ("LAYER",),
            }
        }

    RETURN_TYPES = ("LAYER",)
    RETURN_NAMES = ('LAYER',)
    FUNCTION = "init_Conv_layer"
    OUTPUT_NODE = True
    CATEGORY = "Build and train your network"

    def init_Conv_layer(self, layer=None, in_channels=10, out_channels=10, kernel_size=3, padding=0, stride=1):
        if isinstance(layer, nn.ModuleList):
            layer_a = copy.deepcopy(layer)
            del layer
            layer_a.append(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding,
                          stride=stride))
            layer = copy.deepcopy(layer_a)
        else:
            layer = nn.ModuleList()
            layer.append(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding,
                          stride=stride))
        return (layer,)
