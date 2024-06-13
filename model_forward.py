import torch


class forward_test:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": ("TENSOR",),
                'model': ('MODELS_CLASS',),
            },
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ('tensor',)
    FUNCTION = "forward_i_test"
    OUTPUT_NODE = True
    CATEGORY = "Build and train your network"

    def forward_i_test(self, tensor, model):
        out = model(tensor)
        return (out,)

class show_shape:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": ("TENSOR",),
            },
        }
    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ('tensor Dimensions',)
    FUNCTION = "show_i_shape"
    OUTPUT_NODE = False
    CATEGORY = "Build and train your network"

    def show_i_shape(self, tensor):
        return (tensor.shape,)
