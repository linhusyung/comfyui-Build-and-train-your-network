import torch


class forward_test:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": ("TENSOR",),
                'model': ('MODELS_CLASS',),
                "Flatten": ("BOOLEAN", {"default": False}),
                'return_shape': ("BOOLEAN", {"default": False}),

            },
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ('tensor',)
    FUNCTION = "forward_i_test"
    OUTPUT_NODE = True
    CATEGORY = "Build and train your network"

    def forward_i_test(self, tensor, model, Flatten=False, return_shape=False):
        out = model(tensor)
        if Flatten:
            if return_shape:
                return (out.view(out.size(0), -1).shape,)
            return (out.view(out.size(0), -1),)
        if return_shape:
            return (out.shape,)
        return (out,)

