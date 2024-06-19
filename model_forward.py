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

    RETURN_TYPES = ("TENSOR", "TENSOR",)
    RETURN_NAMES = ('tensor', 'layer tensor',)
    FUNCTION = "forward_i_test"
    OUTPUT_NODE = True
    CATEGORY = "Build and train your network"

    def forward_i_test(self, tensor, model, Flatten=False, return_shape=False):
        out, out_list = model(tensor)
        if Flatten:
            if return_shape:
                return (out.view(out.size(0), -1).shape, out_list)
            return (out.view(out.size(0), -1), out_list)
        if return_shape:
            return (out.shape, ([x.shape for x in out_list[0]], [y.shape for x in out_list[1] for y in x]))
        return (out, out_list)


class show_dimensions:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": ("TENSOR",),
            },
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ('tensor',)
    FUNCTION = "p_shape"
    OUTPUT_NODE = True
    CATEGORY = "Build and train your network"

    def p_shape(self, tensor):
        return (tensor.shape,)
