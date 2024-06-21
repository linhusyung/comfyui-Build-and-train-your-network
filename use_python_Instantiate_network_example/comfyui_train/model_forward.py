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
                'load_ckpt': ("BOOLEAN", {"default": False}),
                "device": (
                    ['cpu', 'cuda'],),


            },
            "optional": {
                "ckpt_path": ("STRING", {
                    "default": "D:/ComfyUI_windows_portable_nvidia_cu121_or_cpu/ComfyUI_windows_portable/lightning_logs/version_1/checkpoints/epoch=29-step=5520.ckpt"}),
            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR",)
    RETURN_NAMES = ('tensor', 'layer tensor',)
    FUNCTION = "forward_i_test"
    OUTPUT_NODE = True
    CATEGORY = "Build and train your network"

    def forward_i_test(self, tensor, model, device='cuda', load_ckpt=False, ckpt_path='', Flatten=False,
                       return_shape=False):
        if device == 'cuda':
            model.to(device)
            tensor = tensor.to(device)

        if load_ckpt and device == 'cuda':
            model.load_state_dict(torch.load(ckpt_path, map_location=device)['state_dict'])
        else:
            model.res_cpu()
            model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['state_dict'])

        with torch.inference_mode():
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
