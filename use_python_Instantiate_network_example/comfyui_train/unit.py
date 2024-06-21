class res_connect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "res_1": ("RES",),
                "res_2": ("RES",),
            },
            "optional": {
                "res_1": ("RES",),
                "res_2": ("RES",),
            }
        }

    RETURN_TYPES = ("RES",)
    RETURN_NAMES = ('res',)
    FUNCTION = "connect"
    OUTPUT_NODE = True
    CATEGORY = "Build and train your network/unit"

    def connect(self, res_1, res_2):
        if isinstance(res_1, tuple):
            res = res_1 + (res_2,)
            return (res,)
        elif isinstance(res_2, tuple):
            res = res_2 + (res_1,)
            return (res,)
        else:
            return ((res_1, res_2),)
