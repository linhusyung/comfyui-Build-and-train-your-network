import torch
import pickle
from comfyui_train.create_any import net

if __name__ == '__main__':
    with open('D:/k.pkl', 'rb') as f:
        k_loaded = pickle.load(f)

    print(k_loaded)
    net = net(k_loaded)
    net.res_cpu()
    net.load_state_dict(torch.load(
        'D:/ComfyUI_windows_portable_nvidia_cu121_or_cpu/ComfyUI_windows_portable/lightning_logs/version_1/checkpoints/epoch=29-step=5520.ckpt')[
                            'state_dict'])
    x = torch.rand([1, 3, 117, 86])
    print(net(x)[0])
