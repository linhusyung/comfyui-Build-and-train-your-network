import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pytorch_lightning import Trainer


class net(pl.LightningModule):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer[0]
        self.res_seq = layer[1]

        self.MSE = nn.MSELoss()
        self.CEB = nn.CrossEntropyLoss()
        self.BCE = nn.BCELoss()

    def forward(self, x):
        print(self.res_seq)
        outputs = [x]
        for i, layer in enumerate(self.layer):
            out = layer(outputs[-1])

            for res in self.res_seq:
                if i == res[1]:
                    if len(res) >= 4:
                        for k in res[3]:
                            res_out = k(outputs[res[0]])
                            out += res_out
                    else:
                        out += outputs[res[0]]

            outputs.append(out)
        return outputs[-1]

    def training_step(self, batch, batch_idx):
        data, label = batch
        out = self(data)
        loss = self.CEB(out, label)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        out = self(data)
        loss = self.CEB(out, label)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer


class create_intput:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor_Dimensions": ("STRING", {"default": "[10,10]"})
            },
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ('tensor',)
    FUNCTION = "create_init_input"
    OUTPUT_NODE = True
    CATEGORY = "Build and train your network"

    def create_init_input(self, tensor_Dimensions):
        return (torch.rand(eval(tensor_Dimensions)),)


class create_model():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "layer": ("LAYER",),
            },
        }

    RETURN_TYPES = ("MODELS_CLASS",)
    RETURN_NAMES = ('model',)
    FUNCTION = "create_init_model"
    OUTPUT_NODE = True
    CATEGORY = "Build and train your network"

    def create_init_model(self, layer):

        model = net(layer)
        return (model,)


class create_dataset:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "resize": ("BOOLEAN", {"default": True}),
                "normalize": ("BOOLEAN", {"default": True}),
                'train_data_path': ("STRING", {
                    "default": "D:/ComfyUI_windows_portable_nvidia_cu121_or_cpu/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui_train/dataset/train"}),
                "val_data": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "resize_w_h": ("STRING", {"default": "[512, 512]"}),
                "mean": ("STRING", {"default": "[0.485, 0.456, 0.406]"}),
                "std": ("STRING", {"default": "[0.229, 0.224, 0.225]"}),
                'val_data_path': ("STRING", {
                    "default": "D:/ComfyUI_windows_portable_nvidia_cu121_or_cpu/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui_train/dataset/val"})
            }
        }

    RETURN_TYPES = ("DATESET", "DATESET", 'TENSOR')
    RETURN_NAMES = ('train dataset', 'val dataset', 'test data')
    FUNCTION = "create_init_dataset"
    OUTPUT_NODE = True
    CATEGORY = "Build and train your network"

    def create_init_dataset(self, resize, normalize, train_data_path, resize_w_h='[512, 512]',
                            mean='[0.485, 0.456, 0.406]', std='[0.229, 0.224, 0.225]', val_data=True,
                            val_data_path=None):
        resize_transform = transforms.Resize(eval(resize_w_h)) if resize else None

        normalize_transform = transforms.Normalize(mean=eval(mean),
                                                   std=eval(std)) if normalize else None
        transform_list = []
        transform_list.append(transforms.ToTensor())
        if resize_transform is not None:
            transform_list.append(resize_transform)
        if normalize_transform is not None:
            transform_list.append(normalize_transform)
        transform = transforms.Compose(transform_list)

        train_dataset = datasets.ImageFolder(root=train_data_path, transform=transform)
        if val_data:
            val_dataset = datasets.ImageFolder(root=val_data_path, transform=transform)
            return (train_dataset, val_dataset, train_dataset[0][0].unsqueeze(0))
        return (train_dataset, None, train_dataset[0][0].unsqueeze(0))


class create_training_task:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "train_dataset": ("DATESET",),
                "model": ("MODELS_CLASS",),
            },
            "optional": {
                "val_dataset": ("DATESET",),
                "batch_size": ("INT", {"default": 8, "min": 1, "max": 0xffffffffffffffff, "step": 1, }),
                "epochs": ("INT", {"default": 100, "min": 1, "max": 0xffffffffffffffff, "step": 1, })

            }
        }

    RETURN_TYPES = ("MODELS_CLASS",)
    RETURN_NAMES = ('model',)
    FUNCTION = "create_init_train"
    OUTPUT_NODE = True
    CATEGORY = "Build and train your network"

    def create_init_train(self, train_dataset, model, val_dataset=None, batch_size=8, epochs=100):
        if val_dataset is None:
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            trainer = Trainer(max_epochs=epochs, accelerator='auto', devices=1)
            trainer.fit(model, train_dataloaders=train_dataloader)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            trainer = Trainer(max_epochs=epochs, accelerator='auto', devices=1)
            trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
            return (model,)
        return (model,)
