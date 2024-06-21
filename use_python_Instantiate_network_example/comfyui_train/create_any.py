import copy
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pytorch_lightning import Trainer
import torchmetrics
import pickle


class net(pl.LightningModule):
    def __init__(self, layer='', loss='', lr=''):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layer = layer[0]
        self.res_seq = list(layer[1])

        for idx, i in enumerate(self.res_seq):
            temp_list = list(i)
            temp_list[3] = temp_list[3].to(device)
            self.res_seq[idx] = tuple(temp_list)

        if loss == 'CrossEntropyLoss':
            self.loss = nn.CrossEntropyLoss()
        if loss == 'Binary Cross Entropy Loss':
            self.loss = nn.BCEWithLogitsLoss()
        # self.BCE = nn.BCELoss()
        # self.MSE = nn.MSELoss()

        self.lr = lr
        # self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2)

    def res_cpu(self):
        for idx, i in enumerate(self.res_seq):
            temp_list = list(i)
            temp_list[3] = temp_list[3].cpu()
            self.res_seq[idx] = tuple(temp_list)

    def res_cuda(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for idx, i in enumerate(self.res_seq):
            temp_list = list(i)
            temp_list[3] = temp_list[3].to(device)
            self.res_seq[idx] = tuple(temp_list)

    def forward(self, x):
        rt_res_out = []
        outputs = [x]
        for i, layer in enumerate(self.layer):
            out = layer(outputs[-1])

            for idx, res in enumerate(self.res_seq):
                if i == res[1]:
                    if len(res) >= 4:
                        cache = outputs[res[0]]
                        res_out_cecha = []
                        if res[3] is None:
                            res_out = outputs[res[0]]
                            res_out_cecha.append(res_out)
                        else:
                            for k in res[3]:
                                res_out = k(cache)
                                cache = res_out
                                res_out_cecha.append(res_out)
                        rt_res_out.append(res_out_cecha)
                        if res[2] == 'add':
                            out += res_out
                        if res[2] == 'concat-dim=0':
                            out = torch.cat((out, res_out), dim=0)
                        if res[2] == 'concat-dim=1':
                            out = torch.cat((out, res_out), dim=1)
                        if res[2] == 'concat-dim=2':
                            out = torch.cat((out, res_out), dim=2)
                        if res[2] == 'concat-dim=3':
                            out = torch.cat((out, res_out), dim=3)
                        if res[2] == 'Mul':
                            out *= res_out
                        if res[2] == 'div':
                            out /= res_out
                        if res[2] == 'sub':
                            out -= res_out
                    else:
                        res_out = outputs[res[0]]
                        if res[2] == 'add':
                            out += res_out
                        if res[2] == 'concat-dim=0':
                            out = torch.cat((out, res_out), dim=0)
                        if res[2] == 'concat-dim=1':
                            out = torch.cat((out, res_out), dim=1)
                        if res[2] == 'concat-dim=2':
                            out = torch.cat((out, res_out), dim=2)
                        if res[2] == 'concat-dim=3':
                            out = torch.cat((out, res_out), dim=3)
                        if res[2] == 'Mul':
                            out *= res_out
                        if res[2] == 'div':
                            out /= res_out
                        if res[2] == 'sub':
                            out -= res_out

            outputs.append(out)
        return outputs[-1], (outputs, rt_res_out)

    def training_step(self, batch, batch_idx):
        data, label = batch
        out, out_list = self(data)
        loss = self.loss(out, label)
        # acc = self.accuracy(torch.argmax(out), label)
        self.log('train_loss', loss, prog_bar=True)
        # self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        out, out_list = self(data)
        loss = self.loss(out, label)
        # acc = self.accuracy(torch.argmax(out), label)
        self.log('val_loss', loss, prog_bar=True)
        # self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
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
                "lr": ("STRING", {"default": "0.001"}),
                "loss": (
                    ['CrossEntropyLoss', 'Binary Cross Entropy Loss'],),
                "save_serialization": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "save_path": ("STRING", {"default": "D:/k.pkl"}),
            }
        }

    RETURN_TYPES = ("MODELS_CLASS",)
    RETURN_NAMES = ('model',)
    FUNCTION = "create_init_model"
    OUTPUT_NODE = True
    CATEGORY = "Build and train your network"

    def create_init_model(self, layer, lr, loss, save_serialization=True,save_path=''):
        if save_serialization:
            with open(save_path, 'wb') as f:
                pickle.dump(layer, f)

        model = net(layer, lr=float(lr), loss=loss)
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
                "number_of_image": ("INT", {"default": 0, "min": 0, "step": 1, }),
            },
            "optional": {
                "resize_w_h": ("STRING", {"default": "[512, 512]"}),
                "mean": ("STRING", {"default": "[0.485, 0.456, 0.406]"}),
                "std": ("STRING", {"default": "[0.229, 0.224, 0.225]"}),
                'val_data_path': ("STRING", {
                    "default": "D:/ComfyUI_windows_portable_nvidia_cu121_or_cpu/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui_train/dataset/val"}),
                "forward_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("DATESET", "DATESET", 'TENSOR', 'TENSOR', 'IMAGE', 'TENSOR')
    RETURN_NAMES = ('train dataset', 'val dataset', 'test data', 'test label', 'image show', 'Data preprocessing')
    FUNCTION = "create_init_dataset"
    OUTPUT_NODE = True
    CATEGORY = "Build and train your network"

    def create_init_dataset(self, resize, normalize, train_data_path, number_of_image, resize_w_h='[512, 512]',
                            mean='[0.485, 0.456, 0.406]', std='[0.229, 0.224, 0.225]', val_data=True,
                            val_data_path=None, forward_image=None):

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

        pre_img = None
        if forward_image is not None:
            pre_transform_list = transform_list[1:]
            pre_transform = transforms.Compose(pre_transform_list)
            forward_image = forward_image.permute(0, 3, 1, 2).contiguous()
            pre_img = pre_transform(forward_image)

        train_dataset = datasets.ImageFolder(root=train_data_path, transform=transform)
        if val_data:
            val_dataset = datasets.ImageFolder(root=val_data_path, transform=transform)
            return (train_dataset, val_dataset, train_dataset[0][0].unsqueeze(0), train_dataset[0][1],
                    train_dataset[number_of_image][0].unsqueeze(0).permute(0, 2, 3, 1), pre_img)
        return (train_dataset, None, train_dataset[0][0].unsqueeze(0), train_dataset[0][1],
                train_dataset[number_of_image][0].unsqueeze(0).permute(0, 2, 3, 1), pre_img)


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
                "epochs": ("INT", {"default": 100, "min": 1, "max": 0xffffffffffffffff, "step": 1, }),

            }
        }

    RETURN_TYPES = ("MODELS_CLASS",)
    RETURN_NAMES = ('model',)
    FUNCTION = "create_init_train"
    OUTPUT_NODE = True
    CATEGORY = "Build and train your network"

    def create_init_train(self, train_dataset, model, val_dataset=None, batch_size=8, epochs=100,
                          ):
        model.res_cuda()
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
