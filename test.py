import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn

if __name__ == '__main__':
    resize_w_h = '[512, 512]'
    resize_w_h = eval(resize_w_h)
    resize_transform = transforms.Resize(resize_w_h)

    transform = transforms.Compose([
        transforms.Resize(resize_w_h),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root='./dataset/train', transform=transform)
    val_dataset = datasets.ImageFolder(root='./dataset/val', transform=transform)

    batch_size = int(32)
    print(batch_size, type(batch_size))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    for data, label in train_loader:
        print(data.shape, label.shape)
    mean = torch.zeros(3)
    std = torch.zeros(3)

    for images, labels in train_loader:
        print(images.shape)
        for i in range(3):
            mean[i] += images[:, i, :, :].mean()
            std[i] += images[:, i, :, :].std()
    mean.div_(len(train_dataset))
    std.div_(len(train_dataset))
    print(mean, std)
