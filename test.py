import torch
import torch.nn as nn
import torchvision.models as models

if __name__ == '__main__':
    # vgg = models.vgg16(weights='VGG16_Weights.DEFAULT')
    # vgg = models.vgg16(weights=None)
    # model = nn.ModuleList()
    # model.append(vgg.features)
    # model.append(nn.Linear(in_features=10, out_features=10))
    # print(model)
    # out = vgg.features(x)
    # print(out.shape)
    # print(vgg)

    x = torch.rand([1, 3, 32, 32])
    # adaptive_avgpool = nn.AdaptiveAvgPool2d([5, 5])
    adaptive_avgpool = nn.MaxPool2d([3, 3])
    # adaptive_avgpool = nn.AvgPool2d(kernel_size=[3, 3])
    output_features = adaptive_avgpool(x)
    print(output_features.shape)

if mode == 'adaptive_avgpool':
    layer_a[0].append(
        nn.AdaptiveAvgPool2d(eval(normalized_shape)))
if mode == 'MaxPool2d':
    layer_a[0].append(
        nn.MaxPool2d(eval(normalized_shape)))
if mode == 'AvgPool2d':
    layer_a[0].append(
        nn.AvgPool2d(eval(normalized_shape)))