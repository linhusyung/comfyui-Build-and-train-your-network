import torch
import torchvision.models as models
from torch import nn


class vgg_16(nn.Module):
    def __init__(self, use_weights):
        super().__init__()
        if use_weights:
            self.vgg16 = models.vgg16(weights='IMAGENET1K_V1').features
        else:
            self.vgg16 = models.vgg16(weights=None).features

    def forward(self, x):
        out = self.vgg16.features(x)
        return out


class resnet_50(nn.Module):
    def __init__(self, use_weights):
        super().__init__()
        if use_weights:
            self.resnet50 = models.resnet50(weights='IMAGENET1K_V1')
        else:
            self.resnet50 = models.resnet50(weights=None)
        self.features = nn.Sequential(
            self.resnet50.conv1,
            self.resnet50.bn1,
            self.resnet50.relu,
            self.resnet50.maxpool,
            self.resnet50.layer1,
            self.resnet50.layer2,
            self.resnet50.layer3,
            self.resnet50.layer4,
        )

    def forward(self, x):
        x = self.features(x)
        return x


class inception_v3(nn.Module):
    def __init__(self, use_weights):
        super().__init__()
        if use_weights:
            self.inception_v3 = models.inception_v3(weights='IMAGENET1K_V1')
            print(1)
        else:
            self.inception_v3 = models.inception_v3(weights=None)

        self.features = nn.Sequential(
            self.inception_v3.Conv2d_1a_3x3,
            self.inception_v3.Conv2d_2a_3x3,
            self.inception_v3.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.inception_v3.Conv2d_3b_1x1,
            self.inception_v3.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.inception_v3.Mixed_5b,
            self.inception_v3.Mixed_5c,
            self.inception_v3.Mixed_5d,
            self.inception_v3.Mixed_6a,
            self.inception_v3.Mixed_6b,
            self.inception_v3.Mixed_6c,
            self.inception_v3.Mixed_6d,
            self.inception_v3.Mixed_6e,
            self.inception_v3.Mixed_7a,
            self.inception_v3.Mixed_7b,
            self.inception_v3.Mixed_7c,
        )

    def forward(self, x):
        x = self.features(x)
        return x


class efficientnet_b0(nn.Module):
    def __init__(self, use_weights):
        super().__init__()
        if use_weights:
            self.efficientnet_b0 = models.efficientnet_b0(weights='IMAGENET1K_V1').features
        else:
            self.efficientnet_b0 = models.efficientnet_b0(weights=None).features

    def forward(self, x):
        out = self.efficientnet_b0.features(x)
        return out


if __name__ == '__main__':
    x = torch.rand([1, 3, 512, 46])

    # model = resnet_50(True)
    # output = model(x)
    # print(output.shape)

    vgg_16 = vgg_16(True)
    out = vgg_16(x)
    print(out.shape)
    out = nn.AdaptiveAvgPool2d([7,7])(out)
    print(out.shape)

    # inception_v3 = inception_v3(True)
    # print(inception_v3(x).shape)

    # efficientnet_b0 = efficientnet_b0(True)
    # print(efficientnet_b0(x).shape)
