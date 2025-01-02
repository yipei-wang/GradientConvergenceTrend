import torch
import numpy as np
import torchvision
from torch import nn

class ResBlock(nn.Module):
    
    def __init__(self, input_size, output_size):
        super(ResBlock, self).__init__()

        if input_size != output_size:
            self.conv1 = nn.Conv2d(input_size, output_size, kernel_size = 3, stride = 2, padding = 1, bias = False)
        else:
            self.conv1 = nn.Conv2d(input_size, output_size, kernel_size = 3, stride = 1, padding = 1, bias = False)

        self.bn1 = nn.BatchNorm2d(output_size)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(output_size, output_size, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(output_size)
        if input_size != output_size:
            self.downsample = nn.Sequential(
                nn.Conv2d(input_size, output_size,kernel_size=1,stride = 2, bias = False),
                nn.BatchNorm2d(output_size)
            )
        else:
            self.downsample = nn.Identity()


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)

        return out

    
class ResNetSmall(nn.Module):
    
    def __init__(self, k= 32, n_class = 10):
        super(ResNetSmall, self).__init__()

        self.conv1 = nn.Conv2d(3,k,kernel_size = 7, stride = 2, padding =3, bias = False)
        self.bn1 = nn.BatchNorm2d(k)
        self.relu = nn.ReLU()

        self.layer1 = nn.Sequential(
            ResBlock(k, k), ResBlock(k,k))
        self.layer2 = nn.Sequential(
            ResBlock(k, 2*k))
        self.layer3 = nn.Sequential(
            ResBlock(2*k, 4*k))
        self.layer4 = nn.Sequential(
            ResBlock(4*k, 8*k))

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(8*k, n_class)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)

        return x
        

class ResNetLarge(nn.Module):
    
    def __init__(self, k= 32, n_class = 10):
        super(ResNetLarge, self).__init__()

        self.conv1 = nn.Conv2d(3,k,kernel_size = 7, stride = 2, padding =3, bias = False)
        self.bn1 = nn.BatchNorm2d(k)
        self.relu = nn.ReLU()

        self.layer1 = nn.Sequential(
            ResBlock(k, k), ResBlock(k,k))
        self.layer2 = nn.Sequential(
            ResBlock(k, 2*k), ResBlock(2*k, 2*k))
        self.layer3 = nn.Sequential(
            ResBlock(2*k, 4*k), ResBlock(4*k, 4*k))
        self.layer4 = nn.Sequential(
            ResBlock(4*k, 8*k), ResBlock(8*k, 8*k))

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(8*k, n_class)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)

        return x



class CNNLarge(nn.Module):
    def __init__(self,
                 k=32,
                 n_class = 10):
        super(CNNLarge, self).__init__()

        self.k = k
        self.backbone = nn.Sequential(
            nn.Conv2d(3, k, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(k),
            nn.ReLU(),
            nn.Conv2d(k, k, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(k),
            nn.ReLU(),
            nn.Conv2d(k, 2 * k, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2 * k),
            nn.ReLU(),
            nn.Conv2d(2 * k, 2 * k, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2 * k),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(2 * k, 4 * k, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4 * k),
            nn.ReLU(),
            nn.Conv2d(4 * k, 4 * k, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4 * k),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(4 * k, 8 * k, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8 * k),
            nn.ReLU(),
            nn.Conv2d(8 * k, 8 * k, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8 * k),
            nn.ReLU(),
            nn.MaxPool2d(8),
            nn.Flatten()
        )
        self.fc = nn.Linear(8 * k, n_class)

    def forward(self, x):
        feature = self.backbone(x)
        y = self.fc(feature)
        return y



class CNNSmall(nn.Module):
    def __init__(self,
                 k=32,
                 n_class = 10):
        super(CNNSmall, self).__init__()

        self.k = k
        self.backbone = nn.Sequential(
            nn.Conv2d(3, k, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(k),
            nn.ReLU(),
            nn.Conv2d(k, 2 * k, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2 * k),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(2 * k, 4 * k, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4 * k),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(4 * k, 8 * k, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8 * k),
            nn.ReLU(),
            nn.MaxPool2d(8),
            nn.Flatten()
        )
        self.fc = nn.Linear(8 * k, n_class)

    def forward(self, x):
        feature = self.backbone(x)
        y = self.fc(feature)
        return y