
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3_bn_relu(in_feature, out_feature, stride=1, relu=True, groups=1):
    modules = [
        nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False),
        nn.BatchNorm2d(out_feature)
    ]
    
    if relu:
        modules.append(nn.ReLU(inplace=True))
    return nn.Sequential(*modules)

def conv1x1_bn_relu(in_feature, out_feature, stride=1, relu=True):
    modules = [
        nn.Conv2d(in_feature, out_feature, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_feature)
    ]
    
    if relu:
        modules.append(nn.ReLU(inplace=True))
    return nn.Sequential(*modules)

class BasicBlock(nn.Module):
    expansion = 1   # 膨胀系数
    
    def __init__(self, in_feature, planes, stride=1, groups=1, width_per_group=64):
        super().__init__()
        
        assert groups == 1 and width_per_group == 64, f"{groups} == 1 and {width_per_group} == 64"
        self.conv1 = conv3x3_bn_relu(in_feature, planes, stride=stride)
        self.conv2 = conv3x3_bn_relu(planes, planes * self.expansion, relu=False)
        self.relu = nn.ReLU(inplace=True)
        
        # 如果有下采样
        if stride != 1 or in_feature != planes * self.expansion:
            self.shortcut = conv1x1_bn_relu(in_feature, planes * self.expansion, stride=stride, relu=False)
        else:
            self.shortcut = nn.Identity()
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + self.shortcut(identity)
        return self.relu(x)

class BottleneckBlock(nn.Module):
    expansion = 4  # 膨胀系数
    
    def __init__(self, in_feature, planes, stride=1, groups=1, width_per_group=64):
        super().__init__()

        #width = (planes * width_per_group) // 64 * groups
        coeff = planes // 64
        width = coeff * width_per_group * groups
        self.conv1 = conv1x1_bn_relu(in_feature, width)
        self.conv2 = conv3x3_bn_relu(width, width, stride=stride, groups=groups)
        self.conv3 = conv1x1_bn_relu(width, planes * self.expansion, relu=False)
        
        # 1x1的卷积，通常会用来改变通道数，并使得通道一致，或者达成某种目的。促使目的完成
        # 目的指，比如说add操作，a、b操作数，是不是得要求a和b具有完全一样的大小和通道数
        # 1x1卷积去搞
        if stride != 1 or in_feature != planes * self.expansion:
            # 需要做映射
            self.shortcut = conv1x1_bn_relu(in_feature, planes * self.expansion, stride=stride, relu=False)
        else:
            self.shortcut = nn.Identity()
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        out = x + self.shortcut(identity)
        return F.relu(out, inplace=True)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, groups=1, width_per_group=64):
        super().__init__()
        
        self.in_feature = 64
        self.num_blocks = num_blocks
        self.block = block
        self.groups = groups
        self.width_per_group = width_per_group
        
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, self.in_feature, kernel_size=7, padding=3, stride=2, bias=False),
            nn.BatchNorm2d(self.in_feature),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self.make_layer(planes=64, stride=1, num_block=self.num_blocks[0])
        self.layer2 = self.make_layer(planes=128, stride=2, num_block=self.num_blocks[1])
        self.layer3 = self.make_layer(planes=256, stride=2, num_block=self.num_blocks[2])
        self.layer4 = self.make_layer(planes=512, stride=2, num_block=self.num_blocks[3])
        
    def make_layer(self, planes, stride, num_block):
        
        modules = [
            self.block(self.in_feature, planes=planes, stride=stride, groups=self.groups, width_per_group=self.width_per_group)
        ]
        
        self.in_feature = planes * self.block.expansion
        
        for i in range(num_block-1):
            modules.append(
                self.block(self.in_feature, planes=planes, groups=self.groups, width_per_group=self.width_per_group)
            )
        return nn.Sequential(*modules)

    def forward(self, x):
        x = self.layer0(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1,x2,x3,x4

def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    return ResNet(BottleneckBlock, [3, 4, 6, 3])

def resnet101():
    return ResNet(BottleneckBlock, [3, 4, 23, 3])

def resnet152():
    return ResNet(BottleneckBlock, [3, 8, 36, 3])

def resnext50_32x4d():
    return ResNet(BottleneckBlock, [3, 4, 6, 3], groups=32, width_per_group=4)

def resnext101_32x8d():
    return ResNet(BottleneckBlock, [3, 4, 23, 3], groups=32, width_per_group=8)