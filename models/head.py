# import resnet
import torch.nn as nn
import torch.nn.functional as F


def upmodule(in_feature, out_feature, scale=2):
    # Upsample + Conv + BN
    return nn.Sequential(
        nn.Upsample(scale_factor=scale, mode="nearest"),
        nn.Conv2d(in_feature, out_feature, kernel_size=3, padding=1, stride=1, bias=False),
        nn.BatchNorm2d(out_feature)
    )


def projection_module(in_feature, out_feature):
    # Conv + BN
    return nn.Sequential(
        nn.Conv2d(in_feature, out_feature, kernel_size=1, padding=0, stride=1, bias=False),
        nn.BatchNorm2d(out_feature)
    )


def head_module(in_feature, out_feature):
    head = nn.Conv2d(in_feature, out_feature, kernel_size=1, stride=1, padding=0)
    return nn.Sequential(
        nn.Conv2d(in_feature, in_feature, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        head
    )

class FPN(nn.Module):
    def __init__(self,wide):
        super(FPN, self).__init__()
        self.u4 = upmodule(512, wide)
        self.p3 = projection_module(256, wide)

        self.u3 = upmodule(wide, wide)
        self.p2 = projection_module(128, wide)

        self.u2 = upmodule(wide, wide)
        self.p1 = projection_module(64, wide)

        self.point = nn.Conv2d(wide, 1, kernel_size=1, padding=0, stride=1)
        self.coordinate = nn.Conv2d(wide, 4, kernel_size=1, padding=0, stride=1)

    def forward(self,inputs):
        x1,x2,x3,x4 = inputs
        u4 = self.u4(x4)
        p3 = self.p3(x3)
        o4 = F.relu(u4 + p3)  # 16倍

        u3 = self.u3(o4)
        p2 = self.p2(x2)
        o3 = F.relu(u3 + p2)  # 8倍

        u2 = self.u2(o3)
        p1 = self.p1(x1)
        o2 = F.relu(u2 + p1)  # 4倍
        point = self.point(o2)
        coordinate = self.coordinate(o2)
        return o2
class DetectHead(nn.Module):
    def __init__(self,wide):
        super(DetectHead,self).__init__()
        self.point = head_module(wide, 1)
        self.offset = head_module(wide, 2)
        self.coordinate = head_module(wide, 2)
    def forward(self,x):
        point = self.point(x)
        offset = self.offset(x)
        wh = self.coordinate(x)
        return point,offset,wh