# import resnet
import torch.nn as nn
import torch.nn.functional as F
import torch


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
        self.u4 = upmodule(320, wide)
        self.p3 = projection_module(96, wide)

        self.u3 = upmodule(wide, wide)
        self.p2 = projection_module(32, wide)

        self.u2 = upmodule(wide, wide)
        self.p1 = projection_module(24, wide)

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
        return o2
class DetectHead(nn.Module):
    def __init__(self,wide,deploy=False):
        super(DetectHead,self).__init__()
        self.point = head_module(wide, 1)
        self.offset = head_module(wide, 2)
        self.coordinate = head_module(wide, 2)
        self.deploy = deploy
    def forward(self,x):
        point = self.point(x)
        offset = self.offset(x)
        wh = self.coordinate(x)
        if self.deploy:
            point = point.sigmoid()
            point = point.permute(0, 2, 3, 1).contiguous().view(-1, 1)
            offset = offset.permute(0, 2, 3, 1).contiguous().view(-1, 2)
            wh = wh.permute(0, 2, 3, 1).contiguous().view(-1, 2)
            return torch.cat([point, offset, wh], -1)
        return point,offset,wh