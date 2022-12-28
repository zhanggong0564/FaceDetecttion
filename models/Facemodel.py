import  torch
import torch.nn as nn
from models.backbone import resnet18,Mobilenetv2
from models.head import FPN,DetectHead

class FaceModel(nn.Module):
    def __init__(self,backbone,neck,head):
        super(FaceModel,self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self,x):
        out = self.backbone(x)
        out = self.neck(out)
        out = self.head(out)
        return out
if __name__ == '__main__':
    backbone = Mobilenetv2()
    neck = FPN(wide=24)
    head = DetectHead(wide=24)
    model = FaceModel(backbone,neck,head)
    x = torch.randn(3,3,800,800)
    y = model(x)
    for i in y:
        print(i.shape)