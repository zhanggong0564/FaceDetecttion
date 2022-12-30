import os.path

from tools.utils import *
from tools.AverageMeter import AverageMeter
from data_process.FaceDatasets import CustomDateset
from torch.utils.data import DataLoader
from models import FaceModel, resnet18, FPN, DetectHead,Mobilenetv2
import torch
from tools.model_trainer import ModelTrainer
from config.config import cfg
import argparse
from losses.losses import *
import cv2
import torch.nn.functional as F

mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]

def build_model(wide):
    backbone = Mobilenetv2()
    neck = FPN(wide=wide)
    head = DetectHead(wide=wide)
    model = FaceModel(backbone, neck, head)
    return model
def load_model(model,model_path):
    model_dict = torch.load(model_path)['state_dict']
    model.load_state_dict(model_dict)
    return model

def process(image):
    image/=255.0
    image-=mean
    image/=std

    image = np.transpose(image,(2,0,1))
    image = np.expand_dims(image,0)
    image = torch.from_numpy(image)
    return image







if __name__ == '__main__':
    stride = 4
    model = build_model(24)
    model_path = "Epoch_80.pt"
    model = load_model(model,model_path)
    model.eval()
    image_flow_path = './widerface/val/images/0--Parade'
    image_paths = os.listdir(image_flow_path)
    image_paths = [os.path.join(image_flow_path,i) for i in image_paths]
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image_show = cv2.resize(image, (800, 800)).astype(np.float32)
        image = process(image_show)
        out = model(image)
        predict_point = out[0].sigmoid()
        predict_wh = out[2]
        predict_offset = out[1]
        pool_predict_point = F.max_pool2d(predict_point, kernel_size=3, padding=1, stride=1)
        threshold = 0.3

        select_nms_masks = pool_predict_point.eq(predict_point) & (predict_point > threshold)
        select_nms_masks = torch.cat([select_nms_masks,select_nms_masks,select_nms_masks,select_nms_masks],1)

        gy, gx = torch.meshgrid(torch.arange(200), torch.arange(200))
        # cell_grid，  1x2x200x200
        cell_grid = torch.stack((gx, gy), dim=0).unsqueeze(0)
        # plt.imshow(select_nms_mask.data.numpy()[0, 0])

        w_h = torch.exp(predict_wh)*stride
        image_px_py = (predict_offset+cell_grid)*stride
        image_bxy = image_px_py - w_h * 0.5 + 0.5
        image_brb = image_px_py + w_h * 0.5 + 0.5
        image_bxy_brbs = torch.cat([image_bxy,image_brb],1)
        for n in range(len(image_bxy_brbs)) :
            box_infos = image_bxy_brbs[n][select_nms_masks[n]].reshape(4, -1).transpose(0, 1)
            confs =predict_point[n][select_nms_masks[n][0][None]]
            for (bx, by, br, bb),conf in zip(box_infos,confs):
                bx, by, br, bb = int(bx),int(by),int(br),int(bb)
                conf  = round (float(conf),2)
                cv2.rectangle(image_show, (bx, by), (br, bb), (0, 255, 0), 1)
                cv2.putText(image_show,str(conf),(bx,by), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.imshow('image_show', image_show)
            cv2.waitKey()
    cv2.destroyAllWindows()






    # position_y, position_x = torch.where(select_nms_mask[0, 0] == 1)
    #
    # for y, x in zip(position_y, position_x):
    #     # predict_offset, predict_wh
    #     w, h = torch.exp(predict_wh[0, :, y, x]).data.numpy() * stride
    #     # w, h = np.exp(predict_wh[0, :, y, x].data.numpy()) * stride-1
    #     ox, oy = predict_offset[0, :, y, x].data.numpy() * stride
    #     image_px, image_py = x * stride + ox, y * stride + oy
    #
    #     # 四舍五入
    #     bx = int(image_px - w * 0.5 + 0.5)
    #     by = int(image_py - h * 0.5 + 0.5)
    #     br = int(image_px + w * 0.5 + 0.5)
    #     bb = int(image_py + h * 0.5 + 0.5)
    #     cv2.rectangle(image_show, (bx, by), (br, bb), (0, 255, 0), 1)
    #     # cv2.circle(show, (image_px, image_py), 20, (0, 255, 0), 2, 16)
    # cv2.imshow('image_show',image_show)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # # print(image.shape)
    # # print(image)
    # plt.show()






