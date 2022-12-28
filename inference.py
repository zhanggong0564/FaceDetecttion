from tools.utils import *
from tools.AverageMeter import AverageMeter
from data_process.FaceDatasets import CustomDateset
from torch.utils.data import DataLoader
from models import FaceModel, resnet18, FPN, DetectHead
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
    backbone = resnet18()
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
    model_path = "train_mode/conventional_training/log/Epoch_44_batch_2999.pt"
    model = load_model(model,model_path)
    model.eval()
    image = cv2.imread('/mnt/e/workspace/FaceDetection/deploys/images/0_Parade_marchingband_1_483.jpg')
    image_show = cv2.resize(image, (800, 800)).astype(np.float32)
    image = process(image_show)
    out = model(image)
    predict_point = out[0].sigmoid()
    predict_wh = out[2]
    predict_offset = out[1]
    pool_predict_point = F.max_pool2d(predict_point, kernel_size=3, padding=1, stride=1)
    plt.imshow(pool_predict_point.data.numpy()[0, 0])

    threshold = 0.5
    select_nms_mask = pool_predict_point.eq(predict_point) & (predict_point > threshold)
    plt.imshow(select_nms_mask.data.numpy()[0, 0])

    position_y, position_x = torch.where(select_nms_mask[0, 0] == 1)
    for y, x in zip(position_y, position_x):
        # predict_offset, predict_wh
        w, h = torch.exp(predict_wh[0, :, y, x]).data.numpy() * stride
        # w, h = np.exp(predict_wh[0, :, y, x].data.numpy()) * stride-1
        ox, oy = predict_offset[0, :, y, x].data.numpy() * stride
        image_px, image_py = x * stride + ox, y * stride + oy

        # 四舍五入
        bx = int(image_px - w * 0.5 + 0.5)
        by = int(image_py - h * 0.5 + 0.5)
        br = int(image_px + w * 0.5 + 0.5)
        bb = int(image_py + h * 0.5 + 0.5)
        cv2.rectangle(image_show, (bx, by), (br, bb), (0, 255, 0), 1)
        # cv2.circle(show, (image_px, image_py), 20, (0, 255, 0), 2, 16)
    cv2.imshow('image_show',image_show)
    cv2.waitKey()
    cv2.destroyAllWindows()
    # print(image.shape)
    # print(image)
    plt.show()






