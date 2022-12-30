import torch

from data_process.TestDatasets import CustomTestDateset
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import FaceModel, FPN, DetectHead,Mobilenetv2
from losses.losses import *
from collections import defaultdict
from evaluate.evaluation import MAPTool
import  cv2
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
def collete(outputs):
    dicts = {}
    images = []
    image_ids = []
    for image,dict_image,image_id in outputs:
        images.append(image)
        dicts.update(dict_image)
        image_ids.append(image_id)
    images = torch.stack(images,0)
    return images,dicts,image_ids


if __name__ == '__main__':
    MAP = MAPTool(['face'])
    dataset = CustomTestDateset(800, 800,
                                "../widerface/train/label.txt",
                                "../widerface/train/images")

    dataloader = DataLoader(dataset,batch_size=4,collate_fn=collete)
    dataloader = tqdm(dataloader)
    model = build_model(24)
    model_path = "../Epoch_80.pt"
    model = load_model(model, model_path).cuda()
    model.eval()
    stride = 4
    threshold = 0.5
    groudTrueth_map_dicts = defaultdict()
    detection_map_ditcs = defaultdict(list) ##{image_id: [[left, top, right, bottom, confidence, classes_index], [left, top, right, bottom, confidence, classes_index]]}
    for inputs,groudTrueth_map_dict,image_ids in dataloader:
        groudTrueth_map_dicts.update(groudTrueth_map_dict)
        inputs = inputs.cuda()
        with torch.no_grad():
            out = model(inputs)
            predict_point = out[0].sigmoid()
            predict_wh = out[2]
            predict_offset = out[1]
            pool_predict_point = F.max_pool2d(predict_point, kernel_size=3, padding=1, stride=1)
            threshold = 0.3

            select_nms_masks = pool_predict_point.eq(predict_point) & (predict_point > threshold)
            select_nms_masks = torch.cat([select_nms_masks, select_nms_masks, select_nms_masks, select_nms_masks], 1)

            gy, gx = torch.meshgrid(torch.arange(200), torch.arange(200))
            # cell_grid，  1x2x200x200
            cell_grid = torch.stack((gx, gy), dim=0).unsqueeze(0).cuda()
            # plt.imshow(select_nms_mask.data.numpy()[0, 0])

            w_h = torch.exp(predict_wh) * stride
            image_px_py = (predict_offset + cell_grid) * stride
            image_bxy = image_px_py - w_h * 0.5 + 0.5
            image_brb = image_px_py + w_h * 0.5 + 0.5
            image_bxy_brbs = torch.cat([image_bxy, image_brb], 1)
            for n in range(len(image_bxy_brbs)):
                box_infos = image_bxy_brbs[n][select_nms_masks[n]].reshape(4, -1).transpose(0, 1)
                confs = predict_point[n][select_nms_masks[n][0][None]]
                image_id = image_ids[n]
                for (bx, by, br, bb), conf in zip(box_infos, confs):
                    bx, by, br, bb = int(bx), int(by), int(br), int(bb)
                    conf = round(float(conf), 2)
                    detection_map_ditcs[image_id].append([ bx, by, br, bb,conf,0])
    MAP.cal_map(groudTrueth_map_dicts,detection_map_ditcs)



