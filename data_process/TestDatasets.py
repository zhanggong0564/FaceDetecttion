from tools import common
# import augment
import albumentations as A
from albumentations.pytorch import  ToTensorV2
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from collections import defaultdict


def bbox_detection_augment(outwidth, outheight, mean, std):
    return A.Compose([
        A.Resize(height=outheight, width=outwidth),
        A.Normalize(mean=mean,std=std),
        ToTensorV2()
    ], bbox_params=A.BboxParams("pascal_voc"))


class CustomTestDateset(Dataset):
    def __init__(self, width, height, annotation, root):
        super(CustomTestDateset, self).__init__()
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        self.width = width
        self.height = height
        self.transform = bbox_detection_augment(width, height, mean, std)
        self.annotation = annotation
        self.root = root
        self.convert_to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])

        assert os.path.exists(annotation), f"{annotation} not exists"
        self.images = common.load_widerface_annotation(annotation)[:240]

        assert len(self.images) > 0, "images is empty"

    def __getitem__(self, index):
        item = self.images[index]
        image_path = f"{self.root}/{item.file}"
        image = cv2.imread(image_path)

        # location + false，构造n x 5维
        # False指，是否转过头了（大于30度），如果转过头，为True，否则为False
        bboxes = [box.location + (False,) for box in item.bboxes if box.width >= 10 and box.height >= 10]

        if image is None:
            print(f"empty image: {image_path}")

            # 可能有部分图像是空的，读取失败的。不能影响整体训练进度
            image = (np.random.normal(size=(self.height, self.width, 3)) * 255).astype(np.uint8)
            bboxes = []

        trans_out = self.transform(image=image, bboxes=bboxes)
        ##{image_id: [[left, top, right, bottom, 0, classes_index], [left, top, right, bottom, 0, classes_index]]}
        map_dict = defaultdict(list)
        image_id =item.file.split('/')[-1]
        for x, y, r, b, invalid in trans_out["bboxes"]:
            map_dict[image_id].append([x,y,r,b,0,0])
        return trans_out["image"],map_dict,image_id

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    dataset = CustomTestDateset(800, 800,
                            "../widerface/train/label.txt",
                            "../widerface/train/images")

    print(len(dataset))
    image, dict_= dataset[6857]
    print(image)
    print(dict_)
