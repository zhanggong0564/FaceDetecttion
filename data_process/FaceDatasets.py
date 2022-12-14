from tools import common
# import augment
import albumentations as A
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T


def bbox_detection_augment(outwidth, outheight, mean, std):
    return A.Compose([
        # augment.RandomAffine(outwidth, outheight),
        A.HorizontalFlip(),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=27),
            A.RandomContrast(limit=0.8),
            A.JpegCompression(quality_lower=5, quality_upper=100),
        ]),
        A.OneOf([
            A.ISONoise(),
            A.IAAAdditiveGaussianNoise(),
            A.IAASharpen(),
        ]),
        # A.OneOf([
        #     A.Cutout(num_holes=32, max_h_size=24, max_w_size=24, p=0.5),
        #     A.RandomRain(p=0.2),
        #     A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2),
        #     A.IAAPerspective(p=0.5)
        # ]),
        A.OneOf([
            A.Blur(blur_limit=9),
            A.MotionBlur(p=1, blur_limit=7),
            A.GaussianBlur(blur_limit=21),
            A.GlassBlur(),
            A.ToGray(),
            A.RandomGamma(gamma_limit=(0, 120), p=0.5),
        ]),
        # ToTensor(normalize={"mean": mean, "std": std})
    ], bbox_params=A.BboxParams("pascal_voc"))


class Dataset:
    def __init__(self, width, height, annotation, root):
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
        self.images = common.load_widerface_annotation(annotation)

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
        stride = 4
        heatmap_height = self.height // stride
        heatmap_width = self.width // stride
        point_heatmap = np.zeros((1, heatmap_height, heatmap_width), dtype=np.float32)
        coord_heatmap = np.zeros((4, heatmap_height, heatmap_width), dtype=np.float32)
        mask_heatmap = np.zeros((4, heatmap_height, heatmap_width), dtype=np.bool)

        for x, y, r, b, invalid in trans_out["bboxes"]:
            cx, cy = (x + r) * 0.5, (y + b) * 0.5
            box_width, box_height = (r - x + 1) / stride, (b - y + 1) / stride
            cell_x, cell_y = int(cx / stride + 0.5), int(cy / stride + 0.5)
            cell_x = max(0, min(cell_x, heatmap_width - 1))
            cell_y = max(0, min(cell_y, heatmap_height - 1))

            common.draw_gauss(point_heatmap[0], cell_x, cell_y, (box_width, box_height))
            if not invalid:
                coord_heatmap[:, cell_y, cell_x] = x, y, r, b
                mask_heatmap[:, cell_y, cell_x] = True

        # 如果在__getitem__返回的np.ndarray，到dataloader上，会转换为torch.tensor（不是用T.to_tensor做的）
        # bxhxwx3
        return self.convert_to_tensor(trans_out["image"]), trans_out["image"], torch.as_tensor(
            point_heatmap), torch.as_tensor(coord_heatmap), torch.as_tensor(mask_heatmap)

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    dataset = Dataset(800, 800,
                      "widerface/train/label.txt",
                      "widerface/train/images")

    print(len(dataset))
    image,image_, point, coord, mask = dataset[6857]

    # cv2.imwrite("image.jpg", image)
    cv2.imshow("image",image_)
    cv2.waitKey()
    cv2.destroyAllWindows()
    print(point.shape)