
from models.Facemodel import FaceModel
from models.backbone import resnet18
from models.head import FPN,DetectHead
import torch
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader
import argparse
import utils
import json
import losses
from visdom import Visdom
import cv2

class Config:
    def __init__(self):
        self.base_directory = "workspace"
        self.batch_size = 32
        self.gpu = "cuda:0"
        self.name = "default"      # 实验名称

    def get_path(self, path):
        return f"{self.base_directory}/{self.name}/{path}"

    def __repr__(self):
        return json.dumps(self.__dict__, indent=4)


def train():

    utils.setup_seed(31)
    utils.mkdirs(config.get_path("models"))
    utils.copy_code_to(".", config.get_path("code"))

    device = config.gpu
    train_set = dataset.Dataset(config.width, config.height, 
        "/data-rbd/wish/four_lesson/dataset/webface/train/label.txt", 
        "/data-rbd/wish/four_lesson/dataset/webface/WIDER_train/images")

    train_dataloader = DataLoader(train_set, batch_size=config.batch_size, num_workers=16, shuffle=True, pin_memory=True)

    glr = 2e-3
    backbone = resnet18()
    neck = FPN(wide=24)
    head = DetectHead(wide=24)
    model = FaceModel(backbone, neck, head)

    if config.weight != "":
        if os.path.exists(config.weight):
            check_point = torch.load(config.weight)
            model.load_state_dict(check_point)
        else:
            logger.error(f"Weight not exists: {config.weight}")

    model.to(device)
    optim = torch.optim.Adam(model.parameters(), glr)
    
    epochs = 150
    num_iter = 0
    coordinate_loss_function = losses.GIoULoss()
    point_loss_function = losses.focal_loss()

    stride = 4
    heatmap_width, heatmap_height = config.width // stride, config.height // stride
    gy, gx = torch.meshgrid(torch.arange(heatmap_height), torch.arange(heatmap_width))

    # cell_grid，  1x4x200x200
    cell_grid = torch.stack((gx, gy, gx, gy), dim=0).unsqueeze(0).to(device)
    num_iter_per_epoch = len(train_dataloader)

    lr_schedule = {
        1: 1e-3,
        100: 1e-4,
        130: 1e-5
    }

    for epoch in range(epochs):

        # 调整学习率
        if epoch in lr_schedule:
            glr = lr_schedule[epoch]
            
            for group in optim.param_groups:
                group["lr"] = glr

        model.train()

        # images ->  batch, 3, height, width
        # raw_images -> batch, height, width, 3
        # point_targets -> batch, 1, fm_height, fm_width
        # coord_targets -> batch, 4, fm_height, fm_width
        # mask_targets -> batch, 4, fm_height, fm_width
        for batch_index, (images, raw_images, point_targets, coord_targets, mask_targets) in enumerate(train_dataloader):
            images = images.to(device)
            point_targets = point_targets.to(device)
            coord_targets = coord_targets.to(device)
            mask_targets = mask_targets.to(device)
            point_predict, coord_predict = model(images)
            point_logits = point_predict.sigmoid()

            # coord_predict ->   batch, 4, 200, 200
            # cell_grid     ->   1, 4, 200, 200
            # coord_predict之后维度： batch, 200, 200, 4
            coord_predict = ((coord_predict + cell_grid) * stride).permute(0, 2, 3, 1)
            coord_targets = coord_targets.permute(0, 2, 3, 1)
            mask_targets = mask_targets.permute(0, 2, 3, 1)
            coord_restore = coord_predict[mask_targets].view(-1, 4)
            
            coord_select = coord_targets[mask_targets].view(-1, 4)
            coordinate_loss = coordinate_loss_function(coord_restore, coord_select)
            point_loss = point_loss_function(point_logits, point_targets)
            loss = point_loss + coordinate_loss
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            num_iter += 1

            if num_iter % 100 == 0:
                current_epoch = epoch + batch_index / num_iter_per_epoch
                logger.info(f"Iter: {num_iter}, Epoch: {current_epoch:.2f}/{epochs}, LR: {glr}, Loss: {loss.item():.3f}, Point: {point_loss.item():.3f}, Coord: {coordinate_loss.item():.3f}")

                show_batch = -1
                point_logits_select = point_logits[show_batch].detach().unsqueeze(0)
                point_pooled = F.max_pool2d(point_logits_select, kernel_size=(3, 3), stride=1, padding=1)
                ys, xs = torch.where((point_logits_select[0, 0] == point_pooled[0, 0]) & (point_logits_select[0, 0] > 0.3))
                raw_image_select = raw_images[show_batch].numpy()  # hxwx3
                for y, x in zip(ys, xs):
                    px, py, pr, pb = coord_predict[show_batch, y, x, :].detach().cpu().long()
                    cv2.rectangle(raw_image_select, (px, py), (pr, pb), (0, 255, 0), 2)

                visual.image(raw_image_select[..., ::-1].transpose(2, 0, 1), win="raw_image", opts={"title": "原始图"})
                visual.image(point_logits_select[0].cpu().numpy(), win="point_logits", opts={"title": "热力图预测值"})
                visual.image(point_targets[show_batch].cpu().numpy(), win="point_targets", opts={"title": "热力图真值"})
        
        if (epoch + 1) % 10 == 0:
            model_path = config.get_path(f"models/{epoch:05d}.pth")
            logger.info(f"Save model to {model_path}")
            torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="实验名称", default="default")
    parser.add_argument("--batch_size", type=int, help="批大小", default=8)
    parser.add_argument("--gpu", type=str, help="定义设备", default="cuda:0")
    parser.add_argument("--weight", type=str, help="预训练模型", default="")
    args = parser.parse_args()

    visual = Visdom(server='http://127.0.0.1', port=8097, env=args.name)
    assert visual.check_connection()

    config = Config()
    config.name = args.name
    config.batch_size = args.batch_size
    config.gpu = args.gpu
    config.width = 800
    config.height = 800
    config.weight = args.weight
    logger = utils.getLogger(config.get_path("logs/log.log"))
    logger.info(f"Startup, config: \n{config}")
    train()