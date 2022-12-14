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

set_seed(2021)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Training")
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--bs', default=4, type=int, help='training batchsize')
parser.add_argument('--resume', default=None, type=str, help='training resume path')
args = parser.parse_args()


def build_model(wide):
    backbone = resnet18()
    neck = FPN(wide=wide)
    head = DetectHead(wide=wide)
    model = FaceModel(backbone, neck, head)
    return model


def build_loss():
    criterion1 = SoomthL1Loss(sigma=3)
    point_loss_function = FocalLoss()

    def inner(preds, targets):
        point, offset, wh = preds
        _, point_target, offset_target, wh_target,mask_heatmap = targets
        point = point.sigmoid()
        loss1 = point_loss_function(point, point_target.cuda())
        loss2 = criterion1(offset, offset_target.cuda(),mask_heatmap.cuda())
        loss3 = criterion1(wh, wh_target.cuda(),mask_heatmap.cuda())
        loss = loss1 + loss2 + 0.2*loss3
        return loss

    return inner


if __name__ == '__main__':


    logger, log_dir = make_logger(cfg.exp_log)
    train_datasets = CustomDateset(cfg.target_width, cfg.target_height, cfg.label_path, cfg.image_root)
    train_loader = DataLoader(dataset=train_datasets, batch_size=args.bs, shuffle=True, num_workers=cfg.num_workers)
    model = build_model(wide=24)
    start_epoch = 0
    if cfg.pretrained and not args.resume:
        model = ModelTrainer.load_model(model, cfg.pretrained)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.factor, milestones=cfg.milestones)
    criterion = build_loss()
    loss_meter = AverageMeter()

    for epoch in range(start_epoch, cfg.end_epoch):
        ModelTrainer.train(data_loader=train_loader, model=model, criterion=criterion, loss_meter=loss_meter,
                           optimizer=optimizer, cur_epoch=epoch, device=device, cfg=cfg, logger=logger)
