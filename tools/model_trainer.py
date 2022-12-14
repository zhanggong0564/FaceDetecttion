import torch
import numpy as np
from tqdm import tqdm
from collections import Counter
from torch.nn import functional as F

# from tools.mixup import mixup_data, mixup_criterion
from tools.utils import Timer
import os


class ModelTrainer(object):
    @staticmethod
    def get_lr(optimizer):
        """Get the current learning rate from optimizer."""
        for param_group in optimizer.param_groups:
            return param_group["lr"]

    @staticmethod
    def fix_bn(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            m.eval()

    @staticmethod
    def load_model(model, model_path):
        weights = torch.load(model_path, map_location="cpu")["state_dict"]
        model_dict = model.state_dict()
        for k in model_dict:
            if k not in weights or model_dict[k].shape!=weights[k].shape :
                print(f"{k} is miss match {model_dict[k].shape}-{weights[k].shape} ")
            else:
                model_dict[k] = weights[k]
        # weights = {k: v for k, v in weights.items() if k in model_dict and model_dict[k].shape == v.shape}
        # model_dict.update(weights)
        model.load_state_dict(model_dict)
        print("load model finish")
        return model

    @staticmethod
    @Timer
    def train(data_loader, model, criterion, loss_meter, optimizer, cur_epoch, device, cfg, logger, ids=0):
        model.train()
        data_loader = tqdm(data_loader)
        label_list = []
        data_loader.set_description(f"Epoch{cur_epoch + 1}")
        for batch_idx, (images, labels, _) in enumerate(data_loader):
            label_list.extend(labels.tolist())  # 记录真实标签
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.squeeze()
            outputs = model.forward(images, labels)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), images.shape[0])

            lr = ModelTrainer.get_lr(optimizer)

            data_loader.set_postfix(lr=lr, loss=loss.item())

            if batch_idx % cfg.print_freq == 0 and batch_idx != 0 and ids == 0:
                loss_avg = loss_meter.avg
                logger.info("Epoch %d, iter %d/%d, lr %f, loss %f" % (cur_epoch, batch_idx, len(data_loader), lr, loss_avg))
                loss_meter.reset()
            if (batch_idx + 1) % cfg.save_freq == 0 and batch_idx != 0 and ids == 0:
                saved_name = "Epoch_%d_batch_%d.pt" % (cur_epoch, batch_idx)
                state = {
                    "state_dict": model.module.state_dict(),
                    # "state_dict": model.state_dict(),
                    "epoch": cur_epoch,
                }
                torch.save(state, os.path.join(cfg.out_dir, saved_name))
                logger.info("Save checkpoint %s to disk." % saved_name)
        if ids == 0:
            saved_name = "Epoch_%d.pt" % cur_epoch
            state = {
                "state_dict": model.module.state_dict(),
                # "state_dict": model.state_dict(),
                "epoch": cur_epoch,
                "optimizer": optimizer.state_dict(),
            }
            torch.save(state, os.path.join(cfg.out_dir, saved_name))
            logger.info("Save checkpoint %s to disk..." % saved_name)
        # return model

    @staticmethod
    def distill_train(data_loader, student_model, teacher_model, criterion, loss_meter, optimizer, cur_epoch, device, cfg, logger):
        student_model.train()
        teacher_model.eval()
        data_loader = tqdm(data_loader)
        data_loader.set_description(f"Epoch{cur_epoch}")
        for batch_idx, (images, labels, _) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.squeeze()
            feat_student = student_model.forward(images,labels)
            with torch.no_grad():
                feat_teacher = teacher_model.forward(images)

            loss = criterion(feat_student, feat_teacher)
            # loss3 = torch.mean(torch.sum((F.normalize(feat_teacher) - F.normalize(feat_student)) ** 2, dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), images.shape[0])
            loss_avg = loss_meter.avg
            lr = ModelTrainer.get_lr(optimizer)
            data_loader.set_postfix(lr=lr, loss=loss_avg)
            if batch_idx % cfg.print_freq == 0 and batch_idx != 0:
                logger.info("Epoch %d, iter %d/%d, lr %f, loss %f" % (cur_epoch, batch_idx, len(data_loader), lr, loss_avg))
                loss_meter.reset()
            if (batch_idx + 1) % cfg.save_freq == 0:
                saved_name = "Epoch_%d_batch_%d.pt" % (cur_epoch, batch_idx)
                state = {"state_dict": student_model.module.state_dict(), "epoch": cur_epoch, "batch_id": batch_idx}
                torch.save(state, os.path.join(cfg.out_dir, saved_name))
                logger.info("Save checkpoint %s to disk." % saved_name)
            loss_meter.reset()
        saved_name = "Epoch_%d.pt" % cur_epoch
        state = {"state_dict": student_model.module.state_dict(), "epoch": cur_epoch, "optimizer": optimizer.state_dict()}
        torch.save(state, os.path.join(cfg.out_dir, saved_name))
        logger.info("Save checkpoint %s to disk..." % saved_name)
