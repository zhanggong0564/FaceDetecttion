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
            if k not in weights or model_dict[k].shape != weights[k].shape:
                print(f"{k} is miss match {model_dict[k].shape}-{weights[k].shape} ")
            else:
                model_dict[k] = weights[k]
        # weights = {k: v for k, v in weights.items() if k in model_dict and model_dict[k].shape == v.shape}
        # model_dict.update(weights)
        model.load_state_dict(model_dict)
        print("load model finish")
        return model

    @staticmethod
    # @Timer
    def train(data_loader, model, criterion, loss_meter, optimizer, cur_epoch, device, cfg, logger, ids=0):
        model.train()
        data_loader = tqdm(data_loader)
        label_list = []
        data_loader.set_description(f"Epoch{cur_epoch + 1}")
        for batch_idx, inputs in enumerate(data_loader):
            images, _, _, _, _ = inputs
            # label_list.extend(labels.tolist())  # 记录真实标签
            images = images.to(device)
            # labels = labels.to(device)
            outputs = model(images)
            loss, loss1, loss2, loss3 = criterion(outputs, inputs)
            if loss.item() == np.nan:
                print(loss1.item())
                print(loss2.item())
                print(loss3.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), images.shape[0])

            lr = ModelTrainer.get_lr(optimizer)

            data_loader.set_postfix(lr=lr, loss=loss.item(), point_loss=loss1.item(), offset_loss=loss2.item(),
                                    wh_loss=loss3.item())

            if batch_idx % cfg.print_freq == 0 and batch_idx != 0 and ids == 0:
                loss_avg = loss_meter.avg
                logger.info(
                    "Epoch %d, iter %d/%d, lr %f, loss %f" % (cur_epoch, batch_idx, len(data_loader), lr, loss_avg))
                loss_meter.reset()
            if (batch_idx + 1) % cfg.save_freq == 0 and batch_idx != 0 and ids == 0:
                saved_name = "Epoch_%d_batch_%d.pt" % (cur_epoch, batch_idx)
                state = {
                    "state_dict": model.state_dict(),
                    # "state_dict": model.state_dict(),
                    "epoch": cur_epoch,
                }
                torch.save(state, os.path.join(cfg.exp_log, saved_name))
                logger.info("Save checkpoint %s to disk." % saved_name)
        if ids == 0:
            saved_name = "Epoch_%d.pt" % cur_epoch
            state = {
                "state_dict": model.state_dict(),
                # "state_dict": model.state_dict(),
                "epoch": cur_epoch,
                "optimizer": optimizer.state_dict(),
            }
            torch.save(state, os.path.join(cfg.exp_log, saved_name))
            logger.info("Save checkpoint %s to disk..." % saved_name)
        # return model

    @staticmethod
    def distill_train(data_loader, student_model, teacher_model, criterion, loss_meter, optimizer, cur_epoch, device,
                      cfg, logger):
        student_model.train()
        teacher_model.eval()
        data_loader = tqdm(data_loader)
        data_loader.set_description(f"Epoch{cur_epoch}")
        for batch_idx, (images, labels, _) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.squeeze()
            feat_student = student_model.forward(images, labels)
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
                logger.info(
                    "Epoch %d, iter %d/%d, lr %f, loss %f" % (cur_epoch, batch_idx, len(data_loader), lr, loss_avg))
                loss_meter.reset()
            if (batch_idx + 1) % cfg.save_freq == 0:
                saved_name = "Epoch_%d_batch_%d.pt" % (cur_epoch, batch_idx)
                state = {"state_dict": student_model.module.state_dict(), "epoch": cur_epoch, "batch_id": batch_idx}
                torch.save(state, os.path.join(cfg.out_dir, saved_name))
                logger.info("Save checkpoint %s to disk." % saved_name)
            loss_meter.reset()
        saved_name = "Epoch_%d.pt" % cur_epoch
        state = {"state_dict": student_model.module.state_dict(), "epoch": cur_epoch,
                 "optimizer": optimizer.state_dict()}
        torch.save(state, os.path.join(cfg.out_dir, saved_name))
        logger.info("Save checkpoint %s to disk..." % saved_name)
    @torch.no_grad()
    def vaild(data_loader, model, device, cfg, logger, ids=0):
        model.train()
        data_loader = tqdm(data_loader)
        label_list = []
        data_loader.set_description(f"evaluation: ")
        for batch_idx, inputs in enumerate(data_loader):
            images, _, _, _, _ = inputs
            # label_list.extend(labels.tolist())  # 记录真实标签
            images = images.to(device)
            # labels = labels.to(device)
            outputs = model(images)

            predict_point = outputs[0].sigmoid() #n 1 200 200
            predict_wh = outputs[2] #n 2 200 200
            predict_offset = outputs[1] #n 2 200 200
            pool_predict_point = F.max_pool2d(predict_point, kernel_size=3, padding=1, stride=1) #n 1 200 200
            # plt.imshow(pool_predict_point.data.numpy()[0, 0])

            threshold = 0.5
            select_nms_mask = pool_predict_point.eq(predict_point) & (predict_point > threshold) #n 1 200 200
            # plt.imshow(select_nms_mask.data.numpy()[0, 0])

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
