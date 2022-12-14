import json
import pickle

import numpy as np
import torch
import random
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import time
import yaml
from torchvision import transforms


def set_seed(seed=2022):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def load_ymal_conf(sch_conf_file):
    with open(sch_conf_file) as f:
        sch_conf = yaml.load(f, Loader=yaml.FullLoader)
    return sch_conf


def vis_image_from_tensor(tensor):
    transforms.ToPILImage()(tensor).save("test.jpg")


class Logger(object):
    def __init__(self, path_log):
        log_name = os.path.basename(path_log)  # log.log
        self.log_name = log_name if log_name else "root"
        self.out_path = path_log

        log_dir = os.path.dirname(self.out_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def init_logger(self, file, console):
        logger = logging.getLogger(self.log_name)
        logger.setLevel(level=logging.INFO)

        # 配置文件Handler
        file_handler = logging.FileHandler(self.out_path, "w")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        # 配置屏幕Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

        # 添加handler
        if file:
            print("file_hander")
            logger.addHandler(file_handler)
        if console:
            print("console_handler")
            logger.addHandler(console_handler)

        return logger


def check_data_dir(path):
    assert os.path.exists(path), "\n\n路径不存在，当前变量中指定的路径是：\n{}\n请检查相对路径的设置，或者文件是否存在".format(os.path.abspath(path))


def make_dirs(dirs):
    dir, name = os.path.split(dirs)
    # print(dir)
    # assert os.path.isdir(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)


def make_logger(out_dir, file=True, console=True, local_rank=0):
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, "%m-%d_%H-%M")
    log_dir = os.path.join(out_dir, time_str)  # 根据config中的创建时间作为文件夹名
    if local_rank == 0:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    path_log = os.path.join(log_dir, "log.log")
    logger = Logger(path_log)
    logger = logger.init_logger(file=file, console=console)
    return logger, log_dir


def show_confMat(confusion_mat, classes, set_name, out_dir, epoch=999, verbose=False, figsize=None, perc=False):
    """
    混淆矩阵绘制并保存图片
    :param confusion_mat:  nd.array
    :param classes: list or tuple, 类别名称
    :param set_name: str, 数据集名称 train or valid or test?
    :param out_dir:  str, 图片要保存的文件夹
    :param epoch:  int, 第几个epoch
    :param verbose: bool, 是否打印精度信息
    :param perc: bool, 是否采用百分比，图像分割时用，因分类数目过大
    :return:
    """
    cls_num = len(classes)

    # 归一化
    confusion_mat_tmp = confusion_mat.copy()
    for i in range(len(classes)):
        confusion_mat_tmp[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # 设置图像大小
    if cls_num < 10:
        figsize = 6
    elif cls_num >= 100:
        figsize = 30
    else:
        figsize = np.linspace(6, 30, 91)[cls_num - 10]
    plt.figure(figsize=(int(figsize), int(figsize * 1.3)))

    # 获取颜色
    cmap = plt.cm.get_cmap("Greys")  # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.imshow(confusion_mat_tmp, cmap=cmap)
    plt.colorbar(fraction=0.03)

    # 设置文字
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, list(classes), rotation=60)
    plt.yticks(xlocations, list(classes))
    plt.xlabel("Predict label")
    plt.ylabel("True label")
    plt.title("Confusion_Matrix_{}_{}".format(set_name, epoch))

    # 打印数字
    if perc:
        cls_per_nums = confusion_mat.sum(axis=0)
        conf_mat_per = confusion_mat / cls_per_nums
        for i in range(confusion_mat_tmp.shape[0]):
            for j in range(confusion_mat_tmp.shape[1]):
                plt.text(x=j, y=i, s="{:.0%}".format(conf_mat_per[i, j]), va="center", ha="center", color="red", fontsize=10)
    else:
        for i in range(confusion_mat_tmp.shape[0]):
            for j in range(confusion_mat_tmp.shape[1]):
                plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va="center", ha="center", color="red", fontsize=10)
    # 保存
    plt.savefig(os.path.join(out_dir, "Confusion_Matrix_{}.png".format(set_name)))
    plt.close()

    if verbose:
        for i in range(cls_num):
            print(
                "class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%}".format(
                    classes[i],
                    np.sum(confusion_mat[i, :]),
                    confusion_mat[i, i],
                    confusion_mat[i, i] / (1e-9 + np.sum(confusion_mat[i, :])),
                    confusion_mat[i, i] / (1e-9 + np.sum(confusion_mat[:, i])),
                )
            )


def plot_line(x, t_y, v_y, mode, out_dir):
    plt.plot(x, t_y, label="Train")
    plt.plot(x, v_y, label="Valid")
    plt.ylabel(str(mode))
    plt.xlabel("Epoch")

    location = "upper right" if mode == "loss" else "upper left"
    plt.legend(loc=location)
    plt.title("_".join([mode]))
    plt.savefig(os.path.join(out_dir, mode + ".png"))
    plt.close()


def load_class_dict(filename):
    extname = os.path.splitext(filename)[-1].lower()
    with open(filename, "r") as f:
        if extname == ".json":
            class_dict = json.load(f)
        else:
            class_dict = pickle.load(f)
    return class_dict


def save_class_dict(filename, class_dict):
    extname = os.path.splitext(filename)[-1].lower()
    with open(filename, "w") as f:
        if extname == ".json":
            json.dump(class_dict, f, indent=4, separators=(",", ": "))
        else:
            pickle.dump(class_dict, f)


def get_class_dict_info(class_dict, with_print=False, desc=None):
    num_list = [len(val) for val in class_dict.values()]
    num_classes = len(num_list)
    num_examples = sum(num_list)
    max_examples_per_class = max(num_list)
    min_examples_per_class = min(num_list)
    if num_classes == 0:
        avg_examples_per_class = 0
    else:
        avg_examples_per_class = num_examples / num_classes
    info = {
        "num_classes": num_classes,
        "num_examples": num_examples,
        "max_examples_per_class": max_examples_per_class,
        "min_examples_per_class": min_examples_per_class,
        "avg_examples_per_class": avg_examples_per_class,
    }
    if with_print:
        desc = desc or "<unknown>"
        print("{} subject number:    {}".format(desc, info["num_classes"]))
        print("{} example number:    {}".format(desc, info["num_examples"]))
        print("{} max number per-id: {}".format(desc, info["max_examples_per_class"]))
        print("{} min number per-id: {}".format(desc, info["min_examples_per_class"]))
        print("{} avg number per-id: {:.2f}".format(desc, info["avg_examples_per_class"]))
    return info


def print_class_dict_info(class_dict, name="<unknown>"):
    info = get_class_dict_info(class_dict)
    print("{} subject number:    {}".format(name, info["num_classes"]))
    print("{} example number:    {}".format(name, info["num_examples"]))
    print("{} max number per-id: {}".format(name, info["max_examples_per_class"]))
    print("{} min number per-id: {}".format(name, info["min_examples_per_class"]))
    print("{} avg number per-id: {:.2f}".format(name, info["avg_examples_per_class"]))


def Timer(func):
    def wrap(*args, **kwargs):
        begin_time = time.time()
        somethings = func(*args, **kwargs)
        end_times = time.time()
        t = end_times - begin_time
        kwargs["logger"].info(f"The training is consumes {t/60} min for this epoch")
        return somethings

    return wrap


class Log:
    """
    创建一个专门的类用来管理logger 日志信息的记录，比如时间，记录loss等

    """

    def __init__(self) -> None:
        pass

    def timer(self):
        pass

    def info():
        pass
