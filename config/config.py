from easydict import EasyDict
cfg = EasyDict()
cfg.image_root = "../../widerface/train/images"
cfg.label_path = "../../widerface/train/label.txt"
cfg.exp_log = './log'
cfg.target_height = 800
cfg.target_width = 800
cfg.factor = 0.1
cfg.milestones = [5,10]
cfg.end_epoch = 100
cfg.num_workers = 8
cfg.pretrained =None
cfg.print_freq = 100
cfg.save_freq = 1500
cfg.out_dir = "./log"