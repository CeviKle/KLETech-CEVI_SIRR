import os
from os.path import join
import torch
import torch.backends.cudnn as cudnn
import util.util as util
# import data.sirs_dataset as datasets
import data.dataset_sir as datasets
from data.image_folder import read_fns
from engine import Engine
from options.net_options.train_options import TrainOptions
from tools import mutils

opt = TrainOptions().parse()

opt.isTrain = False
cudnn.benchmark = True
opt.no_log = True
opt.display_id = 0
opt.verbose = False
datadir = os.path.join(os.path.expanduser('~'), '/opt/datasets/sirs')

eval_dataset_real = datasets.DSRTestDataset("rdnet/RDNet/single_img_test", enable_transforms=True,
                                            fns=read_fns('data/real_test copy.txt'), if_align=opt.if_align)

eval_dataloader_real = datasets.DataLoader(
    eval_dataset_real, batch_size=2, shuffle=False,
    num_workers=8, pin_memory=True)

engine = Engine(opt, eval_dataset_real)


def set_learning_rate(lr):
    for optimizer in engine.model.optimizers:
        print('[i] set learning rate to {}'.format(lr))
        util.set_opt_param(optimizer, 'lr', lr)

"""Main Loop"""
result_dir = os.path.join('./results', opt.name, mutils.get_formatted_time())
set_learning_rate(opt.lr)
ep=0
while ep<200:
    engine.train(eval_dataloader_real)
    ep=ep+1
    print("epoch: ",ep)