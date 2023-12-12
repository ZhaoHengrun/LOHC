# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import timm

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from utils import *
from models import models_mae_bg
from models.models_mae_w_mask import mae_vit_base_patch4

from mae_engine_pretrain_bg import train_one_epoch
from dataset import *


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='default=64, Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int, help='default=400')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch4', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--mask_ratio', default=0.9, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    # parser.add_argument("--dataset_seg", default='', type=str, help="dataset path")
    parser.add_argument("--dataset_img", default='datasets/train/AHP/img/', type=str, help="dataset path")
    parser.add_argument("--dataset_seg", default='datasets/train/AHP/seg/', type=str, help="dataset path")

    # parser.add_argument('--human_mae_path', type=str, default='checkpoints/mae_human/last_model.pth', help='model file to use')
    parser.add_argument('--output_dir', default='checkpoints/pretrain/', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='checkpoints/pretrain/log/', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=1, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    misc.init_distributed_mode(args)

    make_dir(args.output_dir)
    make_dir(args.log_dir)
    make_dir('{}{}'.format(args.output_dir, 'img/'))
    make_dir('{}{}'.format(args.output_dir, 'code/'))

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    cudnn.benchmark = True

    train_set = TrainDatasetWithMask(img_path=args.dataset_img, seg_path=args.dataset_seg)
    training_data_loader = DataLoader(dataset=train_set, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.num_workers)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # define the model
    model = models_mae_bg.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    model_params = 0
    for param in model.parameters():
        model_params += param.numel()
    print('model_params:', model_params / 1e6)

    model.to(device)

    model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))

    # mae_model_human_checkpoint = torch.load(args.human_mae_path, map_location='cuda:0')
    # mae_model_human = mae_vit_base_patch4()
    # mae_model_human.load_state_dict(mae_model_human_checkpoint['model'])
    # mae_model_human = mae_model_human.cuda()
    # mae_model_human.eval()

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    criterion_pixel = torch.nn.MSELoss().to(device)

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        train_stats = train_one_epoch(
            model, training_data_loader,
            optimizer, device, epoch, loss_scaler, criterion_pixel,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir:
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            # with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            #     f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)