# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from utils import make_dir, save_image_tensor, resizeAndPadTensor, resize, resize_back
import torch.nn.functional as F


def run_mae_inference(model_mae, images_masked, mask):
    images_masked, pad_top, pad_left, new_h, new_w, h, w, resize_flag = resize(images_masked, [64, 64])
    mask, _, _, _, _, _, _, _ = resize(mask, [64, 64])
    with torch.no_grad():
        reconstruction_loss, pred, mask = model_mae(images_masked, mask, threshold=0)
        # mae_preds = images_masked + pred * mask
        mae_preds = pred
        if resize_flag is True:
            mae_preds = resize_back(mae_preds, pad_top, pad_left, new_h, new_w, h, w)
    return mae_preds


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, criterion_pixel,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    epoch_loss = 0
    for data_iter_step, (img, seg, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        img = img.to(device, non_blocking=True)
        seg = seg.to(device, non_blocking=True)

        # img = F.interpolate(img, size=(256, 256), mode='bilinear')
        seg = F.interpolate(seg, size=(64, 64), mode='bilinear')
        img = F.interpolate(img, size=(64, 64), mode='bilinear')
        human = img * seg

        img_with_seg = torch.cat([img, seg], dim=1)
        human_with_seg = torch.cat([human, seg], dim=1)

        with torch.cuda.amp.autocast():
            # loss, pred, _ = model(img, human, mae_model_human, mask_ratio=args.mask_ratio)
            pred, _ = model(img_with_seg, human_with_seg, mask_ratio=args.mask_ratio)

        pred_seg = pred[:, -1, :, :].unsqueeze(1)
        pred_img = pred[:, :-1, :, :]
        pred_human = pred_img * pred_seg
        img_loss = criterion_pixel(pred_img, img)
        human_loss = criterion_pixel(pred_human, human)
        seg_loss = criterion_pixel(pred_seg, seg)
        loss = img_loss + human_loss + seg_loss
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

        if data_iter_step % 500 == 0:
            save_image_tensor(img[0, :, :, :].unsqueeze(0), '{}img/epoch{}_img.png'.format(args.output_dir, epoch))
            # save_image_tensor(seg[0, :, :, :].unsqueeze(0), '{}img/epoch{}_seg.png'.format(args.output_dir, epoch))
            save_image_tensor(pred_img[0, :, :, :].unsqueeze(0), '{}img/epoch{}_pred.png'.format(args.output_dir, epoch))
            save_image_tensor(pred_seg[0, :, :, :].unsqueeze(0), '{}img/epoch{}_pred_seg.png'.format(args.output_dir, epoch))
            save_image_tensor(pred_human[0, :, :, :].unsqueeze(0), '{}img/epoch{}_pred_human.png'.format(args.output_dir, epoch))
        epoch_loss += loss_value
    epoch_loss = epoch_loss / len(data_loader)
    with open("{}log.txt".format(args.output_dir), "a") as f:
        f.write('E[{}], loss[{:.4f}], lr[{:.6f}]\n'.format(epoch, epoch_loss, optimizer.param_groups[0]["lr"]))

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
