"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import cv2
import os
import numpy as np
import torch
# import models
import cv2 as cv
import torch.utils.data as data
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import utils as vutils
from utils import save_image_tensor, make_dir
import torch
import pyiqa


class ValidDataset(data.Dataset):
    def __init__(self, img_path, gt_path, seg_path, mask_path, test_length=-1):
        self.img_path = img_path
        self.gt_path = gt_path
        self.seg_path = seg_path
        self.mask_path = mask_path

        if test_length != -1:
            self.img_lis = sorted(os.listdir(self.img_path))[:test_length]
            self.gt_lis = sorted(os.listdir(self.gt_path))[:test_length]
            self.seg_lis = sorted(os.listdir(self.seg_path))[:test_length]
            self.mask_lis = sorted(os.listdir(self.mask_path))[:test_length]

        else:
            self.img_lis = sorted(os.listdir(self.img_path))
            self.gt_lis = sorted(os.listdir(self.gt_path))
            self.seg_lis = sorted(os.listdir(self.seg_path))
            self.mask_lis = sorted(os.listdir(self.mask_path))

    def __getitem__(self, index):
        img_name = self.img_lis[index]
        gt_name = self.gt_lis[index]
        seg_name = self.seg_lis[index]
        mask_name = self.mask_lis[index]
        full_img_path = '{}{}'.format(self.img_path, img_name)
        full_gt_path = '{}{}'.format(self.gt_path, gt_name)
        full_seg_path = '{}{}'.format(self.seg_path, seg_name)
        full_mask_path = '{}{}'.format(self.mask_path, mask_name)
        img = cv.imread(full_img_path)
        gt = cv.imread(full_gt_path)
        seg = cv.imread(full_seg_path, -1)
        _, seg = cv.threshold(seg, 0, 255, cv.THRESH_BINARY)

        mask = cv.imread(full_mask_path, 0)
        _, mask = cv.threshold(mask, 0, 255, cv.THRESH_BINARY)

        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        gt = cv.cvtColor(gt, cv.COLOR_BGR2RGB)
        img = transforms.ToTensor()(img.copy())
        gt = transforms.ToTensor()(gt.copy())
        seg = transforms.ToTensor()(seg.copy())
        mask = transforms.ToTensor()(mask.copy())
        data = {'image': img, 'gt': gt, 'seg': seg, 'mask': mask, 'img_name': img_name}
        return data

    def __len__(self):
        return len(self.img_lis)


def main():
    input_folders = [
                     'TFill-main/results/',
                     'co-mod-gan-pytorch-train/results/',
                     'crfill-master/results/',
                     'PBHC-main/results/'
                     ]
    # center_mask, rectangle_mask, voc_mask, xpie_mask, hku_is_mask
    dataset_test_gt = 'datasets/test/AHP/img/'
    dataset_seg = 'datasets/test/AHP/seg/'

    save_path = 'output/human_part/'
    make_dir(save_path)

    # mask_types = ['center_mask', 'rectangle_mask', 'voc_mask', 'xpie_mask', 'hku_is_mask']
    mask_types = ['center_mask', 'rectangle_mask', 'voc_mask', 'xpie_mask', 'hku_is_mask']

    for input_folder in input_folders:
        for mask_type in mask_types:
            dataset_mask_path = 'datasets/test/AHP/{}/'.format(mask_type)
            dataset_test_input = '{}{}/'.format(input_folder, mask_type)
            if input_folder == 'output/hiin/':
                dataset_test_input = '{}{}'.format(dataset_test_input, 'complete_fake_img/')

            save_path = '{}human_part/'.format(input_folder)
            make_dir(save_path)
            save_path = '{}{}/'.format(save_path, mask_type)
            make_dir(save_path)

            save_path_gt = '{}{}/'.format(save_path, 'gt')
            save_path_pred = '{}{}/'.format(save_path, 'pred')
            save_path_masked = '{}{}/'.format(save_path, 'masked')

            make_dir(save_path_gt)
            make_dir(save_path_pred)
            make_dir(save_path_masked)

            test_set = ValidDataset(dataset_test_input, dataset_test_gt, dataset_seg, dataset_mask_path, -1)
            dataloader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

            print('datasets:{}'.format(dataset_test_input))
            print(len(dataloader), 'test images')

            lpips_sum = 0
            ssim_sum = 0
            fid_sum = 0
            psnr_sum = 0

            lpips_metric = pyiqa.create_metric('lpips').cuda()
            ssim_metric = pyiqa.create_metric('ssim').cuda()
            psnr_metric = pyiqa.create_metric('psnr').cuda()
            fid_metric = pyiqa.create_metric('fid').cuda()

            for iteration, batch in enumerate(dataloader):
                img, gt, seg, mask, input_name_list = batch['image'], batch['gt'], batch['seg'], batch['mask'], batch[
                    'img_name']
                input_name = '{}.png'.format(input_name_list[0].split('.')[0])
                print('processing:[{}/{}][{}]'.format(iteration, len(dataloader), input_name))

                human_gt = gt * seg
                human_pred = img * seg
                masked_gt = gt * (1 - mask)
                masked_human = masked_gt * seg

                save_image_tensor(human_gt, '{}{}'.format(save_path_gt, input_name))
                save_image_tensor(human_pred, '{}{}'.format(save_path_pred, input_name))
                save_image_tensor(masked_human, '{}{}'.format(save_path_masked, input_name))

                lpips_score = lpips_metric(human_pred, human_gt)
                ssim_score = ssim_metric(human_pred, human_gt)
                psnr_score = psnr_metric(human_pred, human_gt)
                lpips_sum += lpips_score
                ssim_sum += ssim_score
                psnr_sum += psnr_score

            lpips_avr = lpips_sum / len(dataloader)
            ssim_avr = ssim_sum / len(dataloader)
            psnr_avr = psnr_sum / len(dataloader)
            print('{}, lpips:[{}]'.format(mask_type, lpips_avr))
            print('{}, ssim:[{}]'.format(mask_type, ssim_avr))
            print('{}, psnr:[{}]'.format(mask_type, psnr_avr))
            fid_avr = fid_metric(save_path_pred, save_path_gt)
            print('{}, fid:[{}]'.format(mask_type, fid_avr))
            with open("output/iqa_results.txt", "a") as f:
                f.write(
                    '{}, {}:\n'.format(input_folder, mask_type))
                f.write('lpips:[{}], ssim:[{}], psnr:[{}], fid:[{}]\n'.format(lpips_avr, ssim_avr, psnr_avr, fid_avr))
    print('finish')


if __name__ == "__main__":
    main()
