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
import torch
import pyiqa


class ValidDataset(data.Dataset):
    def __init__(self, img_path, gt_path, test_length=-1):
        self.img_path = img_path
        self.gt_path = gt_path
        if test_length != -1:
            self.img_lis = sorted(os.listdir(self.img_path))[:test_length]
            self.gt_lis = sorted(os.listdir(self.gt_path))[:test_length]
        else:
            self.img_lis = sorted(os.listdir(self.img_path))
            self.gt_lis = sorted(os.listdir(self.gt_path))

    def __getitem__(self, index):
        img_name = self.img_lis[index]
        gt_name = self.gt_lis[index]
        full_img_path = '{}{}'.format(self.img_path, img_name)
        full_gt_path = '{}{}'.format(self.gt_path, gt_name)
        img = cv.imread(full_img_path)
        gt = cv.imread(full_gt_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        gt = cv.cvtColor(gt, cv.COLOR_BGR2RGB)
        img = transforms.ToTensor()(img.copy())
        gt = transforms.ToTensor()(gt.copy())
        data = {'image': img, 'gt': gt, 'img_name': img_name}
        return data

    def __len__(self):
        return len(self.img_lis)


def main():
    dataset_test_input = 'voc_mask/complete_fake_img/'  # center_mask, rectangle_mask, voc_mask, xpie_mask, hku_is_mask
    dataset_test_gt = 'voc_mask/input/'

    # mask_types = ['center_mask', 'rectangle_mask', 'voc_mask', 'xpie_mask', 'hku_is_mask']
    mask_types = ['voc_mask']

    for mask_type in mask_types:
        # dataset_test_input = '/results/{}/'.format(mask_type)

        test_set = ValidDataset(dataset_test_input, dataset_test_gt, -1)
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
            img, gt, input_name_list = batch['image'], batch['gt'], batch['img_name']
            input_name = '{}.png'.format(input_name_list[0].split('.')[0])
            print('processing:[{}/{}][{}]'.format(iteration, len(dataloader), input_name))

            lpips_score = lpips_metric(img, gt)
            ssim_score = ssim_metric(img, gt)
            psnr_score = psnr_metric(img, gt)
            lpips_sum += lpips_score
            ssim_sum += ssim_score
            psnr_sum += psnr_score

        lpips_avr = lpips_sum / len(dataloader)
        ssim_avr = ssim_sum / len(dataloader)
        psnr_avr = psnr_sum / len(dataloader)
        print('{}, lpips:[{}]'.format(mask_type, lpips_avr))
        print('{}, ssim:[{}]'.format(mask_type, ssim_avr))
        print('{}, psnr:[{}]'.format(mask_type, psnr_avr))
        fid_avr = fid_metric(dataset_test_input, dataset_test_gt)
        print('{}, fid:[{}]'.format(mask_type, fid_avr))
        with open("output/iqa_results.txt", "a") as f:
            f.write(
                '{}, lpips:[{}], ssim:[{}], psnr:[{}], fid:[{}]'.format(mask_type, lpips_avr, ssim_avr, psnr_avr,
                                                                        fid_avr))
    print('finish')


if __name__ == "__main__":
    main()
