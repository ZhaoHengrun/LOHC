import glob
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import os
from os import listdir
import random
from random import randint

import cv2 as cv
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
import time
from utils import *
from torchvision import utils as vutils
import albumentations as A
from util.mask_generator import gen_mask, gen_mask_with_local_mask


def random_free_form_mask(h, w, mv=10, ma=4.0, ml=80, mbw=20):  # training: mv:10, ma:4.0, ml:80, mbw:20
    """Generate a random free form mask with configuration. default: img_shape:[256,256], mv:5, ma:4.0, ml:40, mbw:10
    Args:
        config: Config should have configuration including IMG_SHAPES,
            VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
    Returns:
        tuple: (top, left, height, width)
        :param w:
        :param h:
    """

    mask = np.zeros((h, w))
    num_v = 12 + np.random.randint(mv)  # tf.random_uniform([], minval=0, maxval=config.MAXVERTEX, dtype=tf.int32)

    for i in range(num_v):
        start_x = np.random.randint(w)
        start_y = np.random.randint(h)
        for j in range(1 + np.random.randint(mv)):
            angle = 0.01 + np.random.randint(ma)
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            length = 10 + np.random.randint(ml)
            brush_w = 10 + np.random.randint(mbw)
            end_x = (start_x + length * np.sin(angle)).astype(np.int32)
            end_y = (start_y + length * np.cos(angle)).astype(np.int32)

            cv.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
            start_x, start_y = end_x, end_y

    return mask.reshape(mask.shape + (1,)).astype(np.float32)


def resizeAndPad(img, size, padColor=0, border_type=4):  # BORDER_CONSTANT = 0, BORDER_REPLICATE = 1, BORDER_WRAP = 3, BORDER_REFLECT_101 = 4
    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw:  # shrinking image
        interp = cv.INTER_AREA
    else:  # stretching image
        interp = cv.INTER_CUBIC

    # aspect ratio of image
    aspect = w / h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1:  # horizontal image
        new_w = sw
        new_h = np.round(new_w / aspect).astype(int)
        pad_vert = (sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:  # vertical image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = (sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:  # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, np.ndarray)):  # color image but only one color provided
        padColor = [padColor] * 3

    # scale and pad
    scaled_img = cv.resize(img, (new_w, new_h), interpolation=interp)
    if border_type == 0:
        scaled_img = cv.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv.BORDER_CONSTANT, value=padColor)
    else:
        scaled_img = cv.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=border_type)

    return scaled_img


class MultiDataExpasion(object):
    def __call__(self, img_input):
        hflip = random.random() < 0.5
        # vflip = random.random() < 0.5
        # rot90 = random.random() < 0.5
        output_list = []

        for img in img_input:
            if len(img.shape) == 2:
                img = img[:, :, None]
                unsqueeze_flag = True
            else:
                unsqueeze_flag = False
            if hflip:
                img = img[:, ::-1, :]

            # if vflip:
            #     img = img[::-1, :, :]

            # if rot90:
            #     img = img.transpose(1, 0, 2)
            if unsqueeze_flag is True:
                img = img[:, :]

            output_list.append(img)
        return output_list


class TrainDatasetWithMask(data.Dataset):
    def __init__(self, seg_path, img_path):
        self.seg_path = seg_path
        self.img_path = img_path
        self.img_lis = sorted(os.listdir(self.img_path))
        self.seg_lis = sorted(os.listdir(self.seg_path))
        self.crop_size = 256
        self.transform = A.Compose([
            A.CLAHE(p=0.1),
            A.IAASharpen(p=0.1),
            A.IAAEmboss(p=0.1),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.1),
            A.HueSaturationValue(p=0.1),
            A.IAAAdditiveGaussianNoise(p=0.1),
            A.GaussNoise(p=0.1),
            A.Blur(p=0.1),
            A.MedianBlur(p=0.1),
        ])

    def __getitem__(self, index):
        full_img_path = '{}{}'.format(self.img_path, self.img_lis[index])
        full_seg_path = '{}{}'.format(self.seg_path, self.seg_lis[index])
        img = cv.imread(full_img_path)
        seg = cv.imread(full_seg_path, -1)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # cv.imwrite('seg.png', seg)
        _, seg = cv.threshold(seg, 0, 255, cv.THRESH_BINARY)
        # img = resizeAndPad(img, [self.crop_size, self.crop_size], border_type=randint(1, 4))
        # seg = resizeAndPad(seg, [self.crop_size, self.crop_size], border_type=0)
        twin_trans = self.transform(image=img, mask=seg)
        img, seg = twin_trans['image'], twin_trans['mask']
        img = transforms.ToTensor()(img.copy())
        seg = transforms.ToTensor()(seg.copy())
        mask = gen_mask(self.crop_size, self.crop_size)
        # save_image_tensor(img.unsqueeze(0), 'img.png')
        # save_image_tensor(seg.unsqueeze(0), 'seg_bin.png')
        return img, seg, mask

    def __len__(self):
        return len(self.img_lis)


class TrainDatasetWithLocalMask(data.Dataset):
    def __init__(self, seg_path, img_path):
        self.seg_path = seg_path
        self.img_path = img_path
        self.mask_path = 'datasets/train/soc6k_masks/'
        self.img_lis = sorted(os.listdir(self.img_path))
        self.seg_lis = sorted(os.listdir(self.seg_path))
        self.mask_lis = sorted(os.listdir(self.mask_path))
        self.crop_size = 256
        self.transform = A.Compose([
            A.CLAHE(p=0.1),
            A.IAASharpen(p=0.1),
            A.IAAEmboss(p=0.1),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.1),
            A.HueSaturationValue(p=0.1),
            A.IAAAdditiveGaussianNoise(p=0.1),
            A.GaussNoise(p=0.1),
            A.Blur(p=0.1),
            A.MedianBlur(p=0.1),
        ])

    def __getitem__(self, index):
        mask_index = randint(0, len(self.mask_lis) - 1)
        full_img_path = '{}{}'.format(self.img_path, self.img_lis[index])
        full_seg_path = '{}{}'.format(self.seg_path, self.seg_lis[index])
        full_mask_path = '{}{}'.format(self.mask_path, self.mask_lis[mask_index])
        img = cv.imread(full_img_path)
        seg = cv.imread(full_seg_path, -1)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        mask = cv.imread(full_mask_path, 0)

        # cv.imwrite('seg.png', seg)
        _, seg = cv.threshold(seg, 0, 255, cv.THRESH_BINARY)
        _, mask = cv.threshold(mask, 0, 255, cv.THRESH_BINARY)
        # img = resizeAndPad(img, [self.crop_size, self.crop_size], border_type=randint(1, 4))
        # seg = resizeAndPad(seg, [self.crop_size, self.crop_size], border_type=0)
        twin_trans = self.transform(image=img, mask=seg)
        img, seg = twin_trans['image'], twin_trans['mask']
        img = transforms.ToTensor()(img.copy())
        seg = transforms.ToTensor()(seg.copy())
        mask = gen_mask_with_local_mask(self.crop_size, self.crop_size, mask)
        # save_image_tensor(img.unsqueeze(0), 'img.png')
        # save_image_tensor(seg.unsqueeze(0), 'seg_bin.png')
        return img, seg, mask

    def __len__(self):
        return len(self.img_lis)


class ValidDatasetWithMask(data.Dataset):
    def __init__(self, img_path, mask_path, test_length=-1):
        self.img_path = img_path
        self.mask_path = mask_path
        if test_length != -1:
            self.img_lis = sorted(os.listdir(self.img_path))[:test_length]
            self.mask_lis = sorted(os.listdir(self.mask_path))[:test_length]
        else:
            self.img_lis = sorted(os.listdir(self.img_path))
            self.mask_lis = sorted(os.listdir(self.mask_path))

    def __getitem__(self, index):
        img_name = self.img_lis[index]
        mask_name = self.mask_lis[index]
        full_img_path = '{}{}'.format(self.img_path, img_name)
        full_mask_path = '{}{}'.format(self.mask_path, mask_name)
        img = cv.imread(full_img_path)
        mask = cv.imread(full_mask_path, 0)
        _, mask = cv.threshold(mask, 0, 255, cv.THRESH_BINARY)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = transforms.ToTensor()(img.copy())
        mask = transforms.ToTensor()(mask.copy())
        return img, mask, img_name

    def __len__(self):
        return len(self.img_lis)


class TrainVideoDatasetWithMask(data.Dataset):
    def __init__(self, img_path):
        self.img_path = img_path
        self.folder_list = sorted(listdir(self.img_path))
        self.img_list = []
        self.seg_list = []
        for f in self.folder_list:
            img_folder_path = '{}{}/images/'.format(self.img_path, f)
            self.img_list.append(sorted(listdir(img_folder_path)))
        for f in self.folder_list:
            seg_folder_path = '{}{}/masks/'.format(self.img_path, f)
            self.seg_list.append(sorted(listdir(seg_folder_path)))

        self.crop_size = 256
        self.transform = A.Compose([
            A.CLAHE(p=0.1),
            A.IAASharpen(p=0.1),
            A.IAAEmboss(p=0.1),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.HueSaturationValue(p=0.2),
            A.IAAAdditiveGaussianNoise(p=0.1),
            A.GaussNoise(p=0.1),
            A.Blur(p=0.1),
            A.MedianBlur(p=0.1),
        ])

    def __getitem__(self, idx):
        folder_index = randint(0, (len(self.folder_list) - 1))
        img_index = randint(0, len(self.img_list[folder_index]) - 1)
        full_img_path = '{}{}/images/{}'.format(self.img_path, self.folder_list[folder_index], self.img_list[folder_index][img_index])
        full_seg_path = '{}{}/masks/{}'.format(self.img_path, self.folder_list[folder_index], self.seg_list[folder_index][img_index])
        img = cv.imread(full_img_path)
        seg = cv.imread(full_seg_path, -1)

        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        _, seg = cv.threshold(seg, 0, 255, cv.THRESH_BINARY)

        h, w, _ = img.shape  # 1080, 604
        new_h = self.crop_size  # 256
        new_w = int(w * (self.crop_size / h))  # 143
        img = cv.resize(img, (new_w, new_h))
        seg = cv.resize(seg, (new_w, new_h))
        # img = resizeAndPad(img, [self.crop_size, self.crop_size], border_type=randint(1, 4))
        # seg = resizeAndPad(seg, [self.crop_size, self.crop_size], border_type=0)
        twin_trans = self.transform(image=img, mask=seg)
        img, seg = twin_trans['image'], twin_trans['mask']
        img = transforms.ToTensor()(img.copy())
        seg = transforms.ToTensor()(seg.copy())
        mask = gen_mask(self.crop_size, self.crop_size)
        padding_left = int((self.crop_size - new_w) / 2)
        mask = mask[:, :, padding_left:padding_left + new_w]
        return img, seg, mask

    def __len__(self):
        # return len(self.folder_list)
        return 82340  # real 82340
