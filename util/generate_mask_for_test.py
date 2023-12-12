import os
from os import listdir
import cv2 as cv
import torch
from torchvision import transforms
from torchvision import utils as vutils
from utils import *
from dataset import random_free_form_mask
from util.mask_generator import *

original_img_path = 'datasets/test/AHP/img/'
mask_type = 'deepfill_mask'  # center_mask, random_regular_mask, random_irregular_mask, deepfill_mask

mask_path = 'datasets/test/AHP/deepfill_mask/'


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def gen_mask(h, w, mask_type):
    """load the mask for image completion task"""
    # mask_type = [0, 1, 1, 2, 3]
    # mask_type_index = random.randint(0, len(mask_type) - 1)
    # mask_type = mask_type[mask_type_index]

    if mask_type == 'center_mask':  # center mask
        return center_mask(h, w)
    elif mask_type == 'random_regular_mask':  # random regular mask
        return random_regular_mask(h, w)
    elif mask_type == 'random_irregular_mask':  # random irregular mask
        return random_irregular_mask(h, w)
    elif mask_type == 'deepfill_mask':  # deepfill_mask
        return deepfill_mask(h, w, mv=10, ma=4.0, ml=80, mbw=20)  # training: mv:10, ma:4.0, ml:80, mbw:20


def make_mask_single_folder(original_img_path, mask_path, mask_type):
    make_dir(mask_path)
    for i in listdir(original_img_path):
        if is_image_file(i):
            original_dir = '{}{}'.format(original_img_path, i)
            mask_dir = '{}{}{}'.format(mask_path, i.split(".")[0], '.png')

            # img = cv.imread(original_dir)
            # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            # img = transforms.ToTensor()(img.copy()).unsqueeze(0)
            # img, pad_top, pad_left, new_h, new_w, h, w, resize_flag = resize(img, [256, 256])
            # img = img.squeeze(0)

            mask = gen_mask(256, 256, mask_type)

            # if resize_flag is True:
            #     mask = mask.unsqueeze(0)
            #     mask = resize_back(mask, pad_top, pad_left, new_h, new_w, h, w)
            #     mask = mask.squeeze(0)
            vutils.save_image(mask, mask_dir)
            print('saving:[{}]'.format(mask_dir))


def make_mask_video(original_img_path, mask_type):
    folder_list = sorted(listdir(original_img_path))
    for sub_folder in folder_list:
        sub_folder_path = '{}{}/images/'.format(original_img_path, sub_folder)
        sub_mask_path = '{}{}/{}/'.format(original_img_path, sub_folder, mask_type)
        make_dir(sub_mask_path)
        sub_folder_list = sorted(listdir(sub_folder_path))
        for i in sub_folder_list:
            if is_image_file(i):
                original_dir = '{}{}'.format(sub_folder_path, i)
                mask_dir = '{}{}{}'.format(sub_mask_path, i.split(".")[0], '.png')

                img = cv.imread(original_dir)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                img = transforms.ToTensor()(img.copy()).unsqueeze(0)
                img, pad_top, pad_left, new_h, new_w, h, w, resize_flag = resize(img, [256, 256])
                # img = img.squeeze(0)

                mask = gen_mask(256, 256, mask_type)

                if resize_flag is True:
                    mask = mask.unsqueeze(0)
                    mask = resize_back(mask, pad_top, pad_left, new_h, new_w, h, w)
                    mask = mask.squeeze(0)
                vutils.save_image(mask, mask_dir)
                print('saving:[{}]'.format(mask_dir))


if __name__ == "__main__":
    make_mask_single_folder(original_img_path, mask_path, mask_type)
