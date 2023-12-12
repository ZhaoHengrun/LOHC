import os
from os import listdir
import cv2 as cv
import torch
from torchvision import transforms
from torchvision import utils as vutils
from utils import *
# from dataset import random_free_form_mask
from util.mask_generator import *

original_img_path = 'datasets/AHP/AHP/train/JPEGImages/'
mask_path = 'datasets/AHP/AHP/train/Annotations/'
parsing_path = 'datasets/AHP/AHP/train/PseudoParsingAnnotations/'
output_path = 'datasets/train/AHP/'


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


def crop_AHP(original_img_path, mask_path, parsing_path, output_path):
    make_dir(output_path)
    output_img_path = '{}img/'.format(output_path)
    output_mask_path = '{}mask/'.format(output_path)
    output_parsing_path = '{}parsing/'.format(output_path)
    make_dir(output_img_path)
    make_dir(output_mask_path)
    make_dir(output_parsing_path)
    for img_name in listdir(mask_path):
        mask_dir = '{}{}'.format(mask_path, img_name)
        parsing_dir = '{}{}'.format(parsing_path, img_name)
        original_dir = '{}{}{}'.format(original_img_path, img_name.split(".")[0], '.jpg')

        img = cv.imread(original_dir)
        mask = cv.imread(mask_dir, 0)
        _, mask = cv.threshold(mask, 0, 255, cv.THRESH_BINARY)
        parsing = cv.imread(parsing_dir)

        h, w, _ = img.shape
        top = h
        buttom = 0
        left = w
        right = 0
        for i in range(h):
            for j in range(w):
                if mask[i, j] == 0:
                    # 按照像素值查找需要剪切图片的起始位置，现该位置像素值为[0,0,0]
                    continue
                else:
                    # 获得剪切起始位置
                    if j <= left:
                        left = j
                    if j >= right:
                        right = j
                    if i >= buttom:
                        buttom = i
                    if i <= top:
                        top = i
        new_h = buttom - top
        new_w = right - left
        if new_h > new_w:
            padding_w = new_h - new_w
            half_padding_w = int(padding_w / 2)
            if left - half_padding_w >= 0:
                left = left - half_padding_w
            else:
                left = 0
            if left + new_h < w:
                right = left + new_h
            else:
                right = w
        if new_w > new_h:
            padding_h = new_w - new_h
            half_padding_h = int(padding_h / 2)
            if top - half_padding_h >= 0:
                top = top - half_padding_h
            else:
                top = 0
            if top + new_w < h:
                buttom = top + new_w
            else:
                buttom = h
        padded_h = buttom - top
        padded_w = right - left
        mask = mask[top:buttom, left:right]
        img = img[top:buttom, left:right]
        parsing = parsing[top:buttom, left:right]

        if padded_h > padded_w:
            mask = cv.copyMakeBorder(mask, 0, 0, 0, int(padded_h - padded_w), cv2.BORDER_REFLECT)
            img = cv.copyMakeBorder(img, 0, 0, 0, int(padded_h - padded_w), cv2.BORDER_REFLECT)
            parsing = cv.copyMakeBorder(parsing, 0, 0, 0, int(padded_h - padded_w), cv2.BORDER_REFLECT)
        if padded_w > padded_h:
            mask = cv.copyMakeBorder(mask, 0, int(padded_w - padded_h), 0, 0, cv2.BORDER_REFLECT)
            img = cv.copyMakeBorder(img, 0, int(padded_w - padded_h), 0, 0, cv2.BORDER_REFLECT)
            parsing = cv.copyMakeBorder(parsing, 0, int(padded_w - padded_h), 0, 0, cv2.BORDER_REFLECT)

        mask = cv.resize(mask, (256, 256))
        img = cv.resize(img, (256, 256))
        parsing = cv.resize(parsing, (256, 256))

        cv.imwrite('{}{}'.format(output_mask_path, img_name), mask)
        cv.imwrite('{}{}'.format(output_img_path, img_name), img)
        cv.imwrite('{}{}'.format(output_parsing_path, img_name), parsing)

        print('saving:[{}]'.format(mask_dir))


if __name__ == "__main__":
    crop_AHP(original_img_path, mask_path, parsing_path, output_path)
