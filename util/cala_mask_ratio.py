import os
from os import listdir
import cv2 as cv
import torch
from torchvision import transforms

# from torchvision import utils as vutils
# from utils import *
# from dataset import random_free_form_mask
# from util.mask_generator import *

# center_mask, rectangle_mask, voc_mask, xpie_mask, hku_is_mask
mask_path = '../datasets/test/AHP/hku_is_mask/'
seg_path = '../datasets/test/AHP/seg/'


def calc_ratio(mask_path, seg_path):
    rate_white_sum = 0
    count = 0
    for i in listdir(mask_path):
        count += 1
        mask_dir = '{}{}{}'.format(mask_path, i.split(".")[0], '.png')
        seg_dir = '{}{}{}'.format(seg_path, i.split(".")[0], '.png')

        mask = cv.imread(mask_dir, 0)
        _, mask = cv.threshold(mask, 0, 255, cv.THRESH_BINARY)
        seg = cv.imread(seg_dir, -1)
        _, seg = cv.threshold(seg, 0, 255, cv.THRESH_BINARY)

        mask = transforms.ToTensor()(mask.copy())
        seg = transforms.ToTensor()(seg.copy())

        mask = mask * seg

        # mask = np.array(mask)
        _, x, y = mask.shape  # x:高；y:宽
        w = 0
        human_pix = 0
        for row in range(x):
            for col in range(y):
                if (mask[0][row][col]) == 1:
                    w = w + 1

        for row in range(x):
            for col in range(y):
                if (seg[0][row][col]) == 1:
                    human_pix = human_pix + 1

        rate_white = w / (human_pix)
        rate_white_sum += rate_white
        print('{}/{}, {}%'.format(count, len(listdir(mask_path)), round(rate_white * 100, 2)))
    rate_white_sum = rate_white_sum / len(listdir(mask_path))

    print("white rate:", round(rate_white_sum * 100, 2), '%')


if __name__ == "__main__":
    calc_ratio(mask_path, seg_path)
