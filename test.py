# from __future__ import print_function
import argparse
from os import listdir
from torchvision import transforms
from torchvision import utils as vutils
from PIL import Image
from torchvision import transforms
from dataset import *
from models.models_mae_bg_mask import mae_vit_base_patch4
from models.ffc import Hiin, Lama
from models.UNet import UNet
from math import exp
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import make_dir, save_image_tensor, resizeAndPadTensor, resize, resize_back
from einops import rearrange
import pyiqa

# Training settings
parser = argparse.ArgumentParser(description='Test')
parser.add_argument("--test_length", default=-1, type=int, help="how many img to test")
parser.add_argument("--dataset_test_input", default='datasets/test/AHP/img/', type=str, help="dataset path")
parser.add_argument("--dataset_test_mask", default='datasets/test/AHP/', type=str,
                    help="center_mask, rectangle_mask, voc_mask, xpie_mask, hku_is_mask path")
parser.add_argument('--output_path', default='output/refinement/', type=str, help='where to save the output image')
parser.add_argument('--model_path', type=str, default='checkpoints/refinement/last_model.pth', help='model file to use')
parser.add_argument('--seg_model_path', type=str, default='checkpoints/UNet_seg/last_model.pth', help='model file to use')
parser.add_argument('--cuda', default=True, action='store_true', help='use cuda')
parser.add_argument('--save_img', default=True, action='store_true', help='save img')
opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def mask_patchify(mask):
    h = w = mask.shape[-1]

    mask = rearrange(mask, 'b c (patch_num_h patch_size_h) (patch_num_w patch_size_w) -> b (patch_num_h patch_num_w) (patch_size_h patch_size_w c)',
                     patch_size_h=4, patch_size_w=4)
    patch_mask = (mask.mean(-1) > 0).float().unsqueeze(-1)  # [20, 256, 1]
    patch_mask = rearrange(patch_mask, 'b (patch_num_h patch_num_w) (patch_size_h patch_size_w c) -> b c (patch_num_h patch_size_h) (patch_num_w patch_size_w)',
                           patch_num_h=16, patch_num_w=16, patch_size_h=1, patch_size_w=1, c=1)
    patch_mask = F.interpolate(patch_mask, (h, w), mode='nearest')
    return patch_mask


def test():
    print('loading model')
    generator = Hiin()
    generator_checkpoint = torch.load(opt.model_path, map_location='cuda:0')
    generator.load_state_dict(generator_checkpoint['generator'])
    generator = generator.cuda()
    generator.eval()

    seg_model = UNet()
    seg_checkpoint = torch.load(opt.seg_model_path, map_location='cuda:0')
    seg_model.load_state_dict(seg_checkpoint['net'])
    seg_model.eval()
    seg_model = seg_model.cuda()

    mae_model = mae_vit_base_patch4()
    # mae_checkpoint = torch.load(opt.mae_model_path, map_location='cuda:0')
    mae_model.load_state_dict(generator_checkpoint['mae'])
    mae_model = mae_model.cuda()
    mae_model.eval()

    lpips_metric = pyiqa.create_metric('lpips').cuda()
    ssim_metric = pyiqa.create_metric('ssim').cuda()
    psnr_metric = pyiqa.create_metric('psnr').cuda()
    fid_metric = pyiqa.create_metric('fid').cuda()

    mask_types = ['center_mask', 'rectangle_mask', 'voc_mask', 'xpie_mask', 'hku_is_mask']
    for mask_type in mask_types:
        print('mask_type:[{}]'.format(mask_type))
        opt.dataset_test_mask = 'datasets/test/AHP/{}/'.format(mask_type)

        make_dir('output/hiin_finetune/')
        opt.output_path = 'output/hiin_finetune/{}/'.format(mask_type)

        test_set = ValidDatasetWithMask(opt.dataset_test_input, opt.dataset_test_mask, opt.test_length)
        test_data_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
        print('datasets:{}'.format(opt.dataset_test_input))
        print(len(test_data_loader), 'test images')

        make_dir(opt.output_path)

        input_path = '{}input/'.format(opt.output_path)
        human_path = '{}human/'.format(opt.output_path)
        seg_path = '{}seg/'.format(opt.output_path)
        pred_seg_path = '{}pred_seg/'.format(opt.output_path)
        masked_img_path = '{}masked_img/'.format(opt.output_path)
        masked_human_path = '{}masked_human/'.format(opt.output_path)
        mae_pred_path = '{}mae_pred_human/'.format(opt.output_path)
        complete_fake_img_path = '{}complete_fake_img/'.format(opt.output_path)
        pred_img_path = '{}pred_img/'.format(opt.output_path)
        pred_human_path = '{}pred_human/'.format(opt.output_path)
        mask_path = '{}mask/'.format(opt.output_path)
        patch_mask_path = '{}patch_mask/'.format(opt.output_path)

        if opt.save_img is True:
            make_dir(input_path)
            make_dir(human_path)
            make_dir(seg_path)
            make_dir(pred_seg_path)
            make_dir(masked_img_path)
            make_dir(masked_human_path)
            make_dir(mae_pred_path)
            make_dir(complete_fake_img_path)
            make_dir(pred_img_path)
            make_dir(pred_human_path)
            make_dir(mask_path)
            make_dir(patch_mask_path)

        count = 0
        lpips_sum = 0
        ssim_sum = 0
        psnr_sum = 0

        lpips_sum_human = 0
        ssim_sum_human = 0
        psnr_sum_human = 0

        with torch.no_grad():
            for iteration, batch in enumerate(test_data_loader, 1):
                count += 1
                img, mask, input_name_list = batch[0], batch[1], batch[2]
                input_name = '{}.png'.format(input_name_list[0].split('.')[0])
                print('processing:[{}/{}][{}]'.format(count, len(test_data_loader), input_name))

                img = img.cuda()
                mask = mask.cuda()
                seg = seg_model(img)

                masked_img = img * (1 - mask)
                masked_seg = seg * (1 - mask)
                human = img * seg
                masked_human = img * masked_seg

                masked_img_64 = F.interpolate(masked_img, size=(64, 64), mode='bilinear')
                masked_human_64 = F.interpolate(masked_human, size=(64, 64), mode='bilinear')
                masked_seg_64 = F.interpolate(masked_seg, size=(64, 64), mode='bilinear')
                mask_64 = F.interpolate(mask, size=(64, 64), mode='bilinear')
                patch_mask_64 = mask_patchify(mask_64)
                res_mask = patch_mask_64 - mask_64

                img_with_seg = torch.cat([masked_img_64, masked_seg_64], dim=1)
                human_with_seg = torch.cat([masked_human_64, masked_seg_64], dim=1)

                mae_pred, _ = mae_model(img_with_seg, human_with_seg, mask_64, threshold=0)

                mae_pred_seg = mae_pred[:, -1, :, :].unsqueeze(1)
                mae_pred_img = mae_pred[:, :-1, :, :]
                mae_pred_human = mae_pred_img * mae_pred_seg

                mae_pred_input = torch.cat([mae_pred_img, mae_pred_human, mae_pred_seg, mask_64], dim=1)  # 3+3+1+1=8
                input_tensor = torch.cat([masked_img, masked_human, masked_seg, mask], dim=1)  # 3+3+1+1=8

                pred_img, pred_seg = generator(input_tensor, mae_pred_input, patch_mask_64)

                pred_human = pred_img * seg

                lpips_score_human = lpips_metric(pred_human, human)
                ssim_score_human = ssim_metric(pred_human, human)
                psnr_score_human = psnr_metric(pred_human, human)
                lpips_sum_human += lpips_score_human
                ssim_sum_human += ssim_score_human
                psnr_sum_human += psnr_score_human

                complete_fake_img = pred_img * mask + masked_img

                lpips_score = lpips_metric(complete_fake_img, img)
                ssim_score = ssim_metric(complete_fake_img, img)
                psnr_score = psnr_metric(complete_fake_img, img)
                lpips_sum += lpips_score
                ssim_sum += ssim_score
                psnr_sum += psnr_score

                if opt.save_img is True:
                    save_image_tensor(img, '{}{}'.format(input_path, input_name))
                    save_image_tensor(human, '{}{}'.format(human_path, input_name))
                    # save_image_tensor(seg, '{}{}'.format(seg_path, input_name))
                    # save_image_tensor(masked_img, '{}{}'.format(masked_img_path, input_name))
                    # save_image_tensor(masked_human, '{}{}'.format(masked_human_path, input_name))
                    save_image_tensor(pred_human, '{}{}'.format(pred_human_path, input_name))
                    # save_image_tensor(mae_pred, '{}{}'.format(mae_pred_path, input_name))
                    # save_image_tensor(pred_seg, '{}{}'.format(pred_seg_path, input_name))
                    save_image_tensor(complete_fake_img, '{}{}'.format(complete_fake_img_path, input_name))
                    # save_image_tensor(pred_img, '{}{}'.format(pred_img_path, input_name))
                    # save_image_tensor(mask, '{}{}'.format(mask_path, input_name))
                    # save_image_tensor(patch_mask_64, '{}{}'.format(patch_mask_path, input_name))
        lpips_avr_human = lpips_sum_human / len(test_data_loader)
        ssim_avr_human = ssim_sum_human / len(test_data_loader)
        psnr_avr_human = psnr_sum_human / len(test_data_loader)
        print('lpips_human:[{}]'.format(lpips_avr_human))
        print('ssim_human:[{}]'.format(ssim_avr_human))
        print('psnr_human:[{}]'.format(psnr_avr_human))

        lpips_avr = lpips_sum / len(test_data_loader)
        ssim_avr = ssim_sum / len(test_data_loader)
        psnr_avr = psnr_sum / len(test_data_loader)
        print('lpips:[{}]'.format(lpips_avr))
        print('ssim:[{}]'.format(ssim_avr))
        print('psnr:[{}]'.format(psnr_avr))

        fid_avr_human = fid_metric(human_path, pred_human_path)
        print('fid_human:[{}]'.format(fid_avr_human))

        fid_avr = fid_metric(complete_fake_img_path, input_path)
        print('fid:[{}]'.format(fid_avr))

        with open("output/hiin_finetune/log.txt", "a") as f:
            f.write(
                '\nMask_type:[{}]\nPSNR:[{}]\nSSIM:[{}]\nLPIPS:[{}]\nFID:[{}]\nHUMAN\nPSNR_human:[{}]\nSSIM_human:[{}]\nLPIPS_human:[{}]\nFID_human:[{}]\n'.format(
                    mask_type, psnr_avr, ssim_avr, lpips_avr, fid_avr, psnr_avr_human, ssim_avr_human, lpips_avr_human, fid_avr_human))
    print('finish')


if __name__ == "__main__":
    test()
