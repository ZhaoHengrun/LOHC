import argparse
import os
import sys
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import utils as vutils
import cv2 as cv

from dataset import *
from models.UNet import UNet
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", default=True, action="store_true", help="use cuda?")
# parser.add_argument("--dataset_seg", default='datasets/train/seg/', type=str, help="dataset path")
parser.add_argument("--dataset_seg", default='datasets/train/AHP/seg/', type=str, help="dataset path")
parser.add_argument("--dataset_img", default='datasets/train/AHP/img/', type=str, help="dataset path")
parser.add_argument("--checkpoints_path", default='checkpoints/UNet_seg/', type=str,
                    help="checkpoints path")
parser.add_argument("--resume", default='checkpoints/UNet_seg/last_model.pth', type=str,
                    help="path to latest checkpoint (default: none)")
parser.add_argument("--batchSize", type=int, default=48, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=2000, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=50,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=200")
parser.add_argument("--start_epoch", default=1, type=int, help="manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=8, help="number of threads for data loader to use")
parser.add_argument("--gpu", default=0, type=int, help="which gpu to use")
parser.add_argument("--visualization", default='', type=str, help="none or wandb or visdom")
opt = parser.parse_args()

min_avr_loss = 99999999
epoch_avr_loss = 0
n_iter = 0

lr = opt.lr
adjust_lr_flag_final = False
adjust_lr_flag_dct = False
adjust_lr_flag_pix = False
stop_flag = False
save_flag = True


def main():
    global opt, model
    global stop_flag

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.cuda.set_device(opt.gpu)
    device = torch.device('cuda:{}'.format(opt.gpu))

    print("===> Loading datasets")
    train_set = TrainDatasetWithMask(img_path=opt.dataset_img, seg_path=opt.dataset_seg)
    training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize,
                                      shuffle=True, num_workers=opt.threads)

    # valid_set = ValidDatasetSingleFrame(target_path=opt.dataset_valid_gt, input_path=opt.dataset_valid_input)
    # valid_data_loader = DataLoader(dataset=valid_set, batch_size=1,
    #                                shuffle=False, num_workers=opt.threads)

    print("===> Building model")
    model = UNet()

    loss_fun = nn.L1Loss()

    print("===> Setting GPU")

    model = model.to(device)
    loss_fun = loss_fun.to(device)

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    if opt.resume:
        if os.path.isfile(opt.resume):
            print('==> load pretrained model from:{}'.format(opt.resume))
            checkpoint = torch.load(opt.resume)
            model.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            opt.start_epoch = checkpoint['epoch']
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    print("===> Training")
    if not os.path.exists(opt.checkpoints_path):
        os.makedirs(opt.checkpoints_path)
        print('create path:', opt.checkpoints_path)
        os.makedirs('{}{}'.format(opt.checkpoints_path, 'img/'))
        os.makedirs('{}{}'.format(opt.checkpoints_path, 'code/'))

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('num_params', num_params / 1e6)

    # with open("{}log.txt".format(opt.checkpoints_path), "a") as f:
    #     f.write(
    #         'model:{}\n num_params:{}\n'.format(opt.checkpoints_path, num_params / 1e6))

    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        if stop_flag is True:
            print('finish')
            break
        else:
            train(training_data_loader, optimizer, model, loss_fun, epoch)
            save_checkpoint(model, optimizer, epoch)


def train(training_data_loader, optimizer, model, loss_fun, epoch):
    global min_avr_loss
    global save_flag
    global epoch_avr_loss
    global n_iter
    global final_psnr_avr
    global dct_psnr_avr
    global pix_psnr_avr

    avr_total_loss = 0

    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        n_iter = iteration
        input, target, _ = batch[0], batch[1], batch[2]
        # save_image_tensor(input[1, 2, :, :].unsqueeze(0), '{}{}.png'.format('results/input/', iteration))
        # save_image_tensor(target[1, :, :].unsqueeze(0), '{}{}.png'.format('results/target/', iteration))

        input = input.cuda()
        target = target.cuda()

        # masked_target = target * (1 - mask)
        # masked_img = input * (1 - mask)
        # input_img = torch.cat([masked_img, mask], dim=1)

        output = model(input)

        optimizer.zero_grad()

        loss = loss_fun(output, target)
        loss.backward()
        optimizer.step()

        print("> E[{}]({}/{}): loss:{:.4f} ".format(epoch, iteration, len(training_data_loader), loss.item()))

        if iteration % 500 == 0:
            save_image_tensor(input[0, :, :, :].unsqueeze(0), '{}img/E[{}]_input.png'.format(opt.checkpoints_path, epoch))
            save_image_tensor(target[0, :, :, :].unsqueeze(0), '{}img/E[{}]_target.png'.format(opt.checkpoints_path, epoch))
            # save_image_tensor(masked_target[0, :, :, :].unsqueeze(0), '{}img/iter[{}]_masked_target.png'.format(opt.checkpoints_path, epoch))
            save_image_tensor(output[0, :, :, :].unsqueeze(0), '{}img/E[{}]_output.png'.format(opt.checkpoints_path, epoch))
            # save_image_tensor(output_wo_mask[0, :, :, :].unsqueeze(0), '{}img/iter[{}]_output_wo_mask.png'.format(opt.checkpoints_path, epoch))
        avr_total_loss += loss.item()
        sys.stdout.write("===> Epoch[{}]({}/{}): Loss: {:.6f}\r".format(epoch, iteration, len(training_data_loader), loss.item()))

    avr_total_loss = avr_total_loss / len(training_data_loader)

    with open("{}log.txt".format(opt.checkpoints_path), "a") as f:
        f.write('E[{}], loss[{:.4f}]\n'.format(epoch, avr_total_loss))


def save_checkpoint(model, optimizer, epoch):
    global min_avr_loss
    global save_flag

    checkpoint = {
        "net": model.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": epoch
    }

    model_folder = opt.checkpoints_path
    if (epoch % 100) == 0:
        torch.save(checkpoint, model_folder + "model_epoch_{}.pth".format(epoch))
        print("Checkpoint saved to {}".format(model_folder))
    # if save_flag is True:
    #     torch.save(model, model_folder + "best_model.pth")
    #     save_flag = False
    torch.save(checkpoint, model_folder + "last_model.pth")


if __name__ == "__main__":
    main()
