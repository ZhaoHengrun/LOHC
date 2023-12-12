import argparse
import os
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from dataset import *
# from mae_gan import *
from models.models_mae_bg_mask import mae_vit_base_patch4
from models.ffc import Lama
from models.pix2pixhd import NLayerDiscriminator
from models.UNet import UNet
from losses.perceptual import ResNetPL
from losses.adversarial import NonSaturatingWithR1
from losses.feature_matching import feature_matching_loss, masked_l1_loss
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--start_epoch", type=int, default=1, help="epoch to start training from")
parser.add_argument("--resume", default='checkpoints/mae_gan_2/last_model.pth', type=str, help="path to latest checkpoint (default: none)")
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
# parser.add_argument("--dataset_seg", default='/home/zhr/Project/UNet_Human_Seg/datasets/train/seg/', type=str, help="dataset path")
parser.add_argument("--dataset_img", default='datasets/train/AHP/img/', type=str, help="dataset path")
parser.add_argument("--dataset_seg", default='datasets/train/AHP/seg/', type=str, help="dataset path")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr_g", type=float, default=1e-4, help="adam: learning rate of generator")
parser.add_argument("--lr_d", type=float, default=1e-6, help="adam: learning rate of discriminator")
parser.add_argument("--threads", type=int, default=16, help="number of cpu threads to use during batch generation")
parser.add_argument("--checkpoints_path", default='checkpoints/coarse/', type=str, help="checkpoints path")
parser.add_argument('--mae_model_path', type=str, default='checkpoints/pretrain/last_model.pth', help='model file to use')
# parser.add_argument('--seg_model_path', type=str, default='/home/zhr/Project/UNet_Human_Seg/checkpoints/UNet/last_model.pth', help='model file to use')
opt = parser.parse_args()


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device('cuda')

    if not os.path.exists(opt.checkpoints_path):
        os.makedirs(opt.checkpoints_path)
        print('create path:', opt.checkpoints_path)
        os.makedirs('{}{}'.format(opt.checkpoints_path, 'img/'))
        os.makedirs('{}{}'.format(opt.checkpoints_path, 'code/'))

    discriminator_img = NLayerDiscriminator(input_nc=3).to(device)
    discriminator_human = NLayerDiscriminator(input_nc=7).to(device)

    mae_model = mae_vit_base_patch4()
    mae_checkpoint = torch.load(opt.mae_model_path, map_location='cuda:0')
    mae_model.load_state_dict(mae_checkpoint['model'])
    mae_model = mae_model.cuda()
    print('load mae from:{}'.format(opt.mae_model_path))

    mae_num_params = 0
    for param in mae_model.parameters():
        mae_num_params += param.numel()
    print('mae_num_params:', mae_num_params / 1e6)

    D_num_params = 0
    for param in discriminator_img.parameters():
        D_num_params += param.numel()
    print('D_num_params:', D_num_params / 1e6)

    # with open("{}log.txt".format(opt.checkpoints_path), "a") as f:
    #     f.write('{}\n G_num_params:{}\n D_num_params:{}\n'.format(opt.checkpoints_path, G_num_params / 1e6, D_num_params / 1e6))

    # Losses
    # criterion_pixel = torch.nn.L1Loss().to(device)
    criterion_pl = ResNetPL().to(device)
    # criterion_adv = torch.nn.BCELoss().to(device)
    # criterion_d = NonSaturatingWithR1()

    # Optimizers
    optimizer_G = torch.optim.Adam(mae_model.parameters(), lr=opt.lr_g)
    optimizer_D_img = torch.optim.Adam(discriminator_img.parameters(), lr=opt.lr_d)
    optimizer_D_human = torch.optim.Adam(discriminator_human.parameters(), lr=opt.lr_d)

    if opt.resume:
        if os.path.isfile(opt.resume):
            print('==> load pretrained model from:{}'.format(opt.resume))
            checkpoint = torch.load(opt.resume)
            discriminator_img.load_state_dict(checkpoint['discriminator_img'])
            discriminator_human.load_state_dict(checkpoint['discriminator_human'])
            mae_model.load_state_dict(checkpoint['mae'])
            optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            optimizer_D_img.load_state_dict(checkpoint['optimizer_D_img'])
            optimizer_D_human.load_state_dict(checkpoint['optimizer_D_human'])
            opt.start_epoch = checkpoint['epoch'] + 1
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    train_set = TrainDatasetWithMask(img_path=opt.dataset_img, seg_path=opt.dataset_seg)
    training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size,
                                      shuffle=True, num_workers=opt.threads)

    #  Training
    for epoch in range(opt.start_epoch, opt.n_epochs + 1):
        train(training_data_loader, optimizer_G, optimizer_D_img, optimizer_D_human, discriminator_img, discriminator_human, epoch, criterion_pl, mae_model)


def train(training_data_loader, optimizer_G, optimizer_D_img, optimizer_D_human, discriminator_img, discriminator_human, epoch, criterion_pl, mae_model):
    D_loss_img = 0
    D_loss_human = 0
    G_loss = 0
    adv_img = 0
    adv_human = 0
    pixel = 0
    pl = 0
    fm_i = 0
    fm_h = 0
    mae_model.train()
    discriminator_img.train()
    discriminator_human.train()
    for iteration, batch in enumerate(training_data_loader):
        img, seg, mask = batch[0], batch[1], batch[2]
        img = img.cuda()
        seg = seg.cuda()
        mask = mask.cuda()
        b, _, h, w = img.shape
        mask = mask[0].repeat(b, 1, 1, 1)

        seg = F.interpolate(seg, size=(64, 64), mode='bilinear')
        mask = F.interpolate(mask, size=(64, 64), mode='bilinear')
        img = F.interpolate(img, size=(64, 64), mode='bilinear')
        human = img * seg

        masked_img = img * (1 - mask)
        masked_human = human * (1 - mask)
        masked_seg = seg * (1 - mask)

        img_with_seg = torch.cat([masked_img, masked_seg], dim=1)
        human_with_seg = torch.cat([masked_human, masked_seg], dim=1)
        #  ------------------------------Train Generators------------------------------
        optimizer_G.zero_grad()

        pred, _ = mae_model(img_with_seg, human_with_seg, mask, threshold=0)

        pred_seg = pred[:, -1, :, :].unsqueeze(1)
        pred_img = pred[:, :-1, :, :]
        pred_human = pred_img * pred_seg

        # Total generator loss
        loss_G, l1_loss, adv_img_loss, adv_human_loss, fm_img_loss, fm_human_loss, pl_loss = generator_loss(
            img, pred_img, human, pred_human, seg, pred_seg, mask, discriminator_img, discriminator_human, criterion_pl)

        loss_G.backward()
        optimizer_G.step()

        #  ------------------------------Train Discriminator----------------------------
        optimizer_D_img.zero_grad()
        loss_D_img = discriminator_loss(img, pred_img, mask, discriminator_img)
        loss_D_img.backward()
        optimizer_D_img.step()

        human_with_seg = torch.cat([img, human, seg], dim=1)
        pred_human_with_seg = torch.cat([pred_img, pred_human, pred_seg], dim=1)
        optimizer_D_human.zero_grad()
        loss_D_human = discriminator_loss(human_with_seg, pred_human_with_seg, mask, discriminator_human)
        loss_D_human.backward()
        optimizer_D_human.step()
        #  ------------------------------Training End------------------------------------

        D_loss_img += loss_D_img.item()
        D_loss_human += loss_D_human.item()
        G_loss += loss_G.item()
        adv_img += adv_img_loss.item()
        adv_human += adv_human_loss.item()
        pixel += l1_loss.item()
        fm_i += fm_img_loss.item()
        fm_h += fm_human_loss.item()
        pl += pl_loss.item()

        print("> E[{}]({}/{}): G:{:.4f}, I:{:.4f}, AI:{:.4f}, AH:{:.4f}, FMI:{:.4f}, FMH:{:.4f}, PL:{:.4f}, D_I:{:.4f}, D_H:{:.4f} ".format(
            epoch, iteration, len(training_data_loader), loss_G.item(), l1_loss.item(), adv_img_loss.item(), adv_human_loss.item(), fm_img_loss.item(),
            fm_human_loss.item(), pl_loss.item(), loss_D_img.item(), loss_D_human.item()))

        if iteration % 500 == 0:
            save_image_tensor(masked_img[0, :, :, :].unsqueeze(0), '{}img/epoch{}_masked_img.png'.format(opt.checkpoints_path, epoch))
            save_image_tensor(pred_img[0, :, :, :].unsqueeze(0), '{}img/epoch{}_pred.png'.format(opt.checkpoints_path, epoch))
            save_image_tensor(img[0, :, :, :].unsqueeze(0), '{}img/epoch{}_img.png'.format(opt.checkpoints_path, epoch))
            save_image_tensor(mask[0, :, :, :].unsqueeze(0), '{}img/epoch{}_mask.png'.format(opt.checkpoints_path, epoch))
            save_image_tensor(seg[0, :, :, :].unsqueeze(0), '{}img/epoch{}_seg.png'.format(opt.checkpoints_path, epoch))
            save_image_tensor(human[0, :, :, :].unsqueeze(0), '{}img/epoch{}_human.png'.format(opt.checkpoints_path, epoch))
            save_image_tensor(pred_seg[0, :, :, :].unsqueeze(0), '{}img/epoch{}_pred_seg.png'.format(opt.checkpoints_path, epoch))
            save_image_tensor(pred_human[0, :, :, :].unsqueeze(0), '{}img/epoch{}_pred_human.png'.format(opt.checkpoints_path, epoch))

    avg_D_loss_img = D_loss_img / len(training_data_loader)
    avg_D_loss_human = D_loss_human / len(training_data_loader)
    avg_G_loss = G_loss / len(training_data_loader)
    avg_adv_loss_img = adv_img / len(training_data_loader)
    avg_adv_loss_human = adv_human / len(training_data_loader)
    avg_pixel_loss = pixel / len(training_data_loader)
    avg_fm_i_loss = fm_i / len(training_data_loader)
    avg_fm_h_loss = fm_h / len(training_data_loader)
    avg_pl_loss = pl / len(training_data_loader)

    with open("{}log.txt".format(opt.checkpoints_path), "a") as f:
        f.write(
            'E[{}], G[{:.4f}], I[{:.4f}], AI[{:.4f}], '
            'AH[{:.4f}], FMI[{:.4f}], FMH[{:.4f}], PL[{:.4f}], DI[{:.4f}], DH[{:.4f}]]\n'.format(
                epoch, avg_G_loss, avg_pixel_loss, avg_adv_loss_img, avg_adv_loss_human, avg_fm_i_loss, avg_fm_h_loss, avg_pl_loss,
                avg_D_loss_img, avg_D_loss_human))

    checkpoint = {
        "discriminator_img": discriminator_img.state_dict(),
        "discriminator_human": discriminator_human.state_dict(),
        "mae": mae_model.state_dict(),
        'optimizer_G': optimizer_G.state_dict(),
        'optimizer_D_img': optimizer_D_img.state_dict(),
        'optimizer_D_human': optimizer_D_human.state_dict(),
        "epoch": epoch
    }
    torch.save(checkpoint, opt.checkpoints_path + "last_model.pth")


def generator_loss(img, pred, human, pred_human, seg, pred_seg, mask, discriminator_img, discriminator_human, criterion_pl):
    # L1
    img_loss = masked_l1_loss(pred, img, mask, weight_known=1, weight_missing=1)
    human_loss = masked_l1_loss(pred_human, human, mask, weight_known=1, weight_missing=1)
    seg_loss = masked_l1_loss(pred_seg, seg, mask, weight_known=1, weight_missing=1)

    l1_loss = img_loss + human_loss + seg_loss
    total_loss = l1_loss

    # adv img
    discr_real_pred_img, discr_real_img_features = discriminator_img(img)
    discr_fake_pred_img, discr_fake_img_features = discriminator_img(pred)
    fake_img_loss = F.softplus(-discr_fake_pred_img)
    adv_img_loss = fake_img_loss.mean() * 0.001
    total_loss = total_loss + adv_img_loss

    # adv human
    human_with_seg = torch.cat([img, human, seg], dim=1)
    pred_human_with_seg = torch.cat([pred, pred_human, pred_seg], dim=1)
    discr_real_pred_human, discr_real_human_features = discriminator_human(human_with_seg)
    discr_fake_pred_human, discr_fake_human_features = discriminator_human(pred_human_with_seg)
    fake_loss = F.softplus(-discr_fake_pred_human)
    adv_human_loss = fake_loss.mean() * 0.001
    total_loss = total_loss + adv_human_loss

    # feature matching
    # fm_loss = torch.zeros_like(l1_loss)
    fm_loss = feature_matching_loss(discr_fake_img_features, discr_real_img_features, mask=None) * 0.01
    total_loss = total_loss + fm_loss

    # feature matching human
    fm_human_loss = feature_matching_loss(discr_fake_human_features, discr_real_human_features, mask=None) * 0.01  # * 100
    total_loss = total_loss + fm_human_loss

    # resnet perceptual loss
    pl_loss = torch.zeros_like(l1_loss)
    # pl_loss = criterion_pl(pred, img) * 1  # 30
    # total_loss = total_loss + pl_loss

    return total_loss, l1_loss, adv_img_loss, adv_human_loss, fm_loss, fm_human_loss, pl_loss


def discriminator_loss(img, pred, mask, discriminator):
    pred = pred.detach()
    img = img.detach()
    discr_real_pred, discr_real_features = discriminator(img)
    discr_fake_pred, discr_fake_features = discriminator(pred)

    real_loss = F.softplus(-discr_real_pred)
    grad_penalty = 0
    img.requires_grad = False

    grad_penalty = grad_penalty * 0.001
    fake_loss = F.softplus(discr_fake_pred)

    mask = F.interpolate(mask, discr_fake_pred.shape[-2:], mode='bilinear')
    fake_loss = fake_loss * mask

    sum_discr_loss = real_loss + grad_penalty + fake_loss
    loss_D = sum_discr_loss.mean()

    return loss_D


if __name__ == "__main__":
    main()
