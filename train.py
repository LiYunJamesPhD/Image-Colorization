import torch
import argparse
import os

from data.create_colorize_data import split_imges, make_color_dataloaders
from models.pix2pixGAN import Pix2PixGANModel
from utils import create_loss_meters, update_losses, log_results


def GetArguments():
    parser = argparse.ArgumentParser(description="Image Coloring")
    # basic arguments
    parser.add_argument('--dataroot', required=True, help='path to images (should be a directory)')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--splitPercent', type=float, default=0.97, help='split ratio for training-val data partition')
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu ids: e.g. 0 or 1.')
    # model arguments
    parser.add_argument('--lambdaL1', type=float, default=100.0, help='Lambda to L1 loss')
    parser.add_argument('--lambdaPerceptual', type=float, default=10.0, help='Lambda to Perceptual loss')
    # additional arguments
    parser.add_argument('--epoch', type=int, default=200, help='Max Epoch')
    # training arguments
    parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')

    return parser.parse_args()


def main():

    opt = GetArguments()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    if not os.path.exists(opt.checkpoints_dir):
        os.mkdir(opt.checkpoints_dir)

    # load images and split them to training and validation
    input_dir = opt.dataroot + '/*.jpg'
    train_paths, val_paths = split_imges(input_dir, opt.splitPercent)
    
    train_dataloader = make_color_dataloaders(input_path=train_paths, split='train')
    val_dataloader = None
    if not val_paths:
        val_dataloader = make_color_dataloaders(input_path=val_paths, split='val')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coloring_net = Pix2PixGANModel(device=device, lambda_L1=opt.lambdaL1, lambda_perceptual=opt.lambdaPerceptual) 

    # training
    for e in range(opt.epoch):
        loss_meter_dict = create_loss_meters()

        # === batch loop ===
        for idx, data in enumerate(train_dataloader):
            coloring_net.setup_input(data)
            coloring_net.optimize()
            update_losses(coloring_net, loss_meter_dict, count=data['L'].size(0))

            if idx % opt.print_freq == 0:
                print(f"\nEpoch {e + 1}/{opt.epoch}")
                print(f"Batch: {idx}/{len(train_dataloader)}")
                log_results(loss_meter_dict)
        # === end batch loop ===

        # save model
        coloring_net.save_networks(opt.checkpoints_dir, e + 1)

if __name__ == '__main__':
    main()
