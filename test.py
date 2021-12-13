import torch
import argparse
import os
import glob
import numpy as np

from data.create_colorize_data import split_imges, make_color_dataloaders, make_test_color_dataloaders
from models.pix2pixGAN import Pix2PixGANModel
from utils import lab_to_rgb, save_image, PSNR_Calculation, SSIM_Calculation


def GetArguments():
    parser = argparse.ArgumentParser(description="Image Coloring")
    # basic arguments
    parser.add_argument('--dataroot', type=str, default='', help='path to train images (should be a directory)')
    parser.add_argument('--testdataroot', type=str, default='', help='path to test images (should be a directory)')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints', help='path to pre-trained models')
    parser.add_argument('--syntheticImgs', type=str, default='./syntheticImgs', help='path to synthetis images')
    parser.add_argument('--splitPercent', type=float, default=0.97, help='split ratio for training-val data partition')
    parser.add_argument('--model_epoch', type=int, default=150, help='epoch number to model inference')
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu ids: e.g. 0 or 1.')

    return parser.parse_args()


def testValDataset(testData, net, opt):
    """inference model (generate color images given grayscale images) on val data"""
    avg_PSNR = 0.0
    avg_SSIM = 0.0

    for idx, data in enumerate(testData):

        file_name = data['name']
        net.setup_input(data)
        
        synthetic_colors = net.inference_val()
        real_imgs = lab_to_rgb(net.L, net.ab) * 255.0
        synthetic_imgs = lab_to_rgb(net.L, synthetic_colors) * 255.0

        # save images
        for real, synthethic, name in zip(real_imgs, synthetic_imgs, file_name):
            save_real_img_path = os.path.join(opt.syntheticImgs, 'real-' + name)
            save_synthetic_img_path = os.path.join(opt.syntheticImgs, 'colored-' + name)

            save_image(real.astype(np.uint8), save_real_img_path)
            save_image(synthethic.astype(np.uint8), save_synthetic_img_path)

        # measure PSNR, SSIM, and FID
        avg_PSNR += PSNR_Calculation(real_imgs, synthetic_imgs)
        avg_SSIM += SSIM_Calculation(real_imgs, synthetic_imgs)

    avg_PSNR = avg_PSNR / len(testData)
    avg_SSIM = avg_SSIM / len(testData)

    print('All the images (real and synthetic images) are stored in the disk!')
    print(f'Average PSNR: {avg_PSNR}')
    print(f'Average PSNR: {avg_SSIM}')


def testTestDataset(testData, net, device, opt):
    """inference model (generate color images given grayscale images) on test data"""
    for idx, data in enumerate(testData):

        imgs = data['img'].to(device)
        file_name = data['name']

        synthetic_colors = net.inference_test(imgs)
        synthetic_imgs = lab_to_rgb(imgs, synthetic_colors)

        # save images
        for synthetic_img, name in zip(synthetic_imgs, file_name):
            save_img_path = os.path.join(opt.syntheticImgs, 'colored-' + name)
            synthetic_img = synthetic_img * 255.0
            save_image(synthetic_img.astype(np.uint8), save_img_path)

    print('All the images (real and synthetic images) are stored in the disk!')


def main():

    opt = GetArguments()

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    if not os.path.exists(opt.syntheticImgs):
        os.mkdir(opt.syntheticImgs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coloring_net = Pix2PixGANModel(device=device)
    # load pre-trained models
    coloring_net.load_networks(opt.checkpoint, opt.model_epoch)

    # load test images
    if not opt.testdataroot:
        # load images and split them to training and validation
        input_dir = opt.dataroot + '/*.jpg'
        train_paths, val_paths = split_imges(input_dir, opt.splitPercent)
        
        if not val_paths:
            raise Exception('Must provide a val dataset if test data are unavailable!')

        train_dataloader = make_color_dataloaders(input_path=train_paths, split='train')
        test_data_loader = make_color_dataloaders(input_path=val_paths, split='val')
        
        # test the pre-trained generator on the val dataset
        testValDataset(test_data_loader, coloring_net, opt)
    else:
        input_dir = opt.testdataroot + '/*.jpg'
        input_paths = glob.glob(input_dir)
        test_data_loader = make_test_color_dataloaders(input_path=input_paths)

        # test the pre-trained generator on the test dataset
        testTestDataset(test_data_loader, coloring_net, device, opt)


if __name__ == '__main__':
    main()


