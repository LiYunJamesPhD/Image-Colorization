import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import math
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size/2)**2/float(2*sigma**2)) for x in range(window_size)])
    
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
    
    return window


def SSIM_Calculation(img1, img2):
    """
    img1 (Numpy): real images
    img2 (Numpy): synthetic images
    """
    # convert inputs to tensors
    img1 = torch.from_numpy(img1).float().permute(0, 3, 1, 2)
    img2 = torch.from_numpy(img2).float().permute(0, 3, 1, 2)

    (_, channel, _, _) = img1.size()

    window_size = 11
    window = create_window(window_size, channel)

    mu1 = F.conv2d(img1, window, padding = int(window_size/2), groups = channel)
    mu2 = F.conv2d(img2, window, padding = int(window_size/2), groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = int(window_size/2), groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = int(window_size/2), groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = int(window_size/2), groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()


def PSNR_Calculation(img1, img2):
    """
    img1 (Numpy): real images
    img2 (Numpy): synthetic images
    """
    mse = np.mean( (img1/255. - img2/255.) ** 2 )

    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count


def create_loss_meters():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G_perceptual = AverageMeter()
    loss_G = AverageMeter()

    return {'fake_D_loss': loss_D_fake,
            'real_D_loss': loss_D_real,
            'loss_D': loss_D,
            'G_GAN_loss': loss_G_GAN,
            'G_L1_loss': loss_G_L1,
            'G_perceptual_loss': loss_G_perceptual,
            'G_loss': loss_G}


def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """
    Save a numpy image to the disk
    
    image_numpy (numpy array): input numpy array
    image_path (str): the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """

    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)

    return np.stack(rgb_imgs, axis=0)


def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")
