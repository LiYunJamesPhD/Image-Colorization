import os

import torch
import torchvision.models as models
from torch import nn, optim
from .networks import ResnetGenerator, PixelDiscriminator


# refer to https://github.com/KupynOrest/DeblurGAN
class PerceptualLoss(object):
    def __init__(self, loss=nn.MSELoss()):
        super().__init__()
        self.criterion = loss
        self.contentFunc = self.contentFunc()

    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i,layer in enumerate(list(cnn)):
            model.add_module(str(i),layer)
            if i == conv_3_3_layer:
                break
        return model

    def getLoss(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss


class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        super().__init__()
        
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()

    def get_labels(self, preds, target_is_real):
        labels = self.real_label if target_is_real else self.fake_label
        return labels.expand_as(preds)

    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss


# adopt the code from https://towardsdatascience.com/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8
class Pix2PixGANModel(nn.Module):
    
    def __init__(self, net_G=None, lr_G=2e-4, lr_D=2e-4,
                 beta1=0.5, beta2=0.999, 
                 lambda_L1=100., lambda_perceptual=10, device=None):
        super().__init__()
        
        self.device = device
        self.lambda_L1 = lambda_L1
        self.lambda_perceptual = lambda_perceptual

        if net_G is None:
            net_G = ResnetGenerator(in_nc=1, out_nc=2)
            net_G = net_G.to(self.device)
            self.net_G = self.init_weights(net_G)
        else:
            self.net_G = net_G.to(self.device)

        self.net_D = PixelDiscriminator(in_nc=3, n_layers=3, ndf=64)
        self.net_D = self.net_D.to(self.device)
        self.net_D = self.init_weights(self.net_D)

        self.GAN_loss = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))

    def init_weights(self, net, init='xavier', gain=0.02):
        
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and 'Conv' in classname:
                if init == 'norm':
                    nn.init.normal_(m.weight.data, mean=0.0, std=gain)
                elif init == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif 'BatchNorm2d' in classname:
                nn.init.normal_(m.weight.data, 1., gain)
                nn.init.constant_(m.bias.data, 0.)
        net.apply(init_func)
        print(f"model initialized with {init} initialization")
        
        return net

    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, in_data):
        self.L = in_data['L'].to(self.device)
        self.ab = in_data['ab'].to(self.device)
        self.real_img = torch.cat([self.L, self.ab], dim=1)

    def inference_val(self):
        self.net_G.eval()
        self.net_D.eval()

        with torch.no_grad():
            fake_color = self.net_G(self.L)

        return fake_color

    def inference_test(self, input_data):
        self.net_G.eval()
        self.net_D.eval()

        with torch.no_grad():
            fake_color = self.net_G(input_data)

        return fake_color

    def load_networks(self, save_dir, epoch):
        """
        load a pre-trained models
        epoch (int): current epoch
        """
        for name in ['D', 'G']:
            if isinstance(name, str):
                model_name = '%s_net_%s.pth' % (epoch, name)
                model_path = os.path.join(save_dir, model_name)

                if name == 'G':
                    self.net_G.load_state_dict(torch.load(model_path))
                else:
                    self.net_D.load_state_dict(torch.load(model_path))

    def save_networks(self, save_dir, epoch):
        """
        Save all the networks to the disk
        
        epoch (int): current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in ['D', 'G']:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(save_dir, save_filename)
                net = self.net_G if name == 'G' else self.net_D

                torch.save(net.cpu().state_dict(), save_path)
                self.net_G.to(self.device)
                self.net_D.to(self.device)

    def forward(self):
        self.fake_color = self.net_G(self.L)
    
    def backward_D(self):
        fake_img = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_img.detach())
        self.fake_D_loss = self.GAN_loss(fake_preds, False)

        real_preds = self.net_D(self.real_img)
        self.real_D_loss = self.GAN_loss(real_preds, True)
        self.loss_D = (self.fake_D_loss + self.real_D_loss) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_img = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_img)
        self.G_GAN_loss = self.GAN_loss(fake_preds, True)
        self.G_L1_loss = self.L1_loss(self.fake_color, self.ab) * self.lambda_L1
        self.G_perceptual_loss = self.perceptual_loss.getLoss(fake_img, self.real_img) * self.lambda_perceptual
        self.G_loss = self.G_GAN_loss + self.G_L1_loss + self.G_perceptual_loss
        self.G_loss.backward()

    def optimize(self):
        self.forward()
        # train D 
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()

        # train G
        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()


