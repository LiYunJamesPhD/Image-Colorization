import torch
import torch.nn as nn
import functools

# refer the code https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
class Self_Attn(nn.Module):

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        """ Self attention Layer"""
        
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim = -1)
    
    def forward(self, x):
        """
        inputs:
            x : input feature maps( B X C X W X H)
        
        returns: 
            out : self attention value + input feature
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0,2,1) # B X C X (N)
        proj_key =  self.key_conv(x).view(m_batchsize, -1, width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query, proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height) # B X C X N

        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


# refer the code from https://github.com/LiYunJamesPhD/pytorch-CycleGAN-and-pix2pix
class ResnetBlock(nn.Module):
    
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialization"""
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Build a convolution block"""
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward step"""
        # skip connection
        out = x + self.conv_block(x)
        
        return out


# refer the code from https://github.com/LiYunJamesPhD/pytorch-CycleGAN-and-pix2pix
class ResnetGenerator(nn.Module):

    def __init__(self, in_nc, out_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, 
                 n_blocks=6, padding_type='reflect'):
        """
        in_nc (int): the number of channels in input images
        out_nc (int): the number of channels in output images
        ngf (int): the number of filters in the last conv layer
        norm_layer: normalization layer
        use_dropout (bool): if use dropout layers
        n_blocks (int): the number of ResNet blocks
        padding_type (str): the name of padding layer in conv layers: reflect | replicate | zero
        """
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        net_block = [nn.ReflectionPad2d(3),
                     nn.Conv2d(in_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                     norm_layer(ngf),
                     nn.ReLU(True)]

        n_downsampling = 2
        # add downsampling layers
        for i in range(n_downsampling):
            mult = 2 ** i
            net_block += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
        # add Resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            net_block += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, 
                                      use_dropout=use_dropout, use_bias=use_bias)]
            if i == n_blocks - 1:
                net_block += [Self_Attn(ngf * mult, 'relu')]


        # add upsampling layers
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            net_block += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]

        net_block += [nn.ReflectionPad2d(3)]
        net_block += [nn.Conv2d(ngf, out_nc, kernel_size=7, padding=0)]
        net_block += [nn.Tanh()]

        self.net_block = nn.Sequential(*net_block)

    def forward(self, x):
        """Forward step"""
        return self.net_block(x)


# refer the code from https://github.com/LiYunJamesPhD/pytorch-CycleGAN-and-pix2pix
class PixelDiscriminator(nn.Module):
    """Build a PatchGAN discriminator"""

    def __init__(self, in_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """
        in_nc (int): the number of channels in input images
        ndf (int): the number of filters in the last conv layer
        n_layers (int): the number of conv layers in the discriminator
        norm_layer: normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        # if BatchNorm2d has affine parameters, we do not use bias
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [nn.Conv2d(in_nc, ndf, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        # increase the number of filters
        for layer_n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** layer_n, 8)
            self.net += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                    ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.net += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
                ]

        # self.net += [Self_Attn(ndf * nf_mult, 'relu')]
        # the last layer is for the prediction map (only 1 channel)
        self.net += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        """Forward step"""
        return self.net(x)


