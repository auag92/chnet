"""
source: https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py
@author: ashanker9@gatech.edu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from toolz.curried import curry
from collections import OrderedDict


@curry
def mse_loss(y1, y2, scale=1.):
    """standard MSE definition"""
    return ((y1 - y2) ** 2).sum() / y1.data.nelement() * scale


class Padder(nn.Module):
    def __init__(self, padval=[1,1,1,1], padmode="circular"):
        super(Padder, self).__init__()
        self.padder = curry(nn.functional.pad)(pad=padval, mode=padmode)

    def forward(self, x):
        return self.padder(x)


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, tanh=True, conv=True):
        super(UNet, self).__init__()
            
        features = init_features
        
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")
        
        if conv:
            self.conv = nn.Sequential(Padder(padval=[1]*4, padmode="circular"), 
                                      nn.Conv2d(in_channels=features, 
                                                out_channels=out_channels, 
                                                kernel_size=3, 
                                                padding=0))
        else:
            self.conv = lambda x: x
        
        self.tanh = tanh

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        if self.tanh:
            return torch.tanh(self.conv(dec1))
        else:
            return self.conv(dec1)


    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    ('pad1', Padder(padval=[1]*4, padmode="circular")), 
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=0,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    ('pad2', Padder(padval=[1]*4, padmode="circular")),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=0,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
    
class UNet_loop(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32, temporal=1, tanh=True):
        super(UNet_loop, self).__init__()
        
        
        self.temporal = temporal
        
        layers = OrderedDict() 
        for t in range(temporal):
            
            layers[str(t)] = UNet(in_channels=in_channels, 
                              out_channels=out_channels, 
                              init_features=init_features, 
                              tanh=tanh)
            
        self.unets = nn.Sequential(layers)
        
    def forward(self, x):
        """
        :param x:  5D tensor    bz * t * ch * 240 * 240
        :return:
        """
        u_output = []
        for t in range(self.temporal):
            y = self.unets[t](x[:,t])
            u_output.append(y)
            
        return torch.stack(u_output, dim=1)