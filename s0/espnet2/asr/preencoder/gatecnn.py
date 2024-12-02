# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

from collections.abc import Iterable
from itertools import repeat
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class GateCnn(nn.Module):
    def __init__(self,encoder_output_dim=512):
        super(GateCnn, self).__init__()
        fm01= 64
        fm02= int(fm01*2)	#128
        fm03= int(fm01*4)	#256
        fm04= int(fm01*8)	#512
        fm05= int(fm01*16)	#1024
        fm06= int(fm01*32)	#2048

        self.conv1 = ConvBN(1, fm01, 3, 1, 1, 0.1, 0.1, "ReLU")
        self.pool1 = nn.MaxPool2d((2, 2), (2, 2), ceil_mode=True) #dili change for dim 80, (1,2)->(2,2)

        self.RA1 = ResidualAttention(fm01, fm01, fm01, 3, 1, 1) # channel input size = pos1 ;channel output size = pos1 + pos2; pos3 is project size
        
        self.trans1 = ConvBN(fm01+fm01, fm02, 1, 1, 0, 0.1, 0.1, "ReLU")
        self.pool2 = nn.MaxPool2d((2, 2), (2, 2), ceil_mode=True)

        self.RA2 = ResidualAttention(fm02, fm01, fm02, 3, 1, 1)
        self.RA3 = ResidualAttention(fm02+fm01, fm01, fm02, 3, 1, 1)
        
        self.trans2 = ConvBN(fm02 + 2*fm01, fm03, 1, 1, 0, 0.1, 0.1, "ReLU")
        self.pool3 = nn.MaxPool2d((2, 1), (2, 1), ceil_mode=True) 

        self.RA4 = ResidualAttention(fm03, fm01, fm03, 3, 1, 1)
        self.RA5 = ResidualAttention(fm03 + fm01, fm01, fm03, 3, 1, 1) 
        self.RA6 = ResidualAttention(fm03 + 2*fm01, fm01, fm03, 3, 1, 1)
        self.RA7 = ResidualAttention(fm03 + 3*fm01, fm01, fm03, 3, 1, 1)
        self.RA8 = ResidualAttention(fm03 + 4*fm01, fm01, fm03, 3, 1, 1)
        self.RA9 = ResidualAttention(fm03 + 5*fm01, fm01, fm03, 3, 1, 1)
        self.RA10 = ResidualAttention(fm03 + 6*fm01, fm01, fm03, 3, 1, 1)
        self.RA11 = ResidualAttention(fm03 + 7*fm01, fm01, fm03, 3, 1, 1)
        
        self.trans3 = ConvBN(fm03 + 8*fm01, fm04, 1, 1, 0, 0.1, 0.1, "ReLU") 
        self.pool4 = nn.MaxPool2d((2, 1), (2, 1), ceil_mode=True)

        self.RA12 = ResidualAttention(fm04, fm01, fm04, 3, 1, 1)

        self.trans4 = ConvBN(fm04 + fm01, fm04, 1, 1, 0, 0.1, 0.1, "ReLU")
        self.pool5 = nn.MaxPool2d((2, 1), (2, 1), ceil_mode=True) 
        
        self.conv_output1 = ConvBN(fm04, 1024, 3, 1, (0, 1), 0.1, 0.1, "ReLU")    
        self.conv_output2 = ConvBN(1024, encoder_output_dim, 1, 1, 0, 0.1, 0.1, "ReLU")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, x = self.conv1(x)
        x = self.pool1(x) 

        x = self.RA1(x) 
        _, x = self.trans1(x)
        x = self.pool2(x) 

        x = self.RA2(x)
        x = self.RA3(x)
        _, x = self.trans2(x)
        x = self.pool3(x) 

        x = self.RA4(x)
        x = self.RA5(x)
        x = self.RA6(x)
        x = self.RA7(x)
        x = self.RA8(x)
        x = self.RA9(x)
        x = self.RA10(x)

        x = self.RA11(x)
        _, x = self.trans3(x)
        x = self.pool4(x)  

        x = self.RA12(x)
        _, x = self.trans4(x)
        x = self.pool5(x)  

        _, x = self.conv_output1(x)
        n, c, h, w = x.size()
        x = x.reshape(n, -1, 1, w)
        _, x = self.conv_output2(x)

        return x

class ConvBN(nn.Module):
    def __init__(self, input_channel, output_channel, kernel, stride, pad, bias_value, relu_value, act_type="None"):
        super(ConvBN, self).__init__()
        self.conv=nn.Conv2d(input_channel, output_channel, kernel, stride, pad)
        self.bn=nn.BatchNorm2d(output_channel, momentum=0.99)
        nn.init.constant_(self.conv.bias.data, bias_value)
        nn.init.constant_(self.bn.weight.data, 1)
        nn.init.constant_(self.bn.bias.data, 0)
        self.act_type = act_type
        if act_type == "ReLU":
            self.activation = nn.LeakyReLU(negative_slope=relu_value)
        if act_type == "Tanh":
            self.activation = nn.Tanh()
        if act_type == "None":
            self.activation = NullModule()

    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        conv = self.conv(x)
        bn = self.bn(conv)
        activation = self.activation(bn)
        return conv, activation

class ResidualAttention(nn.Module):
    def __init__(self, fm_in, fm, fm_project, kernel, stride, pad):
        super(ResidualAttention, self).__init__()
        self.project=ConvBN(fm_in, fm_project, (1, 1), 1, 0, 0.1, 0.1, "ReLU")

        self.conv1 = ConvBN(fm_project, fm, kernel, stride, pad, 0.1, 0.1, "ReLU")
        self.conv2 = ConvBN(fm, fm, kernel, stride, pad, 0.1, 0.1, "ReLU")

        self.conv_downsample_1 = ConvBN(fm_project, fm, kernel, stride, pad, 0.1, 0.1, "ReLU")
        self.pool1 = nn.MaxPool2d((1, 2), (1, 2))

        self.conv_downsample_2 = ConvBN(fm, fm, kernel, stride, pad, 0.1, 0.1, "ReLU")
        self.pool2 = nn.MaxPool2d((1, 2), (1, 2))

        self.conv_downsample_3 = ConvBN(fm, fm, kernel, stride, pad, 0.1, 0.1, "ReLU")
        self.pool3 = nn.MaxPool2d((1, 2), (1, 2))

        self.conv_downsample_4 = ConvBN(fm, fm, kernel, stride, pad, 0.1, 0.1, "ReLU")
        self.pool4 = nn.MaxPool2d((1, 2), (1, 2))

        self.conv3 = ConvBN(fm, fm, kernel, stride, pad, 0.1, 0.1, "ReLU")

        self.deconv4 = Deconv(fm, fm, (1, 2), (1, 2), 0, 0.1)
        self.bn_deconv4 = nn.BatchNorm2d(fm, momentum=0.99)
        self.relu_deconv4 = nn.LeakyReLU(negative_slope=0.1)
        self.conv_deconv4 = ConvBN(fm, fm, kernel, stride, pad, 0.1, 0.1, "ReLU")

        self.deconv3 = Deconv(fm, fm, (1, 2), (1, 2), 0, 0.1)
        self.bn_deconv3 = nn.BatchNorm2d(fm, momentum=0.99)
        self.relu_deconv3 = nn.LeakyReLU(negative_slope=0.1)
        self.conv_deconv3 = ConvBN(fm, fm, kernel, stride, pad, 0.1, 0.1, "ReLU")

        self.deconv2 = Deconv(fm, fm, (1, 2), (1, 2), 0, 0.1)
        self.bn_deconv2 = nn.BatchNorm2d(fm, momentum=0.99)
        self.relu_deconv2 = nn.LeakyReLU(negative_slope=0.1)
        self.conv_deconv2 = ConvBN(fm, fm, kernel, stride, pad, 0.1, 0.1, "ReLU")

        self.deconv1 = Deconv(fm, fm, (1, 2), (1, 2), 0, 0.1)
        self.bn_deconv1 = nn.BatchNorm2d(fm, momentum=0.99)
        self.relu_deconv1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv_deconv1 = ConvBN(fm, fm, kernel, stride, pad, 0.1, 0.1, "Tanh")

        self.conv_output_1 = ConvBN(fm, fm, kernel, stride, pad, 0.1, 0.1, "ReLU")
        self.conv_output_2 = ConvBN(fm, fm, kernel, stride, pad, 0.1, 0.1, "ReLU")
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        _, project = self.project(x)
        conv1, relu1 = self.conv1(project)
        conv2, relu2 = self.conv2(relu1)

        conv_downsample_1, relu_downsample_1 = self.conv_downsample_1(project)
        pool_downsample1 = self.pool1(relu_downsample_1)

        conv_downsample_2, relu_downsample_2 = self.conv_downsample_2(pool_downsample1)
        pool_downsample2 = self.pool2(relu_downsample_2)

        conv_downsample_3, relu_downsample_3 = self.conv_downsample_3(pool_downsample2)
        pool_downsample3 = self.pool3(relu_downsample_3)

        conv_downsample_4, relu_downsample_4 = self.conv_downsample_4(pool_downsample3)
        pool_downsample4 = self.pool4(relu_downsample_4)

        _, conv3 = self.conv3(pool_downsample4)

        deconv_upsample_4 = self.deconv4(conv3)
        sum_upsample_4 = conv_downsample_4 + deconv_upsample_4
        bn_upsample_4 = self.bn_deconv4(sum_upsample_4)
        relu_upsample_4 = self.relu_deconv4(bn_upsample_4)
        _, conv_upsample_4 = self.conv_deconv4(relu_upsample_4)

        deconv_upsample_3 = self.deconv3(conv_upsample_4)
        sum_upsample_3 = deconv_upsample_3 + conv_downsample_3
        bn_upsample_3 = self.bn_deconv3(sum_upsample_3)
        relu_upsample_3 = self.relu_deconv3(bn_upsample_3)
        _, conv_upsample_3 = self.conv_deconv3(relu_upsample_3)

        deconv_upsample_2 = self.deconv2(conv_upsample_3)
        sum_upsample_2 = deconv_upsample_2 + conv_downsample_2
        bn_upsample_2 = self.bn_deconv2(sum_upsample_2)
        relu_upsample_2 = self.relu_deconv2(bn_upsample_2)
        _, conv_upsample_2 = self.conv_deconv2(relu_upsample_2)

        deconv_upsample_1 = self.deconv1(conv_upsample_2)
        sum_upsample_1 = deconv_upsample_1 + conv1
        bn_upsample_1 = self.bn_deconv1(sum_upsample_1)
        relu_upsample_1 = self.relu_deconv1(bn_upsample_1)
        _, conv_upsample_1 = self.conv_deconv1(relu_upsample_1)

        output_product = conv_upsample_1 * relu2
        output_sum = output_product + relu2

        _, output_conv1 = self.conv_output_1(output_sum)
        _, output_conv2 = self.conv_output_2(output_conv1)

        output_concatenate = torch.cat((x, output_conv2), dim=1)

        return output_concatenate

class Deconv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel, stride, pad, bias_value):
        super(Deconv, self).__init__()
        self.deconv = nn.ConvTranspose2d(input_channel, output_channel, kernel, stride, pad)
        # xavier(self.deconv.weight)
        nn.init.constant_(self.deconv.bias.data, bias_value)


    def forward(self, x):
        x = self.deconv(x)
        return x


if __name__ == "__main__":
    #(B, C, feat, T)
    from torchsummary import summary
    from espnet2.torch_utils.model_summary import model_summary
    # feats = torch.rand(2,1,80,64*5)
    model = GateCnn().to("cuda")
    summary(model, (1,80,64*5))
    # print(model_summary(model))
    # feats = model(feats)
    # print(feats.shape)