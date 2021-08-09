#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import os
from trochvision import dataset, transform, utils
import torch.nn.functional as F
import cv2
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG

class FEAT(nn.Module):
    def __init__(self):
        super(FEAT,self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1,padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1,padding=1)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)




    def forward(self,x1,x2):

        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)

        x2 = self.conv1(x2)
        x2 = self.conv2(x2)
        x2 = self.conv3(x2)

        target=torch.random(x1.shape())
        for i in range(len(x1[2])):
            for j in range(len(x1[3])):

                x_feat=x1[:,:,i,j]
                x_diff=(x2-x_feat)**2

                a=torch.argmin(x_diff,axi=2)
                b=torch.argmin(x_diff,axi=3)

                target[:,:,a,b]=x2[:,:,a,b]
        catted_feat=torch.cat(x1,target)
        return catted_feat


class FCNs(nn.Module):

    def __init__(self):
        super().__init__()
        #self.n_class = n_class
        #self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        #self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x1,x2):

        output = FEAT(x1,x2)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)
        x2 = output['x2']  # size=(N, 128, x.H/4,  x.W/4)
        x1 = output['x1']  # size=(N, 64, x.H/2,  x.W/2)

        score = self.bn1(self.relu(self.deconv1(x5)))     # size=(N, 512, x.H/16, x.W/16)
        score = score + x4                                # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score)))  # size=(N, 256, x.H/8, x.W/8)
        score = score + x3                                # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = score + x2                                # element-wise add, size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = score + x1                                # element-wise add, size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        #score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score  # size=(N, n_class, x.H/1, x.W/1)



if __name__ == '__main__':
    img1=cv2.imread()
    img2=cv2.imread()

    fcn_model = FCNs()


