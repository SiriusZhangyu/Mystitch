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
        self.relu=nn.ReLU()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.bn1 = nn.BatchNorm2d(512)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='nearest')
        
    
    def forward(self,x1,x2):

        x11 = self.conv1(x1)
        x12 = self.pool(x11)
        x13 = self.conv2(x12)
        x14 = self.pool(x13)
        x15 = self.conv3(x14)
        x16 = self.pool(x15)
        x17 = self.conv4(x16)
        x18 = self.pool(x17)
        x19 = self.conv5(x18)
        x110 = self.pool(x19)
        x111 = self.pool(x110)


        x21 = self.conv1(x2)
        x22 = self.pool(x21)
        x23 = self.conv2(x22)
        x24 = self.pool(x23)
        x25 = self.conv3(x24)
        x26 = self.pool(x25)
        x27 = self.conv4(x26)
        x28 = self.pool(x27)
        x29 = self.conv5(x28)
        x210 = self.pool(x29)
        x211 = self.pool(x210)

        feat1= torch.cat(x110,x210)
        feat2= torch.cat(x111,x211)

        feat= feat1 + self.bn1(self.relu(self.upsample1(feat2)))


        output=self.bn1(self.relu(self.upsample2(feat)))




        return output



if __name__ == '__main__':
    img1=cv2.imread()
    img2=cv2.imread()

    fcn_model = FCNs()

