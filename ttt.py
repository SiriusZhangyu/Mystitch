#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import cv2
import numpy as np
import os
import random
import torch
import torch
import torchvision



model = torchvision.models.resnet50(pretrained=True)

#data = os.listdir('selected_curve')

#name=data[19]
img = cv2.imread('./train_data/resized/124_7.png')
img=torch.from_numpy(img)
img=img.view(3,512,512)

feat=model(img)





