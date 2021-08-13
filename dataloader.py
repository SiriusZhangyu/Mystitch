#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import torch
import torch.utils.data as data
import numpy as np
import cv2


class mydataset(data.Dataset):

    def __init__(self, img1_path, img2_path, mask_path):
        self.img1_path = img1_path
        self.img2_path = img2_path
        self.mask_path =mask_path

    def __len__(self):

        return len(os.listdir(self.img1_path))

    def __getitem__(self, index):
        img_name = os.listdir(self.img1_path)
        img1 = cv2.imread(os.path.join(self.img1_path,img_name[index]))

        img2 = cv2.imread(os.path.join(self.img2_path, img_name[index]))

        mask=cv2.imread(os.path.join(self.mask_path,img_name[index]))

        return img1.float(), img2.float(), mask
