#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import torch
import torch.utils.data as data
import numpy as np
import cv2


class mydataset(data.Dataset):

    def __init__(self, img1_path, img2_path, all_matrix):
        self.img1_path = img1_path
        self.img2_path = img2_path
        self. all_matrix =all_matrix

    def __len__(self):

        return len(os.listdir(self.img1_path))

    def __getitem__(self, index):
        img_name = os.listdir(self.img1_path)
        img1 = cv2.imread(os.path.join(self.img1_path,img_name[index]))

        img2 = cv2.imread(os.path.join(self.img2_path, img_name[index]))

        matrix=self. all_matrix[index]
        matrix=torch.from_numpy(matrix)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        img1 = img1.view(3,512,512)
        img2 = img2.view(3, 512, 512)


        return img1.float(), img2.float(), matrix