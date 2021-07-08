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

        return img1, img2, matrix