#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import cv2
import numpy as np
import os
import random

#data = os.listdir('selected_curve')

#name=data[19]
img = cv2.imread('./train_data/resized/124_7.png')
img1=cv2.imread('./train_data/resized/131_205.png')
#resized_img = cv2.resize(img, (512, 512))

#w = random.randint(200, 512)
#h = random.randint(200, 512)
#angle = random.randint(-180, 180)

#cut_img = resized_img[0:w, 0:h]

#rotateMatrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
#rotated = cv2.warpAffine(cut_img, rotateMatrix, (w, h))

#image = cv2.copyMakeBorder(rotated, 256-h//2, 256-(h-h//2), 256-w//2, 256-(w-w//2), cv2.BORDER_CONSTANT)

#a=np.loadtxt(open("./train_data/train.csv","rb"),delimiter=",",skiprows=0)

img2=cv2.add(img,img1)

cv2.imshow('',img2)
cv2.waitKey()



