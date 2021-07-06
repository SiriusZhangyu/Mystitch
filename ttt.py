#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import cv2
import numpy as np
import os
import random

data = os.listdir('selected_curve')

name=data[19]
img = cv2.imread(os.path.join('./selected_curve', name))
resized_img = cv2.resize(img, (512, 512))

w = random.randint(200, 512)
h = random.randint(200, 512)
angle = random.randint(-180, 180)

cut_img = resized_img[0:w, 0:h]

rotateMatrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
rotated = cv2.warpAffine(cut_img, rotateMatrix, (w, h))

image = cv2.copyMakeBorder(rotated, 256-h//2, 256-(h-h//2), 256-w//2, 256-(w-w//2), cv2.BORDER_CONSTANT)

print(image.shape[0])
print(image.shape[1])



cv2.imshow('', image)
cv2.waitKey()



