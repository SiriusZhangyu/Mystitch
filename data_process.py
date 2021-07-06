#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import cv2
import random

import numpy as np

data=os.listdir('selected_curve')
matrix=[]
for name in data:
    cut=[]
    img=cv2.imread(os.path.join('./selected_curve',name))
    resized_img=cv2.resize(img,(512,512))

    w=random.randint(256, 512)
    h=random.randint(256,512)
    angle=random.randint(-180,180)
    cut.append(w)
    cut.append(h)
    cut.append(angle)
    cut_img=resized_img[0:w, 0:h]

    rotateMatrix = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated = cv2.warpAffine(cut_img, rotateMatrix, (w, h))

    image = cv2.copyMakeBorder(rotated, 256 - h // 2, 256 - (h - h // 2), 256 - w // 2, 256 - (w - w // 2), cv2.BORDER_CONSTANT)

    cv2.imwrite(os.path.join('./cut_img', name), image)

    #cv2.imwrite(os.path.join('./resized',name), resized_img)
    matrix.append(cut)

np.savetxt("train.csv", matrix , delimiter=",",fmt='%d')
