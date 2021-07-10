#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from dataset import mydataset
from model import ResNet,ResBlock

torch.cuda.empty_cache()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))


    source_root='./train_data/cut_img' # get data root path
    template_root='./train_data/resized'
    gt_matrix=np.loadtxt(open("./train_data/train.csv","rb"),delimiter=",",skiprows=0)

    train_dataset = mydataset(img1_path=source_root, img2_path=template_root, all_matrix=gt_matrix)

    batch_size = 1
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0)


    net = ResNet(ResBlock)
    net.to(device)


    epochs = 10
    save_path = './mystitch.pth'
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    loss = nn.MSELoss()
    for epoch in range(epochs):
        # train
        net.train()
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            source_image, template_image,gt = data
            optimizer.zero_grad()



            output=net(source_image.to(device),template_image.to(device))

            transform_image=transforms.RandomAffine(
                degrees=output[0],
                translate=(output[1],output[2]),
                scale=None,
                shear=None,
                resample=False,
                fillcolor=0
            )(source_image)


            final_image=cv2.add(transform_image,template_image)
            angle_and_dis_loss=torch.sqrt(torch.sum((output-gt)**2))

            img_loss=loss(final_image,template_image)

            loss=angle_and_dis_loss+img_loss

            loss.backward()
            optimizer.step()


        net.eval()


    print('Finished Training')


if __name__ == '__main__':
    main()