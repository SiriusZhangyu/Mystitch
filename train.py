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
from newidea import FEAT

torch.cuda.empty_cache()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))


    source_root='./'
    template_root='./'
    mask_root='./'


    train_dataset = mydataset(img1_path=source_root, img2_path=template_root, mask_path=mask_root)

    batch_size = 16
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0)


    net = FEAT()
    net.to(device)


    epochs = 50
    save_path = 'mystitch.pth'
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    loss = nn.MSELoss()
    for epoch in range(epochs):
        # train
        net.train()
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            source_image, template_image,mask = data
            optimizer.zero_grad()



            output=net(source_image.to(device),template_image.to(device))

            loss=loss(mask,output)

            loss.backward()
            optimizer.step()


        net.eval()
    torch.save(net, save_path)


    print('Finished Training')


if __name__ == '__main__':
    main()
