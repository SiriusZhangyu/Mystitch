import os
import cv2
import math
import json
import random
import numpy as np


def transform_image(img, mask, ang_range, trans_range):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over.

    A Random uniform distribution is used to generate different parameters for transformation

    '''
    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols = img.shape
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = max(trans_range/4, trans_range*np.random.uniform()-trans_range/2)
    tr_y = max(trans_range/4, trans_range*np.random.uniform()-trans_range/2)
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    # pts1 = np.float32([[5,5],[20,5],[5,20]])
    # pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    # pt2 = 20+shear_range*np.random.uniform()-shear_range/2
    # pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])
    # shear_M = cv2.getAffineTransform(pts1,pts2)

    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    # img = cv2.warpAffine(img,shear_M,(cols,rows))

    mask = cv2.warpAffine(mask,Rot_M,(cols,rows))
    mask = cv2.warpAffine(mask,Trans_M,(cols,rows))
    # mask = cv2.warpAffine(mask,shear_M,(cols,rows))

    return img, mask


def tmp1():
    img_dir = '/WD1/SnowVision/exp_sherd2sherd_matching/build_testset/select_sherds'
    mask_dir = '/WD1/SnowVision/exp_sherd2sherd_matching/build_testset/mask'
    for i, img_name in enumerate(os.listdir(img_dir)):
        img = cv2.imread(os.path.join(img_dir, img_name), 0)
        mask = cv2.imread(os.path.join(mask_dir, img_name), 0)

        (h, w) = img.shape
        diag = int(math.ceil(math.sqrt(h*h+w*w)))
        square_img = np.zeros([diag, diag], dtype=np.uint8)
        y0, x0 = int((diag-h)/2), int((diag-w)/2)
        square_img[y0:y0+h, x0:x0+w] = img
        square_mask = np.zeros([diag, diag], dtype=np.uint8)
        square_mask[y0:y0+h, x0:x0+w] = mask

        trans_img, trans_mask = transform_image(square_img, square_mask, ang_range=360, trans_range=diag/2)
        overlap_mask = square_mask.copy()
        overlap_mask[trans_mask==0] = 0

        cv2.imshow('1', square_img)
        cv2.imshow('2', square_mask)
        cv2.imshow('3', trans_img)
        cv2.imshow('4', trans_mask)
        cv2.imshow('5', overlap_mask)
        cv2.waitKey()


def synthesize():
    """
    1. Randomly select two sherds A and B (A, B come from different designs)
    2. Randomly crop a part of A with free boundary (resize the mask A, rotate, translate)
    3. Put the cropped part at a random position on B
    """
    img_dir = '/WD1/SnowVision/exp_sherd2sherd_matching/build_testset/curve'
    mask_dir = '/WD1/SnowVision/exp_sherd2sherd_matching/build_testset/mask'
    save_dir = './sync_data'
    sherd_list = os.listdir('/WD1/SnowVision/exp_sherd2sherd_matching/build_testset/select_sherds')
    sherd_list = [x.split('.')[0] for x in sherd_list]
    design_sherd_dict = json.load(open('/WD1/SnowVision/exp_sherd2sherd_matching/build_testset/design_sherd_name_dict.json'))
    sherd_design_dict = dict()
    for k, v in design_sherd_dict.items():
        for sherd_name in v:
            sherd_design_dict[sherd_name] = k
    k, num = 0, 100
    while k < num:
        print(k, '/', num)
        img1_name = random.choice(sherd_list)
        img1 = cv2.imread(os.path.join(img_dir, img1_name+'.png'), 0)
        mask1 = cv2.imread(os.path.join(mask_dir, img1_name+'.png'), 0)
        img1_cluster = design_sherd_dict[sherd_design_dict[img1_name]]
        img2_candidates = [x for x in sherd_list if x not in img1_cluster]
        img2_name = random.choice(img2_candidates)
        img2 = cv2.imread(os.path.join(img_dir, img2_name+'.png'), 0)
        mask2 = cv2.imread(os.path.join(mask_dir, img2_name+'.png'), 0)

        # gt_mask1 = np.zeros(mask1.shape, dtype=np.uint8)
        # min_side = min(img1.shape[0], img1.shape[1], img2.shape[0], img2.shape[1])
        # radius = np.random.randint(min_side/4, min_side/3)
        # ox = np.random.randint(radius, img1.shape[1]-radius-1)
        # oy = np.random.randint(radius, img1.shape[0]-radius-1)
        # cv2.circle(gt_mask1, (ox, oy), radius, 255, -1)

        gt_mask1 = np.zeros(mask1.shape, dtype=np.uint8)
        min_side = min(img1.shape[0], img1.shape[1], img2.shape[0], img2.shape[1])
        if min_side < 300:
            continue
        # radius_x = np.random.randint(min_side/5, min_side/3)
        # radius_y = np.random.randint(min_side/5, min_side/3)
        radius_x = int(min_side/4)
        radius_y = int(min_side/3)
        radius = max(radius_x, radius_y)
        ox = np.random.randint(radius, img1.shape[1]-radius-1)
        oy = np.random.randint(radius, img1.shape[0]-radius-1)
        angle = np.random.rand() * 180
        cv2.ellipse(gt_mask1, (ox, oy), (radius_x, radius_y), angle, 0, 360, 255, -1)

        img1_fore = img1.copy()
        img1_fore[gt_mask1 == 0] = 0
        if np.sum(img1_fore) / np.sum(gt_mask1) < 0.3:
            continue

        angle = np.random.rand() * 180
        rot_mat = cv2.getRotationMatrix2D((img2.shape[1]/2, img2.shape[0]/2), angle, 1)
        img2_fore = cv2.warpAffine(img1_fore, rot_mat, (img2.shape[1], img2.shape[0]))
        gt_mask2 = cv2.warpAffine(gt_mask1, rot_mat, (img2.shape[1], img2.shape[0]))
        # fore_pixels = np.where(img2_fore != 0)
        # min_x, max_x, min_y, max_y = fore_pixels[0].min(), fore_pixels[0].max(), fore_pixels[1].min(), fore_pixels[1].max()
        dx = np.random.randint(-img2.shape[1]/3, img2.shape[1]/3)
        dy = np.random.randint(-img2.shape[0]/3, img2.shape[0]/3)
        trans_mat = np.float32([[1,0,dx],[0,1,dy]])
        img2_fore = cv2.warpAffine(img2_fore, trans_mat, (img2.shape[1], img2.shape[0]))
        gt_mask2 = cv2.warpAffine(gt_mask2, trans_mat, (img2.shape[1], img2.shape[0]))
        if np.sum(img2_fore != 0) < 15000:
            continue

        img2_overlap = img2.copy()
        img2_overlap[gt_mask2 != 0] = 0
        img2_overlap += img2_fore

        cv2.imshow('img1', img1)
        cv2.imshow('mask1', mask1)
        cv2.imshow('img2', img2)
        cv2.imshow('mask2', mask2)
        cv2.imshow('gt_mask1', gt_mask1)
        cv2.imshow('img1_fore', img1_fore)
        cv2.imshow('gt_mask2', gt_mask2)
        cv2.imshow('img2_fore', img2_fore)
        cv2.imshow('img2_overlap', img2_overlap)
        cv2.waitKey()

        # tmp_save_dir = os.path.join(save_dir, str(k))
        # os.mkdir(tmp_save_dir)
        # cv2.imwrite(os.path.join(tmp_save_dir, 'src.png'), img1)
        # cv2.imwrite(os.path.join(tmp_save_dir, 'tar_orig.png'), img2)
        # cv2.imwrite(os.path.join(tmp_save_dir, 'mask_src.png'), gt_mask1)
        # cv2.imwrite(os.path.join(tmp_save_dir, 'mask_tar.png'), gt_mask2)
        # cv2.imwrite(os.path.join(tmp_save_dir, 'tar.png'), img2_overlap)

        k += 1


if __name__ == '__main__':
    synthesize()
