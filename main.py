
import cv2
from skimage import morphology
import numpy as np
from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert

img=cv2.imread('./selected_curve/1548_115.png')

# perform skeletonization
img = skeletonize(img)




cv2.imshow('',img)
cv2.waitKey()


