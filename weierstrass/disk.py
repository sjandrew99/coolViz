#!/usr/bin/env python3

import numpy as np
import cv2
from tqdm import tqdm

def world_coords_to_img_coords(xw,imsize,worldlims,invert=False):
    # xw - the coordinate in world space
    # imsize - scalar, the width or height of the image
    # worldlims - array of length 2
    # invert - for computing the y (vertical) coordinate in images
    
    if invert:
        A = imsize / (worldlims[0] - worldlims[1])
        frameTop = 0
        C = frameTop - A*worldlims[1]
    else:
        A = imsize / (worldlims[1] - worldlims[0])
        frameLeft = 0 # inherited from ocv_plot where theis was non-zero
        C = frameLeft - A * worldlims[0]

    xi = A*xw+C

    #return int(xi)
    return xi.astype(int)

imsize = (800, 800)

img = np.zeros((imsize[1],imsize[0],3),dtype=np.uint8)

# periods:
w1 = 1
w2 = 1j

tau = w2 / w1 # half-period ratio
q = np.exp(1j*np.pi*tau) # nome

# iterate q over the unit disk:
dtheta = .01 # radians
ddr = .001
theta = np.arange(0,2*np.pi,dtheta)
dw = 1
worldlims = [[-dw,dw],[-dw,dw]]
maxr = .3
minr = .002
dr = .0001
while 1:
    img = np.zeros((imsize[1],imsize[0],3),dtype=np.uint8)
    r = np.arange(0,1.5,dr)
    for ith in range(0,len(theta)):
        z = r * np.exp(1j*theta[ith]) # TODO - precomp the exp term
        x = np.real(z)
        y = np.imag(z)
        xi = world_coords_to_img_coords(x, imsize[1],worldlims[0])
        yi = world_coords_to_img_coords(y, imsize[0],worldlims[1],True)
        for k in range(0,len(xi)):
            if xi[k] < 0 or xi[k] >= imsize[1] or yi[k] < 0 or yi[k] >= imsize[0]: continue
            img[yi[k],xi[k],:] = 255

    cv2.imshow('disk',img)
    cv2.waitKey(1)
    dr += ddr
    if ddr <= 0 and dr <= minr:
        ddr = -ddr
    elif ddr >= 0 and dr >= maxr:
        ddr = -ddr 
    
    