#!/usr/bin/env python3

import numpy as np
import cv2
from tqdm import tqdm

# this script is not correct but still gives a cool animation

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

from matplotlib import cm
def data_to_bgr(data):
    #jet = cm.get_cmap('jet')
    #jet = cm.get_cmap('plasma')
    jet = cm.get_cmap('nipy_spectral')
    rgba = jet(data)
    rgb = np.uint8(255*rgba[:,:,0:3])
    bgr = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    return bgr
def scalar_data_to_bgr(data):
    #jet = cm.get_cmap('jet')
    #jet = cm.get_cmap('plasma')
    jet = cm.get_cmap('nipy_spectral')
    rgba = jet(data)
    rgb = np.uint8(255*np.array(rgba[0:3]))
    bgr = cv2.cvtColor(rgb[None,None,:],cv2.COLOR_RGB2BGR)
    return bgr


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
dr = .005

img = np.zeros((imsize[1],imsize[0],3),dtype=np.uint8)
img_abs = np.zeros((imsize[1],imsize[0],3),dtype=np.uint8)
img_real = np.zeros((imsize[1],imsize[0],3),dtype=np.uint8)
r = np.arange(0,1.5,dr)
for ith in tqdm(range(0,len(theta))):
    z = r * np.exp(1j*theta[ith]) # TODO - precomp the exp term
    x = np.real(z)
    y = np.imag(z)
    xi = world_coords_to_img_coords(x, imsize[1],worldlims[0])
    yi = world_coords_to_img_coords(y, imsize[0],worldlims[1],True)
    tau = y / x
    q = np.exp(1j*np.pi*tau) # nome
    for k in range(0,len(xi)):
        if xi[k] < 0 or xi[k] >= imsize[1] or yi[k] < 0 or yi[k] >= imsize[0]: continue
        img[yi[k],xi[k],:] = 255
        if (not np.isnan(np.real(q[k]))) and (not np.isnan(np.imag(q[k]))):
            img_abs[yi[k],xi[k],:] = scalar_data_to_bgr(np.abs(q[k]))
            img_real[yi[k],xi[k],:] = scalar_data_to_bgr(np.real(q[k]))
    cv2.imshow('disk',img)
    cv2.imshow('abs(q)',img_abs)
    cv2.imshow('real(q)',img_real)
    cv2.waitKey(1)


cv2.imshow('disk',img)
cv2.imshow('abs(q)',img_abs)
cv2.imshow('real(q)',img_real)
cv2.waitKey(0)
