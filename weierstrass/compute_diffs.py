#!/usr/bin/env python3
import os
import numpy as np
from pickleLoad import *
from tqdm import tqdm 
import cv2

#dr = 'precomp/terms5'
dr = 'precomp'

f1 = '__weierstrass_da_0.00030_db_0.00010_terms_002_k_03_dw_1.250_imsize_900_800.pkl'
f2 = '__weierstrass_da_0.00020_db_0.00030_terms_002_k_03_dw_1.250_imsize_900_800.pkl'

x = pickleLoad(dr+'/'+f1)
y = pickleLoad(dr + '/' + f2)
d = np.abs(x-y)
cv2.imshow('x',x)
cv2.imshow('y',y)
cv2.imshow('d',d)
cv2.waitKey(0)
th = 1
imsize = (x.shape[1],x.shape[0])
imcenter = (imsize[0]//2, imsize[1]//2)

while 1:
    rotation_matrix = cv2.getRotationMatrix2D(imcenter, th, 1)
    d_ = cv2.warpAffine(d, rotation_matrix, imsize) # continously applying this function to the same image results in blurring
    cv2.imshow('d',d_)
    cv2.waitKey(10)
    th += 1

"""
files = [i for i in os.listdir(dr) if i.endswith('.pkl')]
#files.sort()
for i in range(0,len(files),1):
    x = pickleLoad(dr + '/' + files[i])
    cv2.imshow('img',x)
    cv2.waitKey(10)
"""    
"""
#files = files[0:4] # DEBUG
diffs = np.zeros((len(files),len(files)))
for i in tqdm(range(0,len(files))):
    x = pickleLoad(dr + '/' + files[i])
    for j in range(0,i):
        y = pickleLoad(dr + '/' + files[j])
        diffs[i,j] = np.mean(np.abs(x - y))
        diffs[j,i] = diffs[i,j]
        #break
    #if i > 1: break
    #break
"""