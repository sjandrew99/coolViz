#!/usr/bin/env python3

import numpy as np
import cv2
from tqdm import tqdm
from timeit import default_timer as timer
import os
import pickle

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

#TODO - these functions assume normalized [0,1] data
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


imsize = (900, 800)



# q = exp(-j*2*pi*tau)

def get_weierstrass_invariant(da,db, terms, k, dw):
    precomp_file = 'precomp/__weierstrass_da_%.05f_db_%.05f_terms_%03d_k_%02d_dw_%.03f_imsize_%d_%d.pkl' % (da, db, terms, k, dw, imsize[0],imsize[1])
    if os.path.exists(precomp_file):
        with open(precomp_file,'rb') as fp:
            img = pickle.load(fp)
    else:
        img = np.zeros((imsize[1],imsize[0],3),dtype=np.uint8)
        st = timer()
        A = np.arange(0,0.5,da,dtype=np.float32) # G6 is symmetric about the imaginary axis, so we only need 180 degrees of rotation
        #A = np.arange(0,2*np.pi,da)
        B = np.arange(db,1,db,dtype=np.float32) # TODO - is 1 the proper maximum?
        AA, BB = np.meshgrid(A,B)
        Tau = AA + 1j*BB
        Q = np.exp(1j*2*np.pi*Tau)

        worldlims = [[-dw,dw],[-dw,dw]]

        M = np.arange(-terms,terms+1)
        N = np.arange(-terms,terms+1)
        tprecomp = timer() - st

        st = timer()
        Tau2, MM2, NN2 = np.meshgrid(Tau.ravel(), M, N) # nterms x npts x nterms
        x = 1/ ((Tau2 * NN2 + MM2)**(2*k))
        x[terms,:,terms] = 0
        GG2 = np.sum(x, axis=(0,2))
        tcomp = timer() - st
        
        st = timer()
        # vectorized coordinate conversion and color conversion:
        xi = world_coords_to_img_coords(np.real(Q),imsize[0],worldlims[0],invert=False)
        yi = world_coords_to_img_coords(np.imag(Q),imsize[1],worldlims[1],invert=True)
        yi_neg = world_coords_to_img_coords(-np.imag(Q),imsize[1],worldlims[1],invert=True) # exploit the symmetry
        # normalize, log, then normalize again
        data = np.real(GG2.reshape(Tau.shape).T)
        data[np.nonzero(data < 0)] = 0 # wikipedia sets negative values to 0 
        data = data / np.max(data)
        eps = np.min(data[np.nonzero(data)]) 
        data = np.log10(data + eps/10)
        data = data - np.min(data)
        data = data / np.max(data)
        tnorm = timer() - st

        st = timer()
        # plt.plot(10*np.log10(data)) # very cool
        bgr = data_to_bgr(data) # TODO - why doesn't this work?
        #bgr = data_to_bgr(np.real(GG))
        img[yi.T,xi.T,:] = bgr
        img[yi_neg.T,xi.T,:] = bgr # exploit the symmetry
        tdraw = timer() - st        

        print('da: %.05f db: %.05f terms: %03d: k: %02d: dw: %.03f: imsize: %d_%d' % (da, db, terms, k, dw, imsize[0],imsize[1]),end=' ')
        print('tprecomp: %.1f ms, tcomp: %.1f ms, tnorm: %.1f, tdraw: %.1f' % (tprecomp * 1000, tcomp * 1000, tnorm * 1000, tdraw * 1000))
        with open(precomp_file,'wb') as fp:
            pickle.dump(img,fp)
    return img

# these give a very good picture; takes about 5-10 seconds to compute
da = .001 # real part of tau
db = .001 # imag part of tau
terms = 5 # N/M limits
k = 3 # calculating G6
dw = 1.25
"""
# much faster, kinda crappy
da = .01 # real part of tau
db = .01 # imag part of tau
terms = 1 # N/M limits
k = 3 # calculating G6
dw = 1.25
"""
img = get_weierstrass_invariant(da,db,terms,k,dw)

for da in tqdm(np.arange(.0001, .01,.0001)):
    for db in np.arange(.0001, .01, .0001):
        for terms in [1,2,3,4,5]:
            tau_size = (.5/da) * ((1-db)/db) 
            m_size = 2*terms + 1
            #total_size = 3 * (tau_size * m_size * m_size) # for tau, m, and n matrices
            total_size = (tau_size * m_size * m_size) * 8 # 64-bit numbers, tau matrix
            if (total_size/1e9) > 4: continue # limit to 4Gb, ish
            img = get_weierstrass_invariant(da,db,terms,k,dw)

cv2.imshow('real(G6)',img)
cv2.waitKey(0)

"""
import matplotlib.pyplot as plt
plt.ion()
fig,ax = plt.subplots()
Z = np.zeros((imsize[1],imsize[0]))
Z[yi.T,xi.T] = np.real(GG)
"""
