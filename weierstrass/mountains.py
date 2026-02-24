#!/usr/bin/env python3

import numpy as np
import cv2
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt

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

img = np.zeros((imsize[1],imsize[0],3),dtype=np.uint8)

# q = exp(-j*2*pi*tau)
# these give a very good picture; takes about 5-10 seconds to compute
da = .001 # real part of tau
db = .001 # imag part of tau
terms = 4 # N/M limits
k = 3 # calculating G6
dw = 1.25

# for drawing the mountains, use these, which get close:
db = .01
da = .001
terms = 5
k = 3

"""
# much faster, kinda crappy
da = .01 # real part of tau
db = .01 # imag part of tau
terms = 1 # N/M limits
k = 3 # calculating G6
dw = 1.25
"""

#A = np.arange(0,0.5,da) # G6 is symmetric about the imaginary axis, so we only need 180 degrees of rotation
A = np.arange(0,1,da) # keep the symmetric stuff for the plot of data (the mountains) below
#A = np.arange(0,2*np.pi,da)
B = np.arange(db,1,db) # TODO - is 1 the proper maximum?
AA, BB = np.meshgrid(A,B)
Tau = AA + 1j*BB
Q = np.exp(1j*2*np.pi*Tau)


worldlims = [[-dw,dw],[-dw,dw]]

GG = np.zeros((len(A),len(B)),dtype=np.complex128)
M = np.arange(-terms,terms+1)
N = np.arange(-terms,terms+1)
MM, NN = np.meshgrid(M,N)
#M = np.concatenate((np.arange(-terms,0), np.arange(1,terms+1))) # skip M = 0
k2 = 2 * k
for i,a in tqdm(enumerate(A)):
    break
    for j,b in enumerate(B):
        #G = 0
        #tau = a + 1j*b
        #q = np.exp(1j*2*np.pi*tau); assert np.abs(q) <= 1 # only evaluating on the unit disk
        
        """# commenting out this for loop in favor of the meshgrid approach
        for n in range(-terms,terms+1):

            # omit all the m=0 terms (not just m,n=(0,0)) and you'll get a pretty spiral. but that's not the correct eisenstein series
            if n == 0: continue
            ntau = n*tau 
            G = G + np.sum(1 / ((M + ntau)**(2*k))) 
        """
        #x = 1 / ((tau * NN + MM)**(2*k))
        x = 1 / ((Tau[j,i] * NN + MM)**(k2))
        x[terms,terms] = 0 # the 0,0 element - ignore it in the sum
        G = np.sum(x)
                
        GG[i,j] = G
        """
        continue # see below this double for loop - coord conversion and color conversion can be done in a vectorized manner
        if np.real(G) < 0: continue
        #xi = world_coords_to_img_coords(a,imsize[0],worldlims[0],invert=False)
        #yi = world_coords_to_img_coords(b,imsize[1],worldlims[1],invert=True)
        xi = world_coords_to_img_coords(np.real(q),imsize[0],worldlims[0],invert=False)
        yi = world_coords_to_img_coords(np.imag(q),imsize[1],worldlims[1],invert=True)
        bgr = scalar_data_to_bgr(np.real(G)) # TODO - normalize
        img[yi,xi,:] = bgr
        """
        #break
    #break
Tau2, MM2, NN2 = np.meshgrid(Tau.ravel(), M, N) # nterms x npts x nterms
x = 1/ ((Tau2 * NN2 + MM2)**(k2))
x[terms,:,terms] = 0
GG2 = np.sum(x, axis=(0,2))
#sys.exit(1)
# vectorized coordinate conversion and color conversion:
xi = world_coords_to_img_coords(np.real(Q),imsize[0],worldlims[0],invert=False)
yi = world_coords_to_img_coords(np.imag(Q),imsize[1],worldlims[1],invert=True)
yi_neg = world_coords_to_img_coords(-np.imag(Q),imsize[1],worldlims[1],invert=True) # exploit the symmetry
# normalize, log, then normalize again
#data = np.real(GG)
data = np.real(GG2.reshape(Tau.shape).T)
data[np.nonzero(data < 0)] = 0 # wikipedia sets negative values to 0 
data = data / np.max(data)
eps = np.min(data[np.nonzero(data)]) 
data = np.log10(data + eps/10)
data = data - np.min(data)
data = data / np.max(data)

plt.ion()
plt.plot(10*np.log10(data[:,0:40])) # very cool
bgr = data_to_bgr(data) # TODO - why doesn't this work?
#bgr = data_to_bgr(np.real(GG))
img[yi.T,xi.T,:] = bgr
img[yi_neg.T,xi.T,:] = bgr # exploit the symmetry
        
cv2.imshow('real(G6)',img)
cv2.waitKey(0)

"""
import matplotlib.pyplot as plt
plt.ion()
fig,ax = plt.subplots()
Z = np.zeros((imsize[1],imsize[0]))
Z[yi.T,xi.T] = np.real(GG)
"""
