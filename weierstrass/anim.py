#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os
from pickleLoad import *
import cv2
np.random.seed(0)

class Viewer:
    def __init__(self):
        self.dr = 'precomp'
        self.files = [i for i in os.listdir(self.dr) if i.endswith('.pkl')]
        self.files.sort()
        self.da = np.array([float(i.split('_')[4]) for i in self.files])
        self.db = np.array([float(i.split('_')[6]) for i in self.files])
        self.terms = np.array([int(i.split('_')[8]) for i in self.files])
        
        # TODO - allow these to change:
        self.imsize = (900,800)
        self.k = 3
        
        self.iFile = 0
        
    def draw(self):
        #self.fig.suptitle('da: %.05f db: %.05f terms: %03d' % (self.da[self.iFile],self.db[self.iFile],self.terms[self.iFile]))
        x = pickleLoad(self.dr + '/' + self.files[self.iFile])
        #cv2.imshow('img',x)
        #cv2.waitKey(1)
        return x
    
    def decrement_a(self):
        da_ = self.da[self.iFile]
        db_ = self.db[self.iFile]
        terms_ = self.terms[self.iFile]
        iPos = np.nonzero((self.db == db_) * (self.terms == terms_))[0] # TODO - check if empty
        #print(iPos)
        iLess = np.nonzero(self.da[iPos] < da_)[0] # TODO - check if empty
        if len(iLess) == 0:
            return False, None
        #print(iLess)
        idx = np.argmax(self.da[iPos[iLess]])
        self.iFile = iPos[iLess][idx]
        frame = self.draw()
        return True, frame
        
    def increment_a(self):
        da_ = self.da[self.iFile]
        db_ = self.db[self.iFile]
        terms_ = self.terms[self.iFile]
        iPos = np.nonzero((self.db == db_) * (self.terms == terms_))[0] # TODO - check if empty
        iGreater = np.nonzero(self.da[iPos] > da_)[0] # TODO - check if empty
        if len(iGreater) == 0:
            return False, None
        idx = np.argmin(self.da[iPos[iGreater]])
        self.iFile = iPos[iGreater][idx]
        frame = self.draw()
        return True, frame
        
    def decrement_b(self):
        da_ = self.da[self.iFile]
        db_ = self.db[self.iFile]
        terms_ = self.terms[self.iFile]
        iPos = np.nonzero((self.da == da_) * (self.terms == terms_))[0] # TODO - check if empty
        iLess = np.nonzero(self.db[iPos] < db_)[0] # TODO - check if empty
        if len(iLess) == 0:
            return False, None
        idx = np.argmax(self.db[iPos[iLess]])
        self.iFile = iPos[iLess][idx]
        frame = self.draw()
        return True, frame
        
    def increment_b(self):
        da_ = self.da[self.iFile]
        db_ = self.db[self.iFile]
        terms_ = self.terms[self.iFile]
        iPos = np.nonzero((self.da == da_) * (self.terms == terms_))[0] # TODO - check if empty
        iGreater = np.nonzero(self.db[iPos] > db_)[0] # TODO - check if empty
        if len(iGreater) == 0:
            return False, None
        idx = np.argmin(self.db[iPos[iGreater]])
        self.iFile = iPos[iGreater][idx]
        frame = self.draw()
        return True, frame
    
    """    
    def decrement_terms(self):
        da_ = self.da[self.iFile]
        db_ = self.db[self.iFile]
        terms_ = self.terms[self.iFile]
        iPos = np.nonzero((self.da == da_) * (self.db == db_))[0] # TODO - check if empty
        iLess = np.nonzero(self.terms[iPos] < terms_)[0] # TODO - check if empty
        if len(iLess) == 0:
            return -1, None
        idx = np.argmax(self.terms[iPos[iLess]])
        self.iFile = iPos[iLess][idx]
        self.draw()
        return 0
    
    def increment_terms(self):
        da_ = self.da[self.iFile]
        db_ = self.db[self.iFile]
        terms_ = self.terms[self.iFile]
        iPos = np.nonzero((self.da == da_) * (self.db == db_))[0] # TODO - check if empty
        iGreater = np.nonzero(self.terms[iPos] > terms_)[0] # TODO - check if empty
        if len(iGreater) == 0:
            return -1
        idx = np.argmin(self.terms[iPos[iGreater]])
        self.iFile = iPos[iGreater][idx]
        self.draw()
        return 0
    """
    
    def set_terms(self, terms):
        # inc or dec terms, find the minimum da and db
        iPos = np.nonzero((self.terms == terms))[0] # TODO - check if empty
        #self.iFile = iPos[0] # the first element of iPos is the min of da/db
        self.iFile = iPos[-1] # the last element of iPos is the max of da/db (lowest resolution)
        frame = self.draw()
        return True, frame
    
    def draw_cv(self):
        x = pickleLoad(self.dr + '/' + self.files[self.iFile])
        cv2.imshow('img',x)
        cv2.waitKey(1)
    
    
#name_str = '__weierstrass_da_%.05f_db_%.05f_terms_%03d_k_%02d_dw_%.03f_imsize_%d_%d.pkl'

dds = 1
scale = .5
def inc_scale():
    global scale,dds
    ds = .001*dds
    mins = .5
    maxs = 1.8
    scale += ds
    if scale >= maxs:
        dds = -1
    elif scale <= mins:
        dds = 1


def rotate(frame, theta):
    # theta in degrees
    #scale = 1.4
    R = cv2.getRotationMatrix2D((viewer.imsize[0]//2, viewer.imsize[1]//2), theta, scale) 
    return(cv2.warpAffine(frame, R, viewer.imsize))

def inc_theta():
    global theta
    theta += dth
    if theta >= 360:
        theta = 0


viewer = Viewer()
# da, db, terms
dterm = 1 # increment terms
delay = 10
dth = .1 # degrees
theta = 0


"""
# draw some cracks
nlines = 10; npts = 20; 
dpt = 20 # pixels
line_start_x = np.random.randint(0, viewer.imsize[0], nlines)
line_start_y = np.random.randint(0, viewer.imsize[1], nlines)
line_stop_x = np.random.randint(0, viewer.imsize[0], nlines)
line_stop_y = np.random.randint(0, viewer.imsize[1], nlines)
#line_colors = [(0,0,0),(255,255,255)]
line_colors = [(128,128,128)]
for iline in range(0, nlines):
    iclr = np.random.randint(0, len(line_colors))
    clr = line_colors[iclr]
    x = np.linspace(line_start_x[iline], line_stop_x[iline], npts)
    y = np.linspace(line_start_y[iline], line_stop_y[iline], npts)
    
    x = x + np.random.random(npts) * dpt - (dpt/2) # randomly perturb points
    y = y + np.random.random(npts) * dpt - (dpt/2) 
    
    pts = np.array([x,y]).T.reshape((-1,1,2)).astype(np.int32) # N x 1 x 2
    cv2.polylines(background, [pts], False, clr, 1,cv2.LINE_AA)
    #break
"""
def gen_grid_background(dx=100,dy=200):
    global background
    bgcolor = 30
    background = np.zeros((viewer.imsize[1], viewer.imsize[0], 3),dtype=np.uint8) + bgcolor
    # draw some "45 degree" lines:
    xstart = 0
    xstop = viewer.imsize[0]
    ystart = 0
    ystop = viewer.imsize[1]
    #clr = (40,40,40)
    clr = (64,64,64)
    # right:
    while 1:
        cv2.line(background, (xstart, ystart), (xstop, ystop), clr, 1, cv2.LINE_AA)
        xstart = xstart + dx
        xstop = xstop + dx
        if (xstart > viewer.imsize[0]): break
    #left:
    xstart = 0; xstop = viewer.imsize[0]
    while 1:
        cv2.line(background, (xstart, ystart), (xstop, ystop), clr, 1, cv2.LINE_AA)
        xstart = xstart - dx
        xstop = xstop - dx
        
        if (xstop < 0): break
    #down:
    xstart = 0; xstop = 0;
    xstop = viewer.imsize[0] # use this for a slanted grid
    ystart = 0; ystop = 0
    while 1:
        cv2.line(background, (xstart, ystart), (xstop, ystop), clr, 1, cv2.LINE_AA)
        ystart = ystart + dy
        #ystop = ystop + dx
        xstop = xstop + dx
        if (ystart > 2*viewer.imsize[1]): break

mindx = 10; maxdx = 200
mindy = 10; maxdy = 200
dx = 100; dy = 100; dirdx = 1; dirdy = 1
def inc_background():
    global mindx, maxdx, mindy, maxdy, dirdx, dirdy, dx, dy
    
    dx += (1 * dirdx)
    dy += (1 * dirdy)
    if (dx >= maxdx):
        dirdx = -1
    if (dx <= mindx):
        dirdx = 1
    if (dy >= maxdy):
        dirdy = -1
    if (dy <= mindy):
        dirdy = 1
    
    gen_grid_background(dx,dy)


while 1:
    # min a, min b. highest res
    for i in viewer.da:
        r,frame =viewer.increment_a()
        if not r: break
        inc_background()
        frame = rotate(frame, theta); inc_theta(); inc_scale(); frame = background + frame
        cv2.imshow('img',frame)
        cv2.waitKey(delay)
    #import pdb; pdb.set_trace()
    # max a, min b.
    for i in viewer.db:
        r, frame = viewer.increment_b()
        if not r: break
        inc_background()
        frame = rotate(frame, theta); inc_theta(); inc_scale(); frame = background + frame
        cv2.imshow('img',frame)
        cv2.waitKey(delay)
    """
    # max a, max b. lowest res    
    for i in viewer.da:
        if viewer.decrement_a():
            break
        cv2.waitKey(delay)
    # min a, max b, min terms
    for i in viewer.db:
        if viewer.decrement_b():
            break
        cv2.waitKey(delay)
    # min a, min b, highest res
    # back to lowest res for terms transition:
    for i in viewer.da:
        if viewer.increment_a():
            break
        cv2.waitKey(delay)
    # max a, min b
    for i in viewer.db:
        if viewer.increment_b():
            break
        cv2.waitKey(delay)
    # max a, max b. lowest res"""
        
    terms = viewer.terms[viewer.iFile]
    if terms == np.max(viewer.terms):
        dterm = -1
    elif terms == np.min(viewer.terms):
        dterm = 1
    lastframe = viewer.draw()
    r, newframe = viewer.set_terms(terms + dterm)
    #print(f'turnaround, terms = {viewer.terms[viewer.iFile]}')
    
    # interpolate between frame and newframe:
    lastframe = lastframe.astype(np.float32) # use floating point for increased dynamic range and resolution
    dframe = newframe.astype(np.float32) - lastframe
    nSteps = 15
    dpx = dframe / nSteps
    for step in range(0, nSteps):
        frame = (lastframe + dpx*(step+1)).astype(np.uint8)
        #cv2.imshow('img',lastframe.astype(np.uint8))
        inc_background()
        frame = rotate(frame, theta); inc_theta(); inc_scale(); frame = background + frame
        cv2.imshow('img',frame)
        cv2.waitKey(delay)
    #print('done interpolating')
    # back to highest res for loop
    for i in viewer.da:
        r, frame = viewer.decrement_a()
        if not r:    break
        inc_background()
        frame = rotate(frame, theta); inc_theta(); inc_scale(); frame = background + frame
        cv2.imshow('img',frame)
        cv2.waitKey(delay)
    # min a, max b, min terms
    for i in viewer.db:
        r, frame = viewer.decrement_b()
        if not r:    break
        inc_background()
        frame = rotate(frame, theta); inc_theta(); inc_scale(); frame = background + frame
        cv2.imshow('img',frame)
        cv2.waitKey(delay)
    # min a, min b, min terms

