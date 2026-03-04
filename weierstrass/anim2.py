#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os
from pickleLoad import *
import cv2
np.random.seed(0)
import sys
sys.path.insert(0,'/home/steve/projects')
from coolViz.param_stepper import ParamStepper, ParamStepperDesigner

def rotate(frame, theta, scale):
    # theta in degrees
    width = frame.shape[1]
    height = frame.shape[0]
    R = cv2.getRotationMatrix2D((width//2, height//2), theta, scale) 
    return(cv2.warpAffine(frame, R, (width,height)))

def tint(*args):
    return (int(args[0]), int(args[1]))

def gen_grid_background(dx=100,dy=200):
    bgcolor = 30
    background = np.zeros((imsize[1], imsize[0], 3),dtype=np.uint8) + bgcolor
    # draw some "45 degree" lines:
    xstart = 0
    xstop = imsize[0]
    ystart = 0
    ystop = imsize[1]
    #clr = (40,40,40)
    clr = (64,64,64)
    # right:
    while 1:
        cv2.line(background, tint(xstart, ystart), tint(xstop, ystop), clr, 1, cv2.LINE_AA)
        xstart = xstart + dx
        xstop = xstop + dx
        if (xstart > imsize[0]): break
    #left:
    xstart = 0; xstop = imsize[0]
    while 1:
        cv2.line(background, tint(xstart, ystart), tint(xstop, ystop), clr, 1, cv2.LINE_AA)
        xstart = xstart - dx
        xstop = xstop - dx
        
        if (xstop < 0): break
    #down:
    xstart = 0; xstop = 0;
    xstop = imsize[0] # use this for a slanted grid
    ystart = 0; ystop = 0
    while 1:
        cv2.line(background, tint(xstart, ystart), tint(xstop, ystop), clr, 1, cv2.LINE_AA)
        ystart = ystart + dy
        #ystop = ystop + dx
        xstop = xstop + dx
        if (ystart > 2*imsize[1]): break
    return background

def get_frame(da_, db_, terms_):
    idx = np.nonzero((DA == da_)*(DB==db_)*(TERMS==terms_))[0]
    if len(idx) != 1:
        #import pdb; pdb.set_trace()
        assert False
        #sys.exit(1)
    frame = pickleLoad(dr + '/' + files[idx[0]])
    return frame
def get_frame_and_background(da_, db_, terms_):
    frame = get_frame(da_,db_,terms_)
    frame = rotate(frame, params.update('theta'), params.update('scale')); 
    background = gen_grid_background(params.update('dx'),params.update('dy'))
    frame = background + frame
    return frame

def imshow(frame,delay):
    cv2.imshow('img',frame)
    cv2.waitKey(delay)

# da: theta resolution
# db: radial resolution    




#viewer = Viewer()
# da, db, terms
delay = 10

dr = 'precomp'
files = [i for i in os.listdir(dr) if i.endswith('.pkl')]
files.sort()
frame = pickleLoad(dr + '/' + files[0])
imsize = (frame.shape[1],frame.shape[0])
DA = np.array([float(i.split('_')[4]) for i in files])
DB = np.array([float(i.split('_')[6]) for i in files])
TERMS = np.array([int(i.split('_')[8]) for i in files])

uterms = np.unique(TERMS)
pscale = ParamStepperDesigner(.5, 1.8, 100, 'scale')
ptheta = ParamStepperDesigner(0,360, 3600, 'theta',waveform='upramp')
pdx = ParamStepperDesigner(10,200,190,'dx')
pdy = ParamStepperDesigner(10,200,190,'dy')
pterms = ParamStepperDesigner(np.min(uterms),np.max(uterms),len(uterms),'terms')
pdict = {'scale':pscale.params, 'theta':ptheta.params,'dx':pdx.params,'dy':pdy.params,'terms':uterms}
# different da/db lists for each term
for t in uterms:
    idx = np.nonzero(TERMS == t)[0]
    da_ = np.sort(np.unique(DA[idx]))
    db_ = np.sort(np.unique(DB[idx]))
    aname=f'da_terms_{t}'
    bname=f'db_terms_{t}'
    pdict[aname] = ParamStepperDesigner(da_, name=aname).params
    pdict[bname] = ParamStepperDesigner(db_, name=bname).params
    """
    pdict[aname] = da_
    pdict[bname] = db_
    pdict[aname+'_down'] = np.flipud(da_)
    pdict[bname+'_down'] = np.flipud(db_)
    """
params = ParamStepper(pdict)

terms_ = params.update('terms')
aname = f'da_terms_{terms_}'
bname = f'db_terms_{terms_}'
da_ = params.update(aname)
db_ = params.update(bname)

while 1:
    
    # min a, min b. highest res
    aname = f'da_terms_{terms_}'
    bname = f'db_terms_{terms_}'
    while 1:
        if da_ == np.max(params.params[aname]): break
        #import pdb; pdb.set_trace()
        frame = get_frame_and_background(da_,db_,terms_)
        imshow(frame,delay)
        da_ = params.update(aname)
    # max a, min b.
    while 1:
        if db_ == np.max(params.params[bname]): break
        frame = get_frame_and_background(da_,db_,terms_)
        imshow(frame,delay)
        db_ = params.update(bname)

    # max a, max b. lowest res
    lastframe = get_frame(da_,db_,terms_)
    terms_ = params.update('terms')
    
    aname = f'da_terms_{terms_}_down'
    bname = f'db_terms_{terms_}_down'
    da_ = params.update(aname)
    db_ = params.update(bname)
    
    newframe = get_frame(da_,db_,terms_)
    #print(f'turnaround, terms = {viewer.terms[viewer.iFile]}')
    
    # interpolate between frame and newframe:
    lastframe = lastframe.astype(np.float32) # use floating point for increased dynamic range and resolution
    dframe = newframe.astype(np.float32) - lastframe
    nSteps = 15
    dpx = dframe / nSteps
    for step in range(0, nSteps):
        frame = (lastframe + dpx*(step+1)).astype(np.uint8)
        frame = rotate(frame, params.no_update('theta'), params.no_update('scale')); # TODO - continue rotate/scale throughout interp 
        background = gen_grid_background(params.update('dx'),params.update('dy'))
        frame = background + frame
        imshow(frame,delay)
    #print('done interpolating')
    # back to highest res for loop
    while 1:
        if da_ == np.min(params.params[aname]): break
        frame = get_frame_and_background(da_,db_,terms_)
        imshow(frame,delay)
        da_ = params.update(aname)
    while 1:
        if db_ == np.min(params.params[bname]): break
        frame = get_frame_and_background(da_,db_,terms_)
        imshow(frame,delay)
        db_ = params.update(bname)    
    # min a, max b, min terms
    # min a, min b, min terms

