#!/usr/bin/env python3

import cv2
import numpy as np

# all functions assume the images are the same size and depth

def imMix(im1,im2,mix):
    return ((mix*im1) + ((1-mix)*im2)).astype(np.uint8)

class VaryingMixer:
    # gimme two videos and i will mix them with smoothly-varying weights
    def __init__(self,mixmin=.2,mixmax=.8,dmix=.01,update_interval=5):
        self.mixmin = mixmin
        self.mixmax = mixmax
        self.dmix = dmix
        self.mix_ = mixmin
        self.counter = 0
        self.update_interval = update_interval
    def mix(self,im1,im2):
        imgx = imMix(im1,im2,self.mix_)
        if self.counter > self.update_interval:
            self.mix_ = self.mix_ + self.dmix
            if self.mix_ >= self.mixmax:
                self.dmix = -self.dmix
                self.mix_ = self.mixmax
            elif self.mix_ <= self.mixmin:
                self.dmix = -self.dmix
                self.mix_ =  self.mixmin
            self.counter = 0
        else:
            self.counter += 1
        return imgx

def vignette_mix(im1,im2,vignette_size):
    imsize = (im1.shape[0],im1.shape[1])
    x = np.zeros((imsize),dtype=np.uint8)
    center_x = int(imsize[1]/2); center_y = int(imsize[0]/2)
    rad = np.sqrt(imsize[0]*imsize[1]/np.pi)
    rad = rad * vignette_size
    cv2.circle(x,(center_x,center_y),int(rad),255,-1)
    img_x = np.zeros(im1.shape,dtype=np.uint8)
    # outside:
    zeros = np.nonzero(x == 0)
    img_x[zeros[0],zeros[1],:] = im1[zeros[0],zeros[1],:]
    # inside:
    ones = np.nonzero(x)
    img_x[ones[0],ones[1],:] = im2[ones[0],ones[1],:] # inside is all frame        
    return img_x

def vignette_mix2(im1,im2,vignette_size,mix):
    # same as vignette, but applies the "naive" mixing inside
    imsize = (im1.shape[0],im1.shape[1])
    x = np.zeros((imsize),dtype=np.uint8)
    center_x = int(imsize[1]/2); center_y = int(imsize[0]/2)
    rad = np.sqrt(imsize[0]*imsize[1]/np.pi)
    rad = rad * vignette_size
    cv2.circle(x,(center_x,center_y),int(rad),255,-1,cv2.LINE_AA)
    img_x = np.zeros(im1.shape,dtype=np.uint8)
    # outside:
    zeros = np.nonzero(x == 0)
    img_x[zeros[0],zeros[1],:] = im1[zeros[0],zeros[1],:]
    # inside:
    ones = np.nonzero(x)
    img_x[ones[0],ones[1],:] = ((mix*im1[ones[0],ones[1],:]) + ((1-mix)*im2[ones[0],ones[1],:])).astype(np.uint8)
    return img_x

def vignette_mix3(im1,im2,mask_zeros,mask_ones,mix):
    # same as vignette2, but uses a pre-computed mask
    img_x = np.zeros(im1.shape,dtype=np.uint8)
    # outside:
    img_x[mask_zeros[0],mask_zeros[1],:] = im1[mask_zeros[0],mask_zeros[1],:]
    # inside:
    img_x[mask_ones[0],mask_ones[1],:] = ((mix*im1[mask_ones[0],mask_ones[1],:]) + ((1-mix)*im2[mask_ones[0],mask_ones[1],:])).astype(np.uint8)
    return img_x

def vignette_mix4(im1,im2,mask_zeros,mask_ones,mix):
    # same as vignette3, but modifies im1
    #img_x = np.zeros(im1.shape,dtype=np.uint8)
    # outside:
    #img_x[mask_zeros[0],mask_zeros[1],:] = im1[mask_zeros[0],mask_zeros[1],:]
    # inside:
    im1[mask_ones[0],mask_ones[1],:] = ((mix*im1[mask_ones[0],mask_ones[1],:]) + ((1-mix)*im2[mask_ones[0],mask_ones[1],:])).astype(np.uint8)
    return im1
    
def vignette_mix5(im1,im2,mask_zeros,mask_ones,mix):
    img_x = ((mask_ones * im2) + (mask_zeros * im1)).astype(np.uint8)
    return img_x

class VaryingVignetteMixer:
    def __init__(self,minsz,maxsz,dsz,szUpdate,mixmin,mixmax,dmix,mixUpdate):
        self.minsz = minsz
        self.maxsz = maxsz
        self.dsz = dsz
        self.szUpdate = szUpdate
        self.mixmin = mixmin
        self.mixmax = mixmax
        self.dmix=dmix
        self.mixUpdate = mixUpdate
        
        self.mixCounter = 0
        self.szCounter = 0
        
        self.sz = minsz
        self.mix_ = mixmin
        
    def mix(self,im1,im2):
        img_x = vignette_mix2(im1,im2,self.sz,self.mix_)
        
        if self.mixCounter > self.mixUpdate:
            self.mix_ = self.mix_ + self.dmix
            if self.mix_ >= self.mixmax:
                self.dmix = -self.dmix
                self.mix_ = self.mixmax
            elif self.mix_ <= self.mixmin:
                self.dmix = -self.dmix
                self.mix_ =  self.mixmin
            self.mixCounter = 0
        else:
            self.mixCounter += 1        

        if self.szCounter > self.szUpdate:
            self.sz = self.sz + self.dsz
            if self.sz >= self.maxsz:
                self.dsz = -self.dsz
                self.sz = self.maxsz
            elif self.sz <= self.minsz:
                self.dsz = -self.dsz
                self.sz = self.minsz
            self.szCounter = 0
        else:
            self.szCounter += 1
        return img_x

class VaryingVignetteMixerFixedSz:
    def __init__(self,sz,mixmin,mixmax,dmix,mixUpdate):
        self.mixmin = mixmin
        self.mixmax = mixmax
        self.dmix=dmix
        self.mixUpdate = mixUpdate
        
        self.mixCounter = 0
        
        self.sz = sz
        self.mix_ = mixmin
        

        self.mask_zeros = None
        self.mask_ones = None
        
    def mix(self,im1,im2):
        if self.mask_zeros is None:
            imsize = (im1.shape[0],im1.shape[1])
            x = np.zeros((imsize),dtype=np.uint8)
            center_x = int(imsize[1]/2); center_y = int(imsize[0]/2)
            rad = np.sqrt(imsize[0]*imsize[1]/np.pi)
            rad = rad * self.sz
            cv2.circle(x,(center_x,center_y),int(rad),255,-1,cv2.LINE_AA)
            self.mask_zeros = np.nonzero(x ==0)
            self.mask_ones = np.nonzero(x)
        #img_x = vignette_mix2(im1,im2,self.sz,self.mix_)
        #img_x = vignette_mix3(im1,im2,self.mask_zeros,self.mask_ones,self.mix_)
        img_x = vignette_mix4(im1,im2,self.mask_zeros,self.mask_ones,self.mix_)
        #img_x = vignette_mix5(im1,im2,self.mask_zeros,self.mask_ones,self.mix_)
        
        if self.mixCounter > self.mixUpdate:
            self.mix_ = self.mix_ + self.dmix
            if self.mix_ >= self.mixmax:
                self.dmix = -self.dmix
                self.mix_ = self.mixmax
            elif self.mix_ <= self.mixmin:
                self.dmix = -self.dmix
                self.mix_ =  self.mixmin
            self.mixCounter = 0
        else:

            self.mixCounter += 1        

        return img_x

class VaryingGaussianMixerFixedSz:
    def __init__(self,sz,mixmin,mixmax,dmix,mixUpdate):
        self.mixmin = mixmin
        self.mixmax = mixmax
        self.dmix=dmix
        self.mixUpdate = mixUpdate
        
        self.mixCounter = 0
        
        self.sz = sz
        self.mix_ = mixmin
        

        self.mask_zeros = None
        self.mask_ones = None
        
    def mix(self,im1,im2):
        if self.mask_zeros is None:
            imsize = (im1.shape[0],im1.shape[1])
            xArr = np.arange(0,imsize[0])
            yArr = np.arange(0,imsize[1])
            grid = np.meshgrid(xArr,yArr)
            
            x0 = int(imsize[0])/2
            y0 = int(imsize[1])/2

            rad = 200 # TODO - relate this to sz
            sigx = 2*(rad**2); sigy = 2*(rad**2)

            x_ = ((grid[0]-x0)**2)/sigx
            y_ = ((grid[1]-y0)**2)/sigy
            self.mask_ones = np.repeat(np.exp(-(x_+y_))[:,:,None],3,2)
            self.mask_zeros = 1 - self.mask_ones

        #img_x = vignette_mix2(im1,im2,self.sz,self.mix_)
        #img_x = vignette_mix3(im1,im2,self.mask_zeros,self.mask_ones,self.mix_)
        #img_x = vignette_mix4(im1,im2,self.mask_zeros,self.mask_ones,self.mix_)
        img_x = ((self.mask_ones * im2) + (self.mask_zeros * im1)).astype(np.uint8)
        
        if self.mixCounter > self.mixUpdate:
            self.mix_ = self.mix_ + self.dmix
            if self.mix_ >= self.mixmax:
                self.dmix = -self.dmix
                self.mix_ = self.mixmax
            elif self.mix_ <= self.mixmin:
                self.dmix = -self.dmix
                self.mix_ =  self.mixmin
            self.mixCounter = 0
        else:
            self.mixCounter += 1        

        return img_x

class FixedVignette:
    def __init__(self, sz, mix):
        self.sz = sz
        self.mix_ = mix
        self.foreground_scale = None
        self.background_scale = None
    def mix(self,background,foreground):
        if self.foreground_scale is None:
            # create masks:
            imsize = (foreground.shape[0],foreground.shape[1]) 
            x = np.zeros((imsize),dtype=np.uint8)
            center_x = int(imsize[1]/2); center_y = int(imsize[0]/2)
            rad = np.sqrt(imsize[0]*imsize[1]/np.pi)
            rad = rad * self.sz
            cv2.circle(x,(center_x,center_y),int(rad),255,-1,cv2.LINE_AA)
            mask_zeros = np.nonzero(x ==0)
            mask_ones = np.nonzero(x)
            self.foreground_scale = np.zeros(x.shape)
            self.foreground_scale[mask_ones[0],mask_ones[1]] = 1-self.mix_
            #self.foreground_scale[mask_zeros[0],mask_zeros[1]] = self.mix_ # comment this out to have zero foreground outside
            self.foreground_scale = np.repeat(self.foreground_scale[:,:,None],3,2)
            self.background_scale = np.zeros(x.shape)
            self.background_scale[mask_zeros[0],mask_zeros[1]] = 1-self.mix_
            #self.background_scale[mask_ones[0],mask_ones[1]] = self.mix_
            self.background_scale = np.repeat(self.background_scale[:,:,None],3,2)
        img_x = ((self.foreground_scale * foreground) + (self.background_scale * background)).astype(np.uint8)
        return img_x

class FixedVignette2:
    # assumes that the background has already been sufficiently vignetted
    def __init__(self, sz, mix):
        self.sz = sz
        self.mix_ = mix
        self.foreground_scale = None
        self.background_scale = None
    def mix(self,background,foreground):
        if self.foreground_scale is None:
            # create masks:
            imsize = (foreground.shape[0],foreground.shape[1]) # NOTE - height, width 
            x = np.zeros((imsize),dtype=np.uint8)
            center_x = int(imsize[1]/2); center_y = int(imsize[0]/2)
            rad = np.sqrt(imsize[0]*imsize[1]/np.pi)
            rad = rad * self.sz
            cv2.circle(x,(center_x,center_y),int(rad),255,-1,cv2.LINE_AA)
            mask_zeros = np.nonzero(x ==0)
            mask_ones = np.nonzero(x)
            self.foreground_scale = np.zeros(x.shape)
            self.foreground_scale[mask_ones[0],mask_ones[1]] = 1-self.mix_
            #self.foreground_scale[mask_zeros[0],mask_zeros[1]] = self.mix_ # comment this out to have zero foreground outside
            self.foreground_scale = np.repeat(self.foreground_scale[:,:,None],3,2)
        img_x = ((self.foreground_scale * foreground) + background).astype(np.uint8)
        return img_x


if __name__=="__main__":
    cap = cv2.VideoCapture(0)
    frameCount = 0
    from anims.videoAnimReader import VideoAnimReader
    reader = VideoAnimReader('array_640_480_5.mp4')
    vignette_sz = 0.7
    mixmin=.05
    mixer = VaryingVignetteMixerFixedSz(sz=vignette_sz,mixmin=mixmin,mixmax=.2,dmix=.01,mixUpdate=5)
    from timeit import default_timer as timer
    import sys
    t = []
    while frameCount < 50:
        r,frame= cap.read()
        bg = reader.nextframe()
        st = timer()
        img_x = mixer.mix(bg,frame)
        t.append(timer() - st)
        cv2.imshow('stuff',img_x)
        cv2.waitKey(1)
        frameCount += 1
    print('average %.05f ms to mix' % (np.mean(t)*1000))
    reader.close()
    # maybe faster mixer:
    reader = VideoAnimReader('array_640_480_5.mp4')
    t = []
    frameCount = 0
    """
    # create masks:
    imsize = (frame.shape[0],frame.shape[1]) 
    x = np.zeros((imsize),dtype=np.uint8)
    center_x = int(imsize[1]/2); center_y = int(imsize[0]/2)
    rad = np.sqrt(imsize[0]*imsize[1]/np.pi)
    rad = rad * vignette_sz
    cv2.circle(x,(center_x,center_y),int(rad),255,-1,cv2.LINE_AA)
    mask_zeros = np.nonzero(x ==0)
    mask_ones = np.nonzero(x)
    foreground_scale = np.zeros(x.shape)
    foreground_scale[mask_ones[0],mask_ones[1]] = 1-mixmin
    #foreground_scale[mask_zeros[0],mask_zeros[1]] = mixmin # comment this out to have zero foreground outside
    foreground_scale = np.repeat(foreground_scale[:,:,None],3,2)
    background_scale = np.zeros(x.shape)    
    background_scale[mask_zeros[0],mask_zeros[1]] = 1-mixmin
    #background_scale[mask_ones[0],mask_ones[1]] = mixmin 
    background_scale = np.repeat(background_scale[:,:,None],3,2)
    """
    mixer = FixedVignette(vignette_sz, mixmin)
    while frameCount < 50:
        r,frame= cap.read()
        bg = reader.nextframe()
        st = timer()
        #img_x = ((foreground_scale * frame) + (background_scale * bg)).astype(np.uint8)
        #img_x = ((foreground_scale * frame) + bg).astype(np.uint8) # about 4ms to multiply, 2ms to cast, 3ms to add background
        img_x = mixer.mix(bg,frame)
        t.append(timer() - st)
        cv2.imshow('stuff',img_x)
        cv2.waitKey(1)
        frameCount += 1
    print('average %.05f ms to mix' % (np.mean(t)*1000))
 
    # even faster mixer with a pre-windowed background:
    reader = VideoAnimReader('array_640_480_5_v0.7.mp4')
    t = []
    frameCount = 0
    mixer = FixedVignette2(vignette_sz, mixmin)
    while frameCount < 50:
        r,frame= cap.read()
        bg = reader.nextframe()
        st = timer()
        #img_x = ((foreground_scale * frame) + (background_scale * bg)).astype(np.uint8)
        #img_x = ((foreground_scale * frame) + bg).astype(np.uint8) # about 4ms to multiply, 2ms to cast, 3ms to add background
        img_x = mixer.mix(bg,frame)
        t.append(timer() - st)
        cv2.imshow('stuff',img_x)
        cv2.waitKey(1)
        frameCount += 1
    print('average %.05f ms to mix' % (np.mean(t)*1000))
 

    sys.exit(1)

 
    # main:
    """cap = cv2.VideoCapture(0)
    r,frame = cap.read()
    cap.release()
    cv2.imwrite('frame.png',frame)
    """
    imsize = (800,800)
    frame = cv2.imread('frame.png')
    frame = cv2.resize(frame,imsize)

    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame = np.repeat(frame[:,:,None],3,2)
    #img = np.zeros((imsize[1],imsize[0],3),dtype=np.uint8)

    from hexArray import HexagonalArray

    array = HexagonalArray(maxdx_=10,maxdy_=10)
    response = array.calc_and_draw_response(imsize)

    # naive mix:
    mix = 0.9
    img_x = ((mix*frame) + ((1-mix)*response)).astype(np.uint8)

    """
    # mix with a mask (sucks):
    x = np.zeros((imsize[0]*imsize[1]),dtype=np.uint8)
    x[0::2] = 1
    x = np.reshape(x,(imsize[1],imsize[0]))
    x = np.repeat(x[:,:,None],3,2)
    y = np.zeros((imsize[0]*imsize[1]),dtype=np.uint8)
    y[1::2] = 1
    y = np.reshape(y,(imsize[1],imsize[0]))
    y = np.repeat(y[:,:,None],3,2)
    img_x = (x*frame) + (y*response)
    """

    # mix with a vignette:
    x = np.zeros((imsize),dtype=np.uint8)
    center_x = int(imsize[0]/2); center_y = int(imsize[1]/2)
    rad = np.sqrt(imsize[0]*imsize[1]/np.pi)
    rad = rad / 3
    cv2.circle(x,(center_x,center_y),int(rad),255,-1)
    img_x = np.zeros(frame.shape,dtype=np.uint8)
    # outside:
    zeros = np.nonzero(x == 0)
    img_x[zeros[0],zeros[1],:] = response[zeros[0],zeros[1],:]
    # inside:
    ones = np.nonzero(x)
    #img_x[ones[0],ones[1],:] = frame[ones[0],ones[1],:] # inside is all frame
    img_x[ones[0],ones[1],:] = ((mix*frame[ones[0],ones[1],:]) + ((1-mix)*response[ones[0],ones[1],:])).astype(np.uint8)

    #cv2.imshow('x',x)
    cv2.imshow('stuff',img_x)
    cv2.waitKey(0)
