#!/usr/bin/env python3

# https://blbadger.github.io/julia-sets.html
# TODO - reduce color banding: https://en.wikipedia.org/wiki/Julia_set#Pseudocode_for_multi-Julia_sets
import numpy as np
try:
    import cupy as xp
    using_cuda = 1
except:
    
    using_cuda = 0
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from coolViz.barrel_shifter import BarrelShifter

from matplotlib import cm
def data_to_bgr(data,cmap='nipy_spectral'):
    #jet = cm.get_cmap('jet')
    #jet = cm.get_cmap('plasma')
    #jet = cm.get_cmap('nipy_spectral')
    jet = cm.get_cmap(cmap)
    rgba = jet(data)
    rgb = np.uint8(255*rgba[:,:,0:3])
    bgr = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    return bgr

def julia_set(h_range, w_range, max_iterations,a = -0.744 + 0.148j):
    ''' A function to determine the values of the Julia set. Takes
    an array size specified by h_range and w_range, in pixels, along
    with the number of maximum iterations to try.  Returns an array with 
    the number of the last bounded iteration at each array value.
    '''
    # top left to bottom right
    y, x = xp.ogrid[1.4: -1.4: h_range*1j, -1.4: 1.4: w_range*1j]
    z_array = x + y*1j
    #a = -0.744 + 0.148j
    iterations_till_divergence = max_iterations + xp.zeros(z_array.shape)
    not_already_diverged = xp.ones(z_array.shape).astype(bool)
    diverged_in_past = xp.zeros(z_array.shape).astype(bool)
    
    """
    for h in tqdm(range(h_range)):
        for w in range(w_range):
            z = z_array[h][w]
            for i in range(max_iterations):
                z = z**2 + a
                if z * np.conj(z) > 4:
                    iterations_till_divergence[h][w] = i
                    break
    """
    for i in tqdm(range(max_iterations)):
        z_array = z_array**2 + a
        z_mag_arr = z_array * np.conj(z_array)
        diverging = z_mag_arr > 4
        diverging_now = diverging & not_already_diverged
        iterations_till_divergence[diverging_now] = i
        not_already_diverged = xp.invert(diverging_now) & not_already_diverged
        diverged_in_past = diverged_in_past | diverging_now
        z_array[diverged_in_past] = 0
    
    return iterations_till_divergence.get()

def plot_coords_to_img_coords(imsize,xlim,ylim, xp,yp):
        # plot_coords are in the "space" of the plot. img coords are pixels
        # xi = A*xp + C
        # yi = B*xp + D

        A = imsize[0] / (xlim[1] - xlim[0])
        C = - A * xlim[0]
        B = imsize[1] / (ylim[0] - ylim[1])
        D = - B*ylim[1]

        xi = A*xp+C
        yi = B*yp+D

        return (int(xi),int(yi))


def main(width, height, max_iter=200,record=None, cmap='gist_ncar'):	
    #js = julia_set(500,500,70)
    #js = julia_set(2000,2000,200)
    #plt.imshow(js,cmap='twilight_shifted')
    #plt.imshow(js,cmap='twilight')
    #plt.axis('off')
    #plt.imshow(js,cmap='nipy_spectral'); plt.show()

    #for max_iter in range(10,2500,100):
    #width = 500; height = 500; max_iter = 200
    #width = 800; height = 800; max_iter = 200;
    #width = 640; height = 480; max_iter = 200
    #width = 2000; height = 2000; max_iter = 200 # much prettier
    MAXVAL = []
    a = -0.744 + 0.148j
    maga = np.abs(a)
    anga = np.angle(a)
    #dang = .01; 
    dang = .025 # good
    #dmag0 = .01
    dmag0 = .005
    dmag = dmag0; sgn = 1
    max_mag = 2.5; min_mag = .01;
    min_mag0 = .6; max_mag0 = 1; # i want it to go a lot slower in this region

    phist = BarrelShifter(1500) # particle history
    #phist = BarrelShifter(500) # DEBUG
    aimg_xsize = [-max_mag,max_mag]
    aimg_ysize = [-max_mag,max_mag]
    scale = 1.25
    aimg_xsize[0] *= scale; aimg_xsize[1] *= scale
    aimg_ysize[0] *= scale; aimg_ysize[1] *= scale
    #js = julia_set(height,width,max_iter,a=a) # debug
    #writer = cv2.VideoWriter(f'jout_{width}_{height}_{max_iter}.mp4', cv2.VideoWriter_fourcc('M','P','4','V'), 30.0, (width,height))
    if record is not None:
        writer = cv2.VideoWriter(record, cv2.VideoWriter_fourcc('M','P','4','V'), 30.0, (width,height))    
    maxval = 0
    while 1:
        #a += .01 # move linearly
        #anga += .001; a = maga *np.exp(1j*anga) # move in a circle
        #anga += .001; maga += .01; a = maga*np.exp(1j*anga); # move in an outwards spiral
        
        anga += dang; 
        
        maga += dmag
        if maga >= max_mag:
            sgn = -1
        elif maga <= min_mag:
            sgn = 1 
        if min_mag0 <= maga and maga <= max_mag0:
            dmag = sgn*dmag0 / 10
        else:
            dmag = sgn*dmag0

        print('%.02f / %.03f' % (maga,((anga*180/np.pi) % 360)))
        a = maga*np.exp(1j*anga);
        
        # plot particle trajectory. totally useless but maybe cool:
        aimg = np.zeros((height,width,3),dtype=np.uint8)
        x,y = plot_coords_to_img_coords((width,height),aimg_xsize,aimg_ysize,np.real(a),np.imag(a))
        cv2.circle(aimg,(x,y),3,(0,0,255),-1)
        phist.push(a)
        dclr = 255 / len(phist.queue)
        #dclr = 300 / len(phist.queue)
        clr_rf = 255
        for i in range(len(phist.queue)-1, -1, -1):
            a0 = phist.queue[i]
            x0,y0 = plot_coords_to_img_coords((width,height),aimg_xsize,aimg_ysize,np.real(a0),np.imag(a0))
            clr_rf = clr_rf - dclr # gotta maintain an un-rounded value; otherwise when dclr goes above 0.5, everything gets stuck
            clr_r = int(np.round(clr_rf))
            clr_r = max(clr_r,0)
            cv2.line(aimg, (x,y),(x0,y0),(0,0,clr_r),1,cv2.LINE_AA)
            x = x0; y = y0
        
        js = julia_set(height,width,max_iter,a=a) 
        #maxval = max(np.max(js), maxval)
        maxval = np.max(js)
        #maxval = max_iter/4
        MAXVAL.append(maxval)
        # normalizing by maxval looks really great except it flickers a lot
        # normalizing by max_iter looks great when all the clumps are close together and not great when the clumps are far apart
        # average:
        maxval_wsize = 100
        if len(MAXVAL) < maxval_wsize:
            maxval = np.mean(MAXVAL)
        else:
            maxval = np.mean(MAXVAL[-maxval_wsize:])
        
        bgr = data_to_bgr(js / maxval,cmap=cmap)
        cv2.imshow('anomie',bgr)
        cv2.imshow('a',aimg)
        cv2.waitKey(1)
        if record:
            for i in range(0,4): # this helps reduce flicker
                writer.write(bgr)
        if phist.isFull():
            break
        else:
            print(len(phist.queue))
    if record:
        writer.release()

    #plt.plot(MAXVAL)
    #plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--width',type=int,default=800)
    parser.add_argument('--height',type=int,default=800)
    parser.add_argument('--nloops',type=int,default=5)
    parser.add_argument('--record',default=None)
    parser.add_argument('--vignette_sz',default=None,type=float)
    parser.add_argument('--vignette_mix',default=.05,type=float)
    parser.add_argument('--cmap',default='gist_ncar')
    
    args = parser.parse_args()
    main(args.width, args.height, record=args.record,cmap=args.cmap)