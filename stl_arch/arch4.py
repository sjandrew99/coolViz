#!/usr/bin/env python3

import numpy as np
import cv2
import matplotlib.pyplot as plt
from vidTools import transformations
from copy import deepcopy
np.random.seed(0)
from scipy.signal import get_window
#https://www.ams.org/notices/201002/rtx100200220p.pdf
arch_depth = 400 # ft
arch_dx = 1.5
A = 68.7672; B = 0.0100333
halfwidth = 299.2239

    
# only compute the southern half:    
#x = np.arange(-halfwidth, dx, dx) # include x=0 point
x = np.arange(-halfwidth, halfwidth+arch_dx, arch_dx) # both halves
y = 693.8597 - A * np.cosh(B*x) # centroid curve C. up-down
z = np.zeros(x.shape) + arch_depth # east-west. TODO - give it some depth. maybe put camera (origin) in river?    


# at each point on the centroid curve, the triangle lies in the plane orthogonal to the derivative of the curve, pointing inward
DX = np.diff(x)
DY = np.diff(y) # forward difference. DY[0] = y[1] - y[0]
# append the point for 0, maxheight:
DX = np.concatenate((DX, [arch_dx]))
#DY = np.concatenate((DY, [0])) # if only computing southern half
DY = np.concatenate((DY, [DY[-1]])) # if computing both halves
m_tangent = np.row_stack((DX, DY)) # 2 x N
m_tangent = m_tangent / np.sqrt(np.sum(m_tangent**2, axis=0)) 
m_orth = np.row_stack((DY,-DX)) # rotates the tangent line by -90
m_orth = m_orth / np.sqrt(np.sum(m_orth ** 2,axis=0)) # make unit vector

Q = 1262.6551 - 1.81977*y # area of triangles
d = np.sqrt(4 * Q / np.sqrt(3)) # length of eq. triangles that form cross section
a = np.sqrt((d**2) - ((d/2)**2)) # "height" of triangle, from outside to inside
x_ = m_orth[0,:] * (a/2) + x # point "1" (innermost) for each triangle
y_ = m_orth[1,:] * (a/2) + y
P1 = np.row_stack((x_,y_, np.zeros(x.shape) + arch_depth)) # North, up, west. 3 x N
nTri = P1.shape[1]
x_ = -m_orth[0,:] * (a/2) + x # midpoint between points "2" (outermost, westernmost) and "3" (outermost, easternmost) for each triangle
y_ = -m_orth[1,:] * (a/2) + y
P2 = np.row_stack((x_,y_, np.zeros(x.shape) + arch_depth + a/2)) 
P3 = np.row_stack((x_,y_, np.zeros(x.shape) + arch_depth - a/2)) 


def imshow(frame,delay):
    cv2.imshow('img',frame)
    cv2.waitKey(delay)

delay = 10

imsize = (800, 800)

# worldlims are north, up
vertices = np.concatenate((P1,P2,P3),axis=1) # 3 x N. P1[k] = vertices[:,k]. P2[k] = vertices[:,k+N], etc
                                             # P1[k] on the southern half is P1[-(k+1)] on the northern half
projected_x = vertices[0,:] / vertices[2,:] # north -> right
projected_y = vertices[1,:] / vertices[2,:] # up -> up
# TODO - z-buffer and use a real camera projection matrix

#proj x, y are right, up
worldlims = [[-.9,.9], [-1, 1.7]]
plotter = transformations.Plotter(imsize,worldlims[0],worldlims[1])  # works well with a 1:1 aspect ratio

Xi, Yi = plotter.plot_coords_to_img_coords_vec(projected_x, projected_y)

# draw the river as a series of sinusoids. xyz is nuw:
river_dx = 4
river_x = np.arange(-10*halfwidth, 10*halfwidth, river_dx)
nSigs = 50
river_y = np.zeros((nSigs,len(river_x)),dtype=np.complex128)
for iSig in range(nSigs):
    nwav = 1 # single tone per line
    fmax = 15
    freqs = np.random.random(nwav) * fmax + 10
    amps = np.random.random(nwav)*10
    phases = np.random.random(nwav) * 2 * np.pi
    for i in range(nwav):
        river_y[iSig,:] = river_y[iSig,:] + amps[i]*np.exp(1j*(2*np.pi*freqs[i]*river_x + phases[i])) # complex, take real part later
        #river_y[iSig,:] = river_y[iSig,:] + amps[i]*np.cos(2*np.pi*freqs[i]*river_x + phases[i]) # real
#river_y = river_y - 100 # put it below the arch
river_y_offset = 50
river_z = np.linspace(10,arch_depth/2, nSigs)
river_z = np.repeat(river_z[:,None], len(river_x),axis=1) # nSig x N

# window params from sinusoids.py
class Cycler:
    def __init__(self, data,loop_behaviour):
        self.data = data
        assert len(self.data.shape) == 1
        self.len = len(self.data)
        self.loop_behaviour = loop_behaviour # 0: cycle back to begin at end of array. 1: reverse
        self.step = 1
        self.idx = 0
    def next(self):
        if self.idx >= self.len:
            if self.loop_behaviour == 0:
                self.idx = 0
            else:
                self.idx = self.idx - 1
                self.step = -1
        if self.idx < 0:
            assert self.loop_behaviour == 1
            self.step = 1
            self.idx = 0

        r = self.data[self.idx]
        self.idx += self.step
        return r


#min_stddev = 1/512; max_stddev = 5/16; d_stddev = 4/1024
min_stddev = 40/512; max_stddev = 15/16; d_stddev = 4/1024
#min_stddev = 200/512; max_stddev = 15/16; d_stddev = 4/1024
#fs = dx
#rollInc = 1

stddev = np.arange(min_stddev, max_stddev, d_stddev)
stddev = np.flipud(stddev) # reverse
#stddev = np.append(stddev,stddev) # sawtooth wave
stddev = np.append(stddev,np.flipud(stddev)[1:]) # triangle wave

slowdown = 1/16
stddev = np.append(np.arange(max_stddev, slowdown, -d_stddev), np.arange(slowdown, min_stddev, -d_stddev/4))
stddev = np.append(stddev,np.flipud(stddev)) # triangle wave
#roll = np.arange(0,1500,rollInc).astype(int)
winStddevCycle = Cycler(stddev,loop_behaviour=0)
#wRollCycle = Cycler(roll,loop_behaviour=0)


winroll = 0
iArch = 0
build_arch = 1
archframe = np.zeros((imsize[1],imsize[0],3),dtype=np.uint8)
loop_count = 0
fps = 30
vidname = f'arch_{imsize[0]}_{imsize[1]}.mp4'
writer = cv2.VideoWriter(vidname,cv2.VideoWriter_fourcc('M','P','4','V'), fps, imsize)
while 1:

    frame = np.zeros((imsize[1],imsize[0],3),dtype=np.uint8)
    # render one river frame:    
    win_stddev = winStddevCycle.next()
    window = get_window(('gaussian',win_stddev*len(river_x)),len(river_x)) # std-dev is measured in samples

    phase_shift = 10*np.pi/180
    river_y = river_y * np.exp(1j*phase_shift) # phase shift only works for mean-zero signals. why?
    window = np.roll(window, winroll) # TODO - this should look like a fftshift, at least sometimes. why doesn't it?        
    winroll += 1
    river_y_ = river_y * window

    river_y_ = river_y_ - river_y_offset
    projected_x = river_x / river_z # north -> right. nSig x N
    projected_y = np.real(river_y_) / river_z # up -> up. nSig x N
    Xi_r, Yi_r = plotter.plot_coords_to_img_coords_vec(projected_x, projected_y)
    clr = (255,0,0)
    for iSig in range(0, nSigs):
        pts = [np.array([Xi_r[iSig,:],Yi_r[iSig,:]]).T] # N x 2
        cv2.polylines(frame, pts, 0, clr, 1)
    
    #frame = frame + archframe
    #imshow(frame,delay*4)

    # render one arch frame    
    # draws each triangle from bottom to top
    if build_arch:
        archframe = np.zeros((imsize[1],imsize[0],3),dtype=np.uint8)
        for i in range(0,iArch): 
            for k in [0,1,2]:
                xi = Xi[k*nTri + i]
                yi = Yi[k*nTri + i]
                archframe[yi,xi,:] = (255,255,255)
                
                xi = Xi[-(k*nTri + i)]
                yi = Yi[-(k*nTri + i)]
                archframe[yi,xi,:] = (255,255,255)
            # triangulate:
            # P1 to P2:
            cv2.line(archframe, (Xi[i], Yi[i]), (Xi[nTri+i], Yi[nTri+i]), (255,255,255), 1, cv2.LINE_AA)    
            cv2.line(archframe, (Xi[-(i+1)], Yi[-(i+1)]), (Xi[-(nTri+i+1)], Yi[-(nTri+i+1)]), (255,255,255), 1, cv2.LINE_AA)    
            # P2 to P3:
            cv2.line(archframe, (Xi[i+nTri], Yi[i+nTri]), (Xi[2*nTri+i], Yi[2*nTri+i]), (255,255,255), 1, cv2.LINE_AA)    
            cv2.line(archframe, (Xi[-(i+nTri+1)], Yi[-(i+nTri+1)]), (Xi[-(2*nTri+i+1)], Yi[-(2*nTri+i+1)]), (255,255,255), 1, cv2.LINE_AA)    
            # P3 to P1:
            cv2.line(archframe, (Xi[i+2*nTri], Yi[i+2*nTri]), (Xi[i], Yi[i]), (255,255,255), 1, cv2.LINE_AA)    
            cv2.line(archframe, (Xi[-(1+i+2*nTri)], Yi[-(1+i+2*nTri)]), (Xi[-(i+1)], Yi[-(i+1)]), (255,255,255), 1, cv2.LINE_AA)    
            #if i != int(nTri/2)-1: # TODO - probably need to mess with this condition
            if True:
                # P1 of current triangle to P1 of next:
                cv2.line(archframe, (Xi[i], Yi[i]), (Xi[i+1], Yi[i+1]), (255,255,255), 1, cv2.LINE_AA)    
                cv2.line(archframe, (Xi[-(i+1)], Yi[-(i+1)]), (Xi[-(i+2)], Yi[-(i+2)]), (255,255,255),1,cv2.LINE_AA)  
                # P2:
                cv2.line(archframe, (Xi[i+nTri], Yi[i+nTri]), (Xi[i+1+nTri], Yi[i+1+nTri]), (255,255,255), 1, cv2.LINE_AA)    
                cv2.line(archframe, (Xi[-(i+1+nTri)], Yi[-(i+1+nTri)]), (Xi[-(i+2+nTri)], Yi[-(i+2+nTri)]), (255,255,255),1,cv2.LINE_AA)  
                # P3:
                cv2.line(archframe, (Xi[i+2*nTri], Yi[i+2*nTri]), (Xi[i+1+2*nTri], Yi[i+1+2*nTri]), (255,255,255), 1, cv2.LINE_AA)    
                cv2.line(archframe, (Xi[-(i+1+2*nTri)], Yi[-(i+1+2*nTri)]), (Xi[-(i+2+2*nTri)], Yi[-(i+2+2*nTri)]), (255,255,255),1,cv2.LINE_AA)  
        iArch += 1
        # last point:
        iTop = int(np.round(nTri/2))
        if iArch >= iTop:
            cv2.line(archframe, (Xi[iTop], Yi[iTop]), (Xi[iTop+nTri], Yi[iTop+nTri]), (255,255,255), 1, cv2.LINE_AA)
            cv2.line(archframe, (Xi[iTop+nTri], Yi[iTop+nTri]), (Xi[iTop+nTri*2], Yi[iTop+nTri*2]), (255,255,255), 1, cv2.LINE_AA)
            cv2.line(archframe, (Xi[iTop+2*nTri], Yi[iTop+2*nTri]), (Xi[iTop], Yi[iTop]), (255,255,255), 1, cv2.LINE_AA)
            
            cv2.line(archframe, (Xi[iTop], Yi[iTop]), (Xi[iTop+1], Yi[iTop+1]), (255,255,255), 1, cv2.LINE_AA)
            cv2.line(archframe, (Xi[iTop+nTri], Yi[iTop+nTri]), (Xi[iTop+1+nTri], Yi[iTop+1+nTri]), (255,255,255), 1, cv2.LINE_AA)
            cv2.line(archframe, (Xi[iTop+2*nTri], Yi[iTop+2*nTri]), (Xi[iTop+1+2*nTri], Yi[iTop+1+2*nTri]), (255,255,255), 1, cv2.LINE_AA)
            
            
    imshow(frame+archframe,delay)
    writer.write(frame+archframe)
    if not build_arch:
       iArch += 1
    if iArch >= int(np.round(nTri/2)):
        iArch = 0
        loop_count += 1
        if build_arch:
            build_arch = 0
        else:
            build_arch = 1
    if loop_count > 3:
        break
writer.release()
"""
fig,ax = plt.subplots()
for th in range(0,360):
    river_y = river_y * np.exp(1j*th * np.pi/180)
    river_y_ = river_y - river_y_offset
    projected_y = np.real(river_y_) / river_z
    ax.cla()
    ax.set_ylim([-20,20])
    #ax.plot(river_x, np.real(river_y[0,:]))
    ax.plot(river_x, projected_y[0,:])
    plt.pause(.1)
plt.show()
"""