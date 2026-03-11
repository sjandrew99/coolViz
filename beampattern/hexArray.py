#!/usr/bin/env python3
#import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
speed_of_light = 299792458
import cv2
from copy import deepcopy
from coolViz.collision_detection import detect_border_collision
from coolViz.cmapStepper import CmapStepper
from coolViz.imMixer import FixedVignette2
from tqdm import tqdm
from coolViz.barrel_shifter import BarrelShifter
#plt.ion()
#np.random.seed(0)

from matplotlib import cm
def data_to_bgr(data):
    #jet = cm.get_cmap('jet')
    #jet = cm.get_cmap('plasma')
    jet = cm.get_cmap('nipy_spectral')
    rgba = jet(data)
    rgb = np.uint8(255*rgba[:,:,0:3])
    bgr = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    return bgr

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

class HexagonalArray:
    def __init__(self,maxdx_=40,maxdy_=40,preComp=None):

        # hexagon planar array in xy-plane
        # (SHA)
        f0 = 1e9 # Hz
        self.N_elements = 7
        #self.Nstep = 1000
        self.Nstep = 200
        self.Nstep_mult=2 # ratio of how long elements can move randomly to how often they return to center
        self.Nstep_to_return = int(self.Nstep/3)

        lamda = speed_of_light / f0
        self.lamda = lamda
        grid_spacing = .01 # decreasing this doesn't result in a finer image because there's a resize
        u = np.arange(-1,1,grid_spacing)
        v = np.arange(-1,1,grid_spacing)
        grid = np.meshgrid(u,v)
        # SHA design:
        dx = lamda / 2
        dy = np.sqrt(3)*lamda / 4

        # one element at the origin
        self.p = np.array([[0,0],
                     [dx,0],
                     [dx/2,dy],
                     [-dx/2,dy],
                     [-dx,0],
                     [-dx/2,-dy],
                     [dx/2,-dy]])
        self.dx = dx; self.dy = dy
        self.p_orig = deepcopy(self.p)

        self.maxdx = dx/maxdx_
        self.maxdy = dy/maxdy_
        #self.maxdx = dx / 10; self.maxdy = dy / 10
        self.maxx = lamda*3; self.maxy = lamda*3
        self.element_velocities = np.random.rand(self.N_elements,2) * 2 - 1
        self.element_velocities = self.element_velocities * np.array([self.maxdx,self.maxdy])
        # TODO - implement steering
        u_steer = 0
        v_steer = 0
        self.phase_shifts = np.ones(self.N_elements,dtype=np.complex128) * np.exp(0)

        self.istep = 0
        self.return_to_center = 0
        self.k = -2*np.pi * np.array([grid[0],grid[1]]) / lamda
        self.max_log_response = BarrelShifter(100)

        self.cap = None
        self.nFramesPreComp = 0
        self.frameCounterPreComp = 0
        self.seekDir = 1
        if preComp:
            self.cap = cv2.VideoCapture(preComp)
            self.nFramesPreComp = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def update(self):
        if self.cap: return
        if self.return_to_center:
            if self.istep == self.Nstep_to_return:
                self.istep = 0
                self.return_to_center = 0
                self.element_velocities[:,0] = self.dx_ / np.abs(self.dx_)
                self.element_velocities[:,1] = self.dy_ / np.abs(self.dy_)
                self.element_velocities = self.element_velocities * np.array([self.maxdx,self.maxdy])
                #self.element_velocities = self.element_velocities / 2 # make them kinda slow after the reset 
        else:
            if self.istep == self.Nstep*self.Nstep_mult:
                self.istep = 0
                d = self.p_orig - self.p
                self.dx_ = d[:,0]/(self.Nstep_to_return)
                self.dy_ = d[:,1]/(self.Nstep_to_return)
                self.return_to_center = 1
        if self.return_to_center:
            self.p[:,0] = self.p[:,0] + self.dx_
            self.p[:,1] = self.p[:,1] + self.dy_
            self.istep += 1
        else:
            """
            perturbation = (np.random.rand(N_elements,2)*2 - 1)*dx/15
            elements_to_perturb = np.nonzero(np.random.randint(0,2,size=7))
            p[elements_to_perturb,:] = p[elements_to_perturb,:] + perturbation[elements_to_perturb,:]
            p = np.clip(p,-lamda*4,lamda*4)
            """
            for i in range(0,self.N_elements):
                self.p[i,:] += self.element_velocities[i,:]
                #import pdb; pdb.set_trace()
                #fcollisions = detect_border_collision(self.p[i,0],self.p[i,1],4*self.dx,4*self.dy,[-self.maxx,-self.maxy,self.maxx*2,self.maxy*2])
                fcollisions = detect_border_collision(self.p[i,0],self.p[i,1],self.lamda,self.lamda,[-self.maxx,-self.maxy,self.maxx*2,self.maxy*2])
                if np.any(fcollisions):
                    # move away from the border we just hit:
                    #p[i,0] -= fcollisions[0]
                    #p[i,1] -= fcollisions[1]
                    #print(f'collisions: {fcollisions}')
                    #import pdb; pdb.set_trace()
                    if fcollisions[0]:
                        # move an opposite direction from the left/right border. positive fcollisions[0] is a hit on the right
                        #print(f'collisions: {fcollisions}')
                        #self.element_velocities[i,0] = -(self.element_velocities[i,0]/np.abs(self.element_velocities[i,0])) * np.random.rand()*self.maxdx
                        self.element_velocities[i,0] = -fcollisions[0] * np.random.rand()*self.maxdx
                    if fcollisions[1]:
                        # since i'm detecting collisions in "plot space" not "image space", +1 is a collision with the top
                        #self.element_velocities[i,1] = -(self.element_velocities[i,1]/np.abs(self.element_velocities[i,1])) * np.random.rand()*self.maxdy
                        self.element_velocities[i,1] = -fcollisions[1] * np.random.rand()*self.maxdy

            self.istep += 1

    def draw_array(self,imsize):
            
        array_img = np.zeros((imsize[1],imsize[0],3),dtype=np.uint8)
        for i in range(0,self.N_elements):
            x,y = plot_coords_to_img_coords(imsize,(-self.lamda*4,self.lamda*4),(-self.lamda*4,self.lamda*4),self.p[i,0],self.p[i,1])
            cv2.circle(array_img,(x,y),4,(0,255,0))
        #cv2.imshow('array',array_img)
        return array_img

    def calc_and_draw_response(self,imsize):
        if self.cap:
            """
            if self.frameCounterPreComp >= (self.nFramesPreComp-1) or self.frameCounterPreComp < 0:
                 #import pdb; pdb.set_trace()
                 self.seekDir = -self.seekDir
            self.frameCounterPreComp += self.seekDir 
            self.cap.set(cv2.CAP_PROP_POS_FRAMES,self.frameCounterPreComp)
            """
            r,bgr = self.cap.read()
            if not r:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES,0)
                r,bgr = self.cap.read()
            return bgr
        #st = timer()
        """
        response = np.zeros((len(u),len(v)),dtype=np.complex128)
        for iu in range(0,len(u)):
        
            #for iv in range(0,len(v)):
            #    k = -2*np.pi/lamda * np.array(([u[iu],v[iv]])) # TODO - mag is probably unnecessary
            #    # note that van trees has k as 3 element, but the third is unnecessary for a planar array (pz is always 0)
            #    #V = np.zeros(N_elements,dtype=np.complex128)
            #     V = np.exp(-1j*np.dot(k[:,None].T,p.T)) # 1x2 dot 2xN
            #    response[iu,iv] = np.sum(V*phase_shifts)
        
            k = -2*np.pi * np.array([np.repeat(u[iu],len(v)),v]) / lamda
            V = np.exp(-1j*np.dot(k.T,p.T))
            response[iu,:] = np.sum(V*phase_shifts,axis=1)
        """
        #k = -2*np.pi * np.array([grid[0],grid[1]]) / lamda
        V = np.exp(-1j*np.dot(self.k.T,self.p.T))
        response = np.sum(V*self.phase_shifts,axis=2) # TODO - incorporate phase shifts
    
        #tComp = timer() - st
        #print('%.1f ms to compute response' % (tComp * 1000))
        #x = np.abs(response); x = 2**(x/2); x = np.abs(x) / np.max(np.abs(x))
        
        #x = 20*np.log10(np.abs(response)); x = x - np.min(x); 
        #self.max_log_response.push(np.nanmax(x)); x = x / np.mean(self.max_log_response.queue);
        #x = x / np.max(x); # log space looks cool but it shakes a lot, idk why. probably because i'm normalizing by a different # each time
        
        x = np.abs(response) / np.max(np.abs(response))
        bgr = data_to_bgr(x)
        bgr = cv2.resize(bgr,(imsize[0],imsize[1]),interpolation=cv2.INTER_CUBIC)
        #cv2.imshow('response',bgr)
        return bgr
    
    def __del__(self):
        if self.cap: self.cap.release()

if __name__=="__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--width',type=int,default=800)
    parser.add_argument('--height',type=int,default=800)
    parser.add_argument('--nloops',type=int,default=5)
    parser.add_argument('--record',default=None)
    parser.add_argument('--vignette_sz',default=None,type=float)
    parser.add_argument('--vignette_mix',default=.05,type=float)
    
    args = parser.parse_args()
    
    
    #array = HexagonalArray(maxdx_=10,maxdy_=10)
    array = HexagonalArray()
    cmap_stepper = CmapStepper()
    width = args.width
    height = args.height
    imsize = (width, height)
    mixer = None
    if args.vignette_sz is not None:
        
        mixer = FixedVignette2(args.vignette_sz,args.vignette_mix)
        foreground = np.zeros((height,width,3),dtype=np.uint8)
    frameCount = 0
    maxFrames = args.nloops*(array.Nstep * array.Nstep_mult + array.Nstep_to_return - 1) + 1
    if args.record:
        writer = cv2.VideoWriter(args.record,cv2.VideoWriter_fourcc('M','P','4','V'),30,imsize)
    pbar = tqdm(total=maxFrames)
    while frameCount < maxFrames:
        array.update()
        response = array.calc_and_draw_response(imsize)
        response = response + cmap_stepper.counter
        cmap_stepper.update()
        if mixer is not None:
            response = mixer.mix(response, foreground)
        cv2.imshow('img',response)
        cv2.imshow('array',array.draw_array(imsize))
        cv2.waitKey(1)
        frameCount += 1
        if args.record:
            writer.write(response)
        pbar.update(1)#frameCount)
    pbar.close()
    if args.record:
        writer.release()
                      