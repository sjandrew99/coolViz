#!/usr/bin/env python3

import numpy as np
import cv2
from copy import deepcopy
np.random.seed(0)
cos = np.cos; sin = np.sin
# using this: https://en.wikipedia.org/wiki/Perspective_(graphical)#Two-point_perspective
# another type of projection: https://en.wikipedia.org/wiki/Parallel_projection
# actually use this: https://en.wikipedia.org/wiki/3D_projection#Mathematical_formula

class Projector:
    def __init__(self,c,e,theta=None):
        self.R = None
        self.c = c # camera position
        self.e = e # display surface. use [0,0,1] for now
        if theta is not None:
            self.R = self.camera_rotation_matrix(theta)
    def project_point(self,a):
        d = self.R @ (a - self.c)
        bx = (self.e[2]/d[2])*d[0] + self.e[0]
        by = (self.e[2]/d[2])*d[1] + self.e[1]
        return bx, by
    
    def camera_rotation_matrix(self,theta):
        # compute camera rotation matrices:
        Rx = np.array([[1,                          0,              0],
                       [0,              cos(theta[0]),  sin(theta[0])],
                       [0,             -sin(theta[0]),  cos(theta[0])]])
                       
        Ry = np.array([[cos(theta[1]),              0, -sin(theta[1])],               
                       [0,                          1,              0],
                       [sin(theta[1]),              0,  cos(theta[1])]])
                       
        Rz = np.array([[cos(theta[2]),   sin(theta[2]),             0],
                       [-sin(theta[2]),  cos(theta[2]),             0],
                       [0,                          0,             1]])
                       
        R = Rz @ Ry @ Rx
        return R
    def set_camera_theta(self,theta):
        self.R = self.camera_projection_matrix(theta)

def plot_coords_to_img_coords(xp,yp=None):
    # copied over from ocv_plot/plot.py
    # plot_coords are in the "space" of the plot. img coords are pixels
    # xi = A*xp + C
    # yi = B*xp + D
    if (yp is None):
        # caller passed a list
        yp = xp[1]
        xp = xp[0]

    w = imwidth; h = imheight
    A = w / (xlim[1] - xlim[0])
    C = - A * xlim[0]
    B = h / (ylim[0] - ylim[1])
    D = - B*ylim[1]

    xi = A*xp+C
    yi = B*yp+D

    return (int(xi),int(yi))

def render_rectangle(img, width, height, center, color):
    # top-left, go clockwise:
    p0 = [center[0] - width/2, center[0] + height/2, center[2]]
    p1 = [center[0] + width/2, center[0] + height/2, center[2]]
    p2 = [center[0] + width/2, center[0] - height/2, center[2]]
    p3 = [center[0] - width/2, center[0] - height/2, center[2]]

    a0 = projector.project_point(p0)
    a1 = projector.project_point(p1)
    a2 = projector.project_point(p2)
    a3 = projector.project_point(p3)

    x0,y0 = plot_coords_to_img_coords(a0)
    x1,y1 = plot_coords_to_img_coords(a1)
    x2,y2 = plot_coords_to_img_coords(a2)
    x3,y3 = plot_coords_to_img_coords(a3)
    
    if x0 < 0 or x1 >= imwidth:
        assert False
    thickness = 2
    cv2.line(img, (x0,y0), (x1,y1), color, thickness)
    cv2.line(img, (x1,y1), (x2,y2), color, thickness)
    cv2.line(img, (x2,y2), (x3,y3), color, thickness)
    cv2.line(img, (x3,y3), (x0,y0), color, thickness)
    
    return ((x0, y0), (x1, y1), (x2,y2), (x3,y3))


#xlim = [-100, 100]
#ylim = [-100, 100]
xlim = [-1, 1]
ylim =  [-1,1]
imwidth = 800; imheight = 800
#imwidth = 1280; imheight = 720
imwidth = 1920; imheight = 1080

#width_out = 640; height_out = 480

#MIN_DEPTH = .0001
#MIN_DEPTH = 5 # for 800x800
MIN_DEPTH = 3
# point A to project:
#a = np.array([10,10,10] )
c = np.array([0,0,0]) # camera position. origin
theta = [0,0,0] # camera orientation, radians
e = [0,0,1] # display surface position (?)

projector = Projector(c,e,theta)

frameCount = 0
#MAX_DEPTH = 20 # assume we can't see anything beyond this
#QUANTIZATION = 20 # draw one cross-section at every MAX_DEPTH/QUANTIZATION depth steps
#observer_pos = 0

nRects = 40; max_rect_depth = 300
nRects = 20; max_rect_depth = 100
#nRects = 10; max_rect_depth = 50
#rect_width = 10; rect_height = 10 # good for aspect ratio 1 x 1
rect_width = 6; rect_height = (imwidth/imheight)*rect_width

rect_depth_array = np.linspace(MIN_DEPTH*2, max_rect_depth, nRects)
ddepth = np.mean(np.diff(rect_depth_array))
add_new_rect = 0

frameStop = None
writeVid = 1
if writeVid:
    writer = cv2.VideoWriter(f'space_{imwidth}x{imheight}.mp4',cv2.VideoWriter_fourcc('M','P','4','V'), 30, (imwidth,imheight))
    frameStop = 6000
    
nStars = 1000
starXY = np.zeros((nStars, 2),dtype=int)
starXY[:,0] = np.random.randint(low=0,high=imwidth, size=(nStars))    
starXY[:,1] = np.random.randint(low=0,high=imheight, size=(nStars))    
starSize = np.random.randint(low=1,high=3,size=nStars)
starColor = np.random.randint(64,255,size=nStars)
starBlink = np.random.randint(0,2,size=nStars)
while 1:
    #break
    img = np.zeros((imheight,imwidth,3),dtype=np.uint8)
    
    #observer_pos += frameCount * .003
    #observer_pos = 
    
    # put a rectangle in front of the camera:
    #render_rectangle(img, 10, 10, (0,0,10), (255,0,0,0))
    #render_rectangle(img, 10, 10, (0,0,1), (255,0,0,0))
    #depth_offset = frameCount * .05#.003
    
    # draw some "stars":
    
    for i in range(nStars):
        """
        if 400 < starXY[i,0] and starXY[i,0] < 1000:
            clr = (128,128,128)
        else:
            clr = (255,255,255)
        """
        s = starColor[i]
        #import pdb; pdb.set_trace()
        if starBlink[i]:
            # increase brightness:
            s = s + 5
            if s > 255:
                s = 255
                starBlink[i] = 0
        else:
            s -= 5
            if s < 64:
                s = 64
                starBlink[i] = 1
            
        starColor[i] = s
            
        clr = (int(s),int(s),int(s))
        cv2.circle(img, starXY[i,:], starSize[i], clr, -1, cv2.LINE_AA)

    rect_depth_array = rect_depth_array - .2 #0.02 # use .02 for realtime, .1 for 30 fps video
    iBehind = np.nonzero(rect_depth_array <= MIN_DEPTH)[0]
    if len(iBehind):
        assert len(iBehind) == 1
        assert iBehind[0] == 0
        rect_depth_array = rect_depth_array[1:] # remove 0th element
        add_new_rect = 1
    #print(f'text_rect: {text_rect}')
    if add_new_rect:
        # check if the last rectangle is close enough to the camera:
        if max_rect_depth - rect_depth_array[-1] >= ddepth:
            rect_depth_array = np.append(rect_depth_array, max_rect_depth)
            add_new_rect = 0
            #break
    rect_color = np.linspace(255,0, len(rect_depth_array))
    pts = [] # lazy
    for i in range(0,len(rect_depth_array)):

        clr = int(255 - 5*rect_depth_array[i])
        pts.append(render_rectangle(img, rect_width, rect_height, (0,0,rect_depth_array[i]), (clr,0,0)))

    cv2.imshow('img',img)
    if writeVid:
        #writer.write(cv2.resize(img, (width_out, height_out)))
        writer.write(img)
    #if cv2.waitKey(0) == 120: break # ord('x')
    cv2.waitKey(1)
    #cv2.waitKey(int(1000/30))
    if frameStop is not None and frameCount >= frameStop:
        break
    frameCount += 1


# insert 5 "fade" frames:
for i in range(0, 8):
    img = (img / 2).astype(np.uint8)
    # repeat:
    for j in range(0,5):
        cv2.imshow('img',img)
        cv2.waitKey(1)
        if writeVid:
            writer.write(cv2.resize(img, (imwidth, imheight)))
if writeVid:
    writer.release()                