#!/usr/bin/env python3

import numpy as np
import cv2
import matplotlib.pyplot as plt # for debug plotting
np.random.seed(0)
from constants import gravitational_constant as G



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

    return int(xi)

class Particle:
    # spherical particle with position, velocity, mass, charge...
    def __init__(self,universe_size,
                 x=None,y=None,z=None,
                 xdot=None,ydot=None,zdot=None,
                 mass=None,
                 radius=None,
                 color=None):
        self.universe_size = universe_size
        self.x = 0 if x is None else x # meter
        self.y = 0 if y is None else y
        self.z = 0 if z is None else z
        self.xdot = 0 if xdot is None else xdot # m/s
        self.ydot = 0 if ydot is None else ydot
        self.zdot = 0 if zdot is None else zdot
        self.mass = 1 if mass is None else mass # kg. yes it's dumb to talk about kg with subatomic particles but i want to use standard units for everything
        self.radius = 1 if radius is None else radius
        self.color = (255,255,255) if color is None else color
        self.is_dead = 0
        
    def world_coords_to_img_coords(self,imsize):
        # 2d projection:
        self.x_ = world_coords_to_img_coords(self.x, imsize[1], self.universe_size[0])
        self.y_ = world_coords_to_img_coords(self.y, imsize[0], self.universe_size[1],invert=True)
        # this only works when both the universe and the image are square:
        self.radius_ = int(imsize[0] * self.radius / (self.universe_size[0][1] - self.universe_size[0][0]))
        # bounding box for collision detection:
        self.width_ = self.radius_ * 2
        self.height_ = self.radius_ * 2
        
    def draw2d(self,img):
        s = img.shape
        imsize = (s[1],s[0])
        self.world_coords_to_img_coords(imsize)
        if self.x_ < 0 or self.x_ >= imsize[0] or self.y_ < 0 or self.y_ >= imsize[1]:
             return
        cv2.circle(img,(self.x_,self.y_),self.radius_,self.color,-1,cv2.LINE_AA)
    def update(self,dt,idx=None,gravitational_field=None):
        if gravitational_field is not None:
            accel = gravitational_field[idx,:] 
            dv = accel * dt
            self.xdot = self.xdot + dv[0]
            self.ydot = self.ydot + dv[1]
            self.zdot = self.zdot + dv[2]
        self.x = self.x + self.xdot * dt
        self.y = self.y + self.ydot * dt
        self.z = self.z + self.zdot * dt
    
    def compute_force_field(self,pos,idx):
        # pos is N x 3 where force field should be evaluated
        R = pos - np.array([self.x, self.y, self.z]) # N x 3
        Rmag = np.linalg.norm(R,axis=1) # N x 3 
        
        #gravitational_field = (-G * self.mass * R.T / (Rmag**3)).T # N x 3
        Rhat = R / Rmag[:,None] # N x 3
        gravitational_field = (-G * self.mass * Rhat / (Rmag[:,None]**2)) # N x 3
        
        gravitational_field[idx,:] = 0 # no force on self
        return gravitational_field # TODO - compute electric also

if __name__ == "__main__":
    import sys
    from collision_detection import detect_object_collision

    imsize = (800,800) # width, height
    du = 1000
    universe_size = [[-du,du],[-du,du],[-du,du]]
    G = G * 5e8 # make it really strong

    p1 = Particle(universe_size, mass=1e7,radius=50)
    #p1 = Particle(universe_size,mass=1.9885e30) # sun
    #p2 = Particle(universe_size,x=.1,y=.1,mass=10)
    #p3 = Particle(universe_size,x=-.1,y=-.1,mass=10)
    #p2 = Particle(universe_size,x=.1,y=0,mass=10)
    #p3 = Particle(universe_size,x=0,y=-.1,mass=10)
    p2 = Particle(universe_size,x=500,y=30,xdot=.1,ydot=.1,radius=4)
    #p3 = Particle(universe_size,x=-30,y=-50,xdot=1,ydot=1) # coupled with a du of 1000, this gives a nice smooth periodicity in vel and accel
    p3 = Particle(universe_size,x=-300,y=-500,xdot=6,ydot=6,radius=5)
    p4 = Particle(universe_size,x=-200,y=-40,xdot=.1,ydot=.1,radius=10)
    p5 = Particle(universe_size,x=300,y=-500,xdot=10,ydot=10,radius=5) # like p3, but with a velocity less than escape
    escape_vel = np.sqrt(2 * G * p1.mass / np.sqrt(p5.x**2 + p5.y**2))
    p5_speed = np.sqrt(p5.xdot**2 + p5.ydot**2)
    assert p5_speed < escape_vel
    
    # make a stable orbit where the initial velocity is orthogonal to the sun:
    d = du / 4
    x = np.random.random()*du*2 + universe_size[0][0]
    y = np.sqrt(d**2 - x**2)
    escape_vel = np.sqrt(2*G*p1.mass / d)
    speed = 0.8 * escape_vel
    unit_vector = np.array([(x/d),(y/d)]) # points away from the sun
    xdot = -unit_vector[1] * speed # unit vector rotated by 90 degrees
    ydot = unit_vector[0] * speed
    p6 = Particle(universe_size,x=x,y=y,xdot=xdot,ydot=ydot,radius=20,color=(0,0,255))
    
    particles = [p1, p2, p3, p4, p5,p6]
    nParticles = len(particles)
    dt = .01
    
    #debug variables:
    """
    x = []
    y = []
    xdot = []
    ydot = []
    xdd = []; ydd = []
    r = []
    """
    frameCount = 0
    
    while 1:
        img = np.zeros((imsize[0],imsize[1],3),dtype=np.uint8)
        particle_pos = np.column_stack(([i.x for i in particles], [i.y for i in particles], [i.z for i in particles])) # N x 3
        #break
        grav_field = p1.compute_force_field(particle_pos,0) # only compute grav field due to p1, ignore the rest
        for i in range(0,nParticles):
             if particles[i].is_dead: continue
             particles[i].update(dt,i,grav_field)
             #continue
             particles[i].draw2d(img)
             
             if i == 0: continue
             # look for collisions with the "sun"
             collisions = detect_object_collision(particles[i].x_, particles[i].y_, particles[i].radius_, particles[i].radius_,
                                      p1.x_, p1.y_, p1.radius_, p1.radius_) # note - using radius, not width/height of bounding box
             if any(collisions):
                 #particles_to_delete.append(i)
                 particles[i].is_dead = 1 
             
             """
             # draw velocity vector, length 10 pixels:
             l = 100
             vmag = np.sqrt(particles[i].xdot**2 + particles[i].ydot**2)
             if vmag != 0:
                 vx = (l * particles[i].xdot / vmag) + particles[i].x_
                 vy = -(l * particles[i].ydot / vmag) + particles[i].y_
                 cv2.line(img, (particles[i].x_, particles[i].y_), (int(vx), int(vy)), (255,0,0))
             
             accel = grav_field[i,:] * dt
             amag = np.sqrt(accel[0]**2 + accel[1]**2)
             ax = (l * accel[0] / amag) + particles[i].x_
             ay = -(l * accel[1] / amag) + particles[i].y_
             cv2.line(img, (particles[i].x_, particles[i].y_), (int(ax),int(ay)), (0,0,255))
             #if i == 2:
             #    print('%d: (%.3f,%.3f) (%.3f, %.3f)' % (i,particles[i].x,particles[i].y,particles[i].xdot, particles[i].ydot))
             """
        
        # debug:
        """
        # draw bounding boxes for p1 and p4:
        cv2.rectangle(img, (int(p1.x_ - p1.width_/2), int(p1.y_+p1.height_/2)), (int(p1.x_ + p1.width_/2), int(p1.y_-p1.height_/2)),(0,255,0))
        cv2.rectangle(img, (int(p4.x_ - p4.width_/2), int(p4.y_+p4.height_/2)), (int(p4.x_ + p4.width_/2), int(p4.y_-p4.height_/2)),(0,255,0))
        x.append(p3.x)
        y.append(p3.y)     
        xdot.append(p3.xdot)
        ydot.append(p3.ydot)
        accel = grav_field[2,:] * dt
        xdd.append(accel[0])
        ydd.append(accel[1])
        r.append(np.sqrt(p3.x**2 + p3.y**2))
        """
        
        cv2.imshow('universe',img)
        cv2.waitKey(1)
        frameCount += 1
        #if frameCount > 3500:
        if frameCount > 200000:
            break
    
    import matplotlib.pyplot as plt
    if len(xdot):
        fig,ax=plt.subplots(4,sharex='all')
        ax[0].plot(x,'b',y,'r')
        ax[1].plot(xdot,'b',ydot,'r')
        ax[2].plot(xdd,'b',ydd,'r')
        ax[3].plot(r)
        plt.show()
    