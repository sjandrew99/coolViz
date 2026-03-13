#!/usr/bin/env python3
import numpy as np
from copy import deepcopy
import cv2

def tint(tup):
    return (int(tup[0]),int(tup[1]))

class Hexagon:
    def __init__(self,org,dx):
        # Topmost edge is horziontal
        dy = np.sqrt(3)*dx/2
        self.dy = dy
        # 6 x 2 array of vert positions:
        self.p = np.array([[dx,0], # right. next go clockwise
                     [dx/2,dy],
                     [-dx/2,dy],
                     [-dx,0],
                     [-dx/2,-dy],
                     [dx/2,-dy]])
        self.p = self.p + org
        # TODO - rotate
        self.org = org
        self.dx = dx
        self.maxx = np.max(self.p[:,0])
        self.minx = np.min(self.p[:,0])
        self.maxy = np.max(self.p[:,1])
        self.miny = np.min(self.p[:,1])        
        #self.neighborMap = np.zeros((6))
        #HexMap.append(org)
                    
    def draw(self,img,edgecolor=(255,255,255),facecolor=(0,0,0),thickness=2):
        for i in range(0,5):
            cv2.line(img,tint(self.p[i,:]),tint(self.p[i+1,:]),edgecolor,thickness,cv2.LINE_AA)
        cv2.line(img,tint(self.p[5,:]),tint(self.p[0,:]),edgecolor,thickness,cv2.LINE_AA)
        cv2.fillConvexPoly(img,self.p.astype(int),facecolor,cv2.LINE_AA)
    
    def getNeighborOrg(self,direction):
        # direction 0: lower-right, up to 5 going clockwise
        # 0 -> 3, 1 -> 4, 2 -> 5
        if direction == 0:
            org = (self.org[0]+3*self.dx/2, self.org[1]+self.dy)
        elif direction == 1:
            org = (self.org[0], self.org[1]+self.dy*2)
        elif direction == 2:
            org = (self.org[0] - 3*self.dx/2, self.org[1]+self.dy)
        elif direction == 3:
            org = (self.org[0] - 3*self.dx/2, self.org[1]-self.dy)
        elif direction == 4:
            org = (self.org[0], self.org[1]-2*self.dy)
        elif direction == 5:
            org = (self.org[0]+3*self.dx/2, self.org[1]-self.dy)
        return org
    
    def nextHex(self,direction):
        org = self.getNeighborOrg(direction)
        return Hexagon(org,self.dx)
    
    def checkDirection(self,direction,hexList):
        neighborOrg = self.getNeighborOrg(direction)
        for h in hexList:
            if np.abs(h.org[0] - neighborOrg[0]) < self.dx/2 and np.abs(h.org[1] - neighborOrg[1]) < self.dx/2:
                return False
        return True
    
    def getChildDirection(self,direction):
        return (direction + 3) % 6

    def isValid(self,imsize):
        # checks whether any pixels are inside the frame
        if 0 <= self.maxx or self.minxx < imsize[0] or 0 <= self.maxy or self.miny <= imsize[1]:
            return True
        return False

    """
    def initNeighborMap(self,imsize):
        for i in range(0,6):
            if self.neighborMap[i]: continue
            h = self.nextHex(i)
            if not h.isValid(imsize):
                self.neighborMap[i] = -1
    def isFull(self):
        return np.sum(np.abs(self.neighborMap)) == 6
    """

def surround(h,hexMap,img):
    # generates neighbors for a hexagon in random order and sometimes doesn't generate all of them
    # hexMap - N x 3, 0th column is alreadyDrawn?, 1st column is X coord of hex center, 2nd is Y
    newhexlist = []
    dirs = [0,1,2,3,4,5]
    np.random.shuffle(dirs)
    for i in dirs:
        if np.random.random() > 0.7: continue # skip some neighbors
        nextHex = h.nextHex(i) 
        org = np.array(nextHex.org).astype(int) 
        iFound = np.nonzero((np.abs(hexMap[:,1] - org[0]) < 2 ) * (np.abs(hexMap[:,2] - org[1]) < 2))[0] # TODO tolerance of two, could be better
        if len(iFound) == 0:
            # this is a strange case that I should investigate but I've already seen that hexMap covers the whole space so I'm gonna ignore it
            continue
        iFound = iFound[0]
        if hexMap[iFound,0]:
            # already drawn
            continue
        clr = (np.random.randint(40,60),np.random.randint(200,240),np.random.randint(210,255)) # yellow-gold
        nextHex.draw(img,facecolor=clr)
        hexMap[iFound,0] = 1
                
        newhexlist.append(nextHex)
    return newhexlist
        

class HexagonalTesselation:
    def __init__(self,org,sz,imsize):
        self.org = org
        self.sz = sz
        self.imsize = imsize
        # tesselate:
        h = Hexagon(org,sz)
    
        Y = np.arange(h.org[1]-h.dy,imsize[1]+h.dy,h.dy)
        X1 = np.arange(h.org[0]-3*h.dx/2, imsize[0]+3*h.dx/2,3*h.dx) # row above
        X2 = np.arange(h.org[0]-3*h.dx, imsize[0]+3*h.dx,3*h.dx) # same row
        pts = []
        # fills up pts in top-down, left-right order
        for iY in range(0,len(Y)):
            X = X1 if iY % 2 == 0 else X2
            for iX in range(0,len(X)):
                org = tint((X[iX],Y[iY]))
                pts.append(org)
        self.pts = pts
    def raster(self,img):
        for org in self.pts:
            nextHex = Hexagon(org,self.sz)
            nextHex.draw(img,facecolor=(51,215,255))
            yield
    
    def random_emplace(self,img):
        hexMap = np.zeros((len(self.pts)))
        icount = 0
        while np.sum(hexMap) < len(hexMap):
            if icount > 10000: break
            ip = np.random.randint(len(self.pts))
            if hexMap[ip]: 
                icount += 1
                continue
            icount =0
            org = self.pts[ip]
            hexMap[ip] = 1
            nextHex = Hexagon(org,self.sz)
            #nextHex.draw(img,facecolor=(51,215,255))
            clr = (np.random.randint(40,60),np.random.randint(200,240),np.random.randint(210,255))
            nextHex.draw(img,facecolor=clr)
            yield
            #cv2.imshow('tesselation',img)
            #cv2.waitKey(3)
        iNotFilled = np.nonzero(hexMap == 0)[0]
        for i in iNotFilled:
            org = self.pts[i]
            nextHex = Hexagon(org,self.sz)
            nextHex.draw(img,facecolor=(51,215,255))
            #cv2.imshow('tesselation',img)
            #cv2.waitKey(1)
        yield
    
    def bloom(self,img):
        # yields 0 or 1. 0 -> bloom center. 1 -> neighbor
        hexMap = np.zeros((len(self.pts),3))
        for i in range(0,len(self.pts)):
            hexMap[i,1:3] = self.pts[i]
    
        while np.sum(hexMap[0,:]) < len(self.pts):
            # choose bloom centers:
            nChoose = 3
            iNotFilled = np.nonzero(hexMap[:,0] == 0)[0]
            if len(iNotFilled) == 0: break
            iChoose = np.random.randint(0,len(iNotFilled),3)
            hexlist = []
            for i in iChoose:
                idx = iNotFilled[i]
                org = self.pts[idx]
                nextHex = Hexagon(org,self.sz)
                clr = (np.random.randint(40,60),np.random.randint(200,240),np.random.randint(210,255)) # yellow-gold
                nextHex.draw(img,facecolor=clr)
                hexMap[idx,0] = 1
                hexlist.append(nextHex)
                
            yield 0

            # populate neighbors:
            # TODO - figure out recursion
            for h in hexlist:
                newhexlist = surround(h,hexMap,img)
                yield 1
                if np.random.random() > 0.5: continue
                for hnew in newhexlist:
                    newhexlist2 = surround(hnew,hexMap,img)       
                    yield 1
                    if np.random.random() > 0.5: continue
                    for h2 in newhexlist2:
                        newhexlist3 = surround(h2,hexMap,img)       
                        yield 1
                        if np.random.random() > 0.1: continue
                        for h3 in newhexlist3:
                            surround(h3,hexMap,img)
                            yield 1
 
def test():
    for i in [1,2,3]:
        yield 
    
if __name__ == "__main__":
    np.random.seed(0)
    delay = 100
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--straight',action='store_true')
    parser.add_argument('--random',action='store_true')
    parser.add_argument('--bloom',action='store_true')
    parser.add_argument('--record',default=None)
    parser.add_argument('--tile_size',default=20,type=int)
    
    args = parser.parse_args()
    writer = None
    imsize = (800,800)
    if args.record:
        writer = cv2.VideoWriter(args.record,cv2.VideoWriter_fourcc('M','P','4','V'),30,imsize)
    
    
    """ this section draws on hex in the center and then surrounds it
    img = np.zeros((imsize[1],imsize[1],3),dtype=np.uint8)
    
    org = tint((imsize[1]/2,imsize[0]/2))
    sz = 100
    h = Hexagon(org,sz)
    h.draw(img,facecolor=(51,215,255))
    cv2.imshow('s',img)
    cv2.waitKey(delay)
    
    for i in range(0,6):
        h0 = h.nextHex(i)
        h0.draw(img)            
        cv2.imshow('s',img)
        cv2.waitKey(delay)
    """
    
    # tesselate:
    org = (0,0) # TODO - randomize        
    sz = args.tile_size

    tesselation = HexagonalTesselation(org,sz,imsize)
    pts = tesselation.pts
    
    if args.straight:
        img = np.zeros((imsize[1],imsize[0],3),dtype=np.uint8)
        for frame in tesselation.raster(img):
            if writer:
                writer.write(img)
            cv2.imshow('tesselation',img)
            cv2.waitKey(1)
        if writer:
             writer.write(img)
        cv2.imshow('tesselation',img)
        cv2.waitKey(1000)
    
    if args.random:
        img = np.zeros((imsize[1],imsize[0],3),dtype=np.uint8)
        for frame in tesselation.random_emplace(img):
            if writer:
                writer.write(img)
            cv2.imshow('tesselation',img)
            cv2.waitKey(1)
        if writer:
            writer.write(img)
        cv2.imshow('tesselation',img)
        cv2.waitKey(1000)
    
        
    
    """
    
    backimg = deepcopy(img)
    wparams = [{'org':tint((imsize[0]/10,imsize[1]/4)),'text':'RACHEL IS KOZI',
    'font':cv2.FONT_HERSHEY_DUPLEX,'fontSize':2,'color':(0,0,255),'thickness':5},
    {'org':tint((imsize[0]/4,imsize[1]/3)),'text':'NICK JOHNSON',
    'font':cv2.FONT_HERSHEY_DUPLEX,'fontSize':2,'color':(0,0,255),'thickness':5},
    {'org':tint((imsize[0]/12,imsize[1]/2)),'text':'AND SURPRISE GUESTS',
    'font':cv2.FONT_HERSHEY_DUPLEX,'fontSize':2,'color':(0,0,255),'thickness':5},
    {'org':tint((imsize[0]/12,3*imsize[1]/4)),'text':"Steve's Basement",
    'font':cv2.FONT_HERSHEY_DUPLEX,'fontSize':1.4,'color':(0,0,0),'thickness':3},
    {'org':tint((imsize[0]/12,4*imsize[1]/5)),'text':"Aug 30",
    'font':cv2.FONT_HERSHEY_DUPLEX,'fontSize':1.4,'color':(0,0,0),'thickness':3},
    ]
    for w in wparams:
        img = deepcopy(backimg)
        for i in range(0,len(w['text'])):
            cv2.putText(img,w['text'][:i+1],w['org'],w['font'],w['fontSize'],w['color'],w['thickness'],cv2.LINE_AA)
            cv2.imshow('tesselation',img)
            cv2.waitKey(50)
        backimg = deepcopy(img)
    """     
    """
    fontOrg = tint((imsize[0]/10,imsize[1]/4))	
    cv2.putText(img,'RACHEL IS KOZI',fontOrg,cv2.FONT_HERSHEY_DUPLEX,2,(0,0,255),5,cv2.LINE_AA)
    fontOrg = tint((imsize[0]/4,imsize[1]/3))
    cv2.putText(img,'NICK JOHNSON',fontOrg,cv2.FONT_HERSHEY_DUPLEX,2,(0,0,255),5,cv2.LINE_AA)
    fontOrg = tint((imsize[0]/12,imsize[1]/2))
    cv2.putText(img,'AND SURPRISE GUESTS',fontOrg,cv2.FONT_HERSHEY_DUPLEX,2,(0,0,255),5,cv2.LINE_AA)
    
    fontOrg = tint((imsize[0]/12,3*imsize[1]/4))
    cv2.putText(img,"Steve's Basement",fontOrg,cv2.FONT_HERSHEY_DUPLEX,1.4,(0,0,0),3,cv2.LINE_AA)
    fontOrg = tint((imsize[0]/12,4*imsize[1]/5))
    cv2.putText(img,"Aug 30",fontOrg,cv2.FONT_HERSHEY_DUPLEX,1.4,(0,0,0),3,cv2.LINE_AA)
    """
            
    
    # "blooming":
    if args.bloom:
        img = np.zeros((imsize[1],imsize[0],3),dtype=np.uint8)
        for iframe in tesselation.bloom(img):
            delay = 20 if iframe else 20
            if writer:
                writer.write(img)
            cv2.imshow('tesselation',img)
            cv2.waitKey(delay)
        if writer:
            writer.write(img)
        cv2.imshow('tesselation',img)
        cv2.waitKey(1000)                    
    """
    hexMap = np.zeros((len(pts),3))
    for i in range(0,len(pts)):
        hexMap[i,1:3] = pts[i]
    
    #hexlist = []
    while np.sum(hexMap[0,:]) < len(pts):
        # choose bloom centers:
        nChoose = 3
        iNotFilled = np.nonzero(hexMap[:,0] == 0)[0]
        if len(iNotFilled) == 0: break
        iChoose = np.random.randint(0,len(iNotFilled),3)
        hexlist = []
        for i in iChoose:
            idx = iNotFilled[i]
            org = pts[idx]
            nextHex = Hexagon(org,sz)
            clr = (np.random.randint(40,60),np.random.randint(200,240),np.random.randint(210,255))
            nextHex.draw(img,facecolor=clr)
            hexMap[idx,0] = 1
            hexlist.append(nextHex)
                
        cv2.imshow('tesselation',img)
        cv2.waitKey(200)    

        # populate neighbors:
        for h in hexlist:
            newhexlist = surround(h,hexMap)
            cv2.imshow('tesselation',img)
            cv2.waitKey(20)          
            if np.random.random() > 0.5: continue
            for hnew in newhexlist:
                newhexlist2 = surround(hnew,hexMap)       
                cv2.imshow('tesselation',img)
                cv2.waitKey(20)
                if np.random.random() > 0.5: continue
                for h2 in newhexlist2:
                    newhexlist3 = surround(h2,hexMap)       
                    cv2.imshow('tesselation',img)
                    cv2.waitKey(20)
                    if np.random.random() > 0.1: continue
                    for h3 in newhexlist3:
                        surround(h3,hexMap)
                        cv2.imshow('tesselation',img)
                        cv2.waitKey(20)           
    """
    if writer:
        writer.release()
            