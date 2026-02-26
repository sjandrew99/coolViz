#!/usr/bin/env python3

# generate a "blocky" svg font

import cv2
import os
import numpy as np

def render_string(frame, string, colors=None,glyphWidth = 40,glyphHeight = 40,line_spacing = 10,spacing = 10,left_margin = 0,top_margin = 10,centered=False):
    # units of input args are pixels
    glyphTop = top_margin; glyphLeft = left_margin
    if centered:
        assert '\n' not in string[0:-1] # must be a single line
        if string[-1] == '\n':
            wordlen = (len(string)-2) * (glyphWidth + spacing)
        else:
            wordlen = (len(string)-1) * (glyphWidth + spacing)
        glyphLeft = int((frame.shape[1] - wordlen - left_margin)/2 + left_margin)
        if glyphLeft < left_margin:
            print(f'WARNING - left margin {left_margin} will not accomodate centered string {string}')
    for i in string:
        clr = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
        if i == ' ':
            #pass
            glyphLeft += glyphWidth/2 + spacing
            glyphLeft = int(glyphLeft)
            continue
        elif i == '\n':
            #import pdb; pdb.set_trace()
            glyphTop += glyphHeight + line_spacing
            glyphLeft = left_margin
            continue
        else:
            glyph = letters[i.upper()]
            renderGlyph(frame[glyphTop:glyphTop+glyphHeight, glyphLeft:glyphLeft+glyphWidth, :], glyph, fillcolor=clr)
        glyphLeft += glyphWidth + spacing
    return glyphLeft, glyphTop

def render_string_wordwrapped(frame, string, colors=None,glyphWidth = 40,glyphHeight = 40,line_spacing = 15,spacing = 10,left_margin = 0,top_margin = 10):
    # units of input args are pixels
    _words = string.split(' ')
    # handle nelines, ex "stuff\nthings"
    words = []
    for i in range(0,len(_words)):
        if '\n' in _words[i]:
            lines=_words[i].split('\n')
            for k in range(0,len(lines)):
                words.append(lines[k])
                if k != len(lines)-1:
                    words.append('\n')
        else:
            words.append(_words[i])

    glyphTop = top_margin; glyphLeft = left_margin
    for word in words:
        # word wrap:
        wordLength = (len(word))*(glyphWidth+spacing) # +1 for the space
        if glyphLeft + wordLength > frame.shape[1]:
            glyphTop += glyphHeight + line_spacing
            glyphLeft = left_margin
        if word == '\n':
            glyphTop += glyphHeight + line_spacing
            glyphLeft = left_margin
            continue
        for i in word:
            clr = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
            if i == ' ':
                pass
            
            else:
                glyph = letters[i.upper()]
                renderGlyph(frame[glyphTop:glyphTop+glyphHeight, glyphLeft:glyphLeft+glyphWidth, :], glyph, fillcolor=clr)
            glyphLeft += glyphWidth + spacing
        glyphLeft += glyphWidth + spacing # insert a space


def renderGlyph(frame, glyph, linecolor=(255,255,255),fillcolor=(0,0,0),backgroundcolor=(0,0,0),closed=True,
    rotation = None): 

    margin = .01 # leave a margin around the border so anti-aliased lines always show up
    # render filled: TODO - fillPoly might be able to do this in one call
    for curve in glyph['curves']:
        points = curve['points']
        imsize = frame.shape
        pts = np.array(points).reshape((-1,1,2)) # N x 1 x 2
        # scale to [margin, 1-margin]:
        pts = pts*(1-2*margin) + margin
        if rotation is not None:
            pts = np.dot(rotation, pts)
        # scale to full frame:
        pts[:,:,0] = pts[:,:,0] * imsize[1]
        pts[:,:,1] = pts[:,:,1] * imsize[0]
        pts = pts.astype(np.int32)
        if curve['type'] == 'outer':
            cv2.fillPoly(frame, [pts], fillcolor)
            cv2.polylines(frame, [pts], closed, linecolor, 1,cv2.LINE_AA)
        else:
            cv2.fillPoly(frame, [pts], backgroundcolor)
            cv2.polylines(frame, [pts], closed, linecolor, 1,cv2.LINE_AA)

def get_point_on_line(pt1, pt2 = None, y=None, m=None,x=None):
    if m is None:
        m = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
        b = pt1[1] - m*pt1[0]
    if x is None:
        x = (y - b) / m
        return x
def gen_square(center, dist):
    top = center[1] - dist/2
    left = center[0] - dist/2
    right = center[0] + dist/2
    bottom = center[1] + dist/2
    return [[left,top], [right,top],[right,bottom],[left,bottom]]

def draw_curve(frame, points):
    line_color = (255,255,255)
    imsize = frame.shape
    pts = np.array(points).reshape((-1,1,2)) # N x 1 x 2
    # scale to [margin, 1-margin]:
    pts = pts*(1-2*margin) + margin
    # scale to full frame:
    pts[:,:,0] = pts[:,:,0] * imsize[1]
    pts[:,:,1] = pts[:,:,1] * imsize[0]
    pts = pts.astype(np.int32)
    cv2.polylines(frame, [pts], True, line_color, 1, cv2.LINE_AA)



# each letter is a set of curves. each curve is a list of points. each point is left-top, 0-1    
letters = {}
# A:
def glyphA():
    fullheight = 1; fullwidth = 1
    width = .2 # width of a leg
    legheight = .4
    inner_scale = 3/4
    inner_center_scale = 2/3 # vertical position of hole
    points = [[0,fullheight],[fullwidth/2,0],[fullwidth,fullheight]] # the "triangle"
    m = (points[1][1] - points[0][1]) / (points[1][0] - points[0][0]) # slope of first stroke. it's negative because of inverted y axis
    points.append([1-width, 1]) # bottom left of right leg
    y1 = 1 - legheight; x0 = 1-width; y0 = 1
    points.append([-(y1 - (y0 + m*x0))/m, y1]) # use the negative of m
    pt = [width, 1] # bottom right of right leg
    x0 = width
    points.append([(y1 - (y0 - m*x0))/m, y1])
    points.append(pt)
    # draw the inner hole:
    centerx = fullwidth / 2
    centery = (1 - legheight) * inner_center_scale
    # square. length of each side is half the distance from center to leg
    x = get_point_on_line(pt1=points[0], pt2 = points[1], y=centery)
    dist = (fullwidth/2) - x
    dist = dist * inner_scale
    points2 = gen_square([centerx,centery], dist)
    glyph = {'curves' : [
            {'points': points, 'type':'outer'},
            {'points': points2, 'type': 'inner'}]}
    return glyph


def glyphB():
    fullheight = 1; fullwidth = 1
    width = .2 # width of a leg
    #legheight = .4
    inner_scale = 1/4
    inner_center_scale = 1/4 # vertical position of hole
    inner_center_scale_x = 3/8
    indent_height = .2
    indent_width = .2

    points = [[0,fullheight],[0,0], [fullwidth, 0]]
    # triangular notch:
    points.append([fullwidth-indent_width, fullheight/2])
    # bottom right point:
    points.append([fullwidth, fullheight])
    # holes:
    centerx = fullwidth*inner_center_scale_x
    centery = fullheight * inner_center_scale
    points2 = gen_square([centerx,centery],inner_scale*fullwidth)
    points3 = gen_square([centerx,1-centery],inner_scale*fullwidth)
    glyph = {'curves' : [
            {'points': points, 'type':'outer'},
            {'points': points2, 'type':'inner'},
            {'points': points3, 'type':'inner'}]}
    return glyph

def glyphC():
    fullheight = 1; fullwidth = 1
    width = .2 # width of a leg
    points = [[0,fullheight],[0,0], [fullwidth, 0], [fullwidth-width, width], [width, width], [width,fullheight-width], [fullwidth-width, fullheight-width], [fullwidth, fullheight]]
    glyph = {'curves' : [
            {'points': points, 'type':'outer'}]}
    return glyph

def glyphD():
    fullheight = 1; fullwidth = 1
    bevel = .2
    scale = .4
    
    points = [[0,fullheight],[0,0], [fullwidth-bevel, 0], [fullwidth, bevel], [fullwidth, fullheight-bevel], [fullwidth-bevel, fullheight]]
    points2 = gen_square([fullwidth/2, fullheight/2], fullwidth*scale)
    glyph = {'curves' : [
            {'points': points, 'type':'outer'},
            {'points': points2, 'type':'inner'},]}
    return glyph

def glyphE():
    fullheight = 1; fullwidth = 1
    bevel = .2
    width = .2
    
    points = [[0,fullheight],[0,0], [fullwidth-bevel, 0], [fullwidth, bevel], [width, bevel]]
    blank_dist = (fullheight - 3*width)/2
    middle_top = width + blank_dist
    points.append([width, middle_top])
    points.append([fullwidth-width, middle_top])
    points.append([fullwidth, middle_top+width])
    points.append([width, middle_top+width])
    points.extend([[width,2*width + 2*blank_dist],[fullwidth-bevel,2*width+2*blank_dist]])
    points.append([fullwidth, fullheight])
    glyph = {'curves' : [
            {'points': points, 'type':'outer'}]}
    return glyph

def glyphF():
    fullheight = 1; fullwidth = 1
    bevel = .2
    width = .2
    
    points = [[0,fullheight],[0,0], [fullwidth-bevel, 0], [fullwidth, bevel], [width, bevel]]
    blank_dist = (fullheight - 3*width)/2
    middle_top = width + blank_dist
    points.append([width, middle_top])
    points.append([fullwidth-width, middle_top])
    points.append([fullwidth, middle_top+width])
    points.append([width, middle_top+width])
    points.extend([[width,2*width + 2*blank_dist]])
    points.append([width, fullheight])
    glyph = {'curves' : [
            {'points': points, 'type':'outer'}]}
    return glyph

def glyphG():
    fullheight = 1; fullwidth = 1
    bevel = .2
    width = .2 
    tailheight = .1
    tailwidth = .1
               # bottom left   # tl   # tr
    points = [[0,fullheight], [0,0], [fullwidth, 0]]
    points.append([fullwidth, width])
    points.append([width, width])
    points.append([width, fullheight-width])
    points.append([fullwidth - width, fullheight-width])
    points.append([fullwidth - width, fullheight-width-tailheight])
    points.append([fullwidth-2*width, fullheight-width-tailheight])
    points.append([fullwidth-2*width, fullheight-width-tailheight-tailwidth])
    points.append([fullwidth, fullheight-width-tailheight-tailwidth])
    points.append([fullwidth, fullheight])
    glyph = {'curves' : [
            {'points': points, 'type':'outer'}]}
    return glyph

def glyphH():
    fullheight = 1; fullwidth = 1
    width = .2 
    vpos = (fullheight - width)/2
    points = [[0,fullheight], [0,0], [width, 0], [width, vpos], [fullwidth-width, vpos], [fullwidth-width, 0], [fullwidth, 0]]
    points.extend([[fullwidth, fullheight], [fullwidth - width, fullheight], [fullwidth-width, fullheight-vpos], [width, fullheight-vpos],[width, fullheight ]])
    glyph = {'curves' : [
            {'points': points, 'type':'outer'}]}
    return glyph
    
def glyphI():
    fullheight = 1; fullwidth = 1
    width = .2 # top width
    mwidth = .1
    hpos = (fullwidth - mwidth)/2
    points = [[0,0], [fullwidth, 0], [fullwidth, width], [hpos+mwidth, width], [hpos+mwidth, fullheight-width], [fullwidth, fullheight-width], [fullwidth, fullheight]]
    points.extend([[0,fullheight], [0,fullheight-width], [hpos, fullheight-width], [hpos, width], [0, width]])
    glyph = {'curves' : [
            {'points': points, 'type':'outer'}]}
    return glyph

def glyphJ():
    fullheight = 1; fullwidth = 1
    width = .2 # top width
    mwidth = .1
    hpos = (fullwidth - mwidth)/2
    points = [[0,0], [fullwidth, 0], [fullwidth, width], [hpos+mwidth, width], [hpos+mwidth, fullheight], [0, fullheight], [0,fullheight-width], [hpos, fullwidth-width], [hpos, width], [0, width]]
    glyph = {'curves' : [
            {'points': points, 'type':'outer'}]}
    return glyph

def glyphK():
    fullheight = 1; fullwidth = 1
    width = .2
    spacing = .1
    vpos = (fullheight - width)/2
    points = [[0,fullheight],[0,0],[width,0],[width, vpos],[fullwidth, 0], [fullwidth, width], [2*width, vpos+spacing]]#, [2*width, vpos+spacing]]
    points.extend([[fullwidth, fullheight-width], [fullwidth, fullheight], [width, vpos+2*spacing], [width, fullheight]])
    glyph = {'curves' : [
            {'points': points, 'type':'outer'}]}
    return glyph

def glyphL():
    fullheight = 1; fullwidth = 1
    width = .2
    points = [[0,fullheight],[0,0],[width,0],[width, fullheight-width],[fullwidth, fullheight-width], [fullwidth, fullheight]]
    glyph = {'curves' : [
            {'points': points, 'type':'outer'}]}
    return glyph

def glyphM():
    fullheight = 1; fullwidth = 1
    width = .2
    depth = .4
    points = [[0,fullheight],[0,0],[width,0],[fullwidth/2, depth],[fullwidth-width, 0], [fullwidth, 0], [fullwidth, fullheight], [fullwidth-width, fullheight]]
    points.extend([[fullwidth-width, width]]) 
    points.extend([[fullwidth/2, depth+width]])
    points.append([width, width])
    points.append([width, fullheight])
    glyph = {'curves' : [
            {'points': points, 'type':'outer'}]}
    return glyph

def glyphN():
    fullheight = 1; fullwidth = 1
    width = .2
    depth = .4
    points = [[0,fullheight],[0,0],[width,0],[fullwidth-width,fullheight-width],[fullwidth-width, 0], [fullwidth, 0], [fullwidth, fullheight]]
    points.extend([[fullwidth-width, fullheight], [width,width], [width, fullheight]])
    glyph = {'curves' : [
            {'points': points, 'type':'outer'}]}
    return glyph
def glyphO():
    fullheight = 1; fullwidth = 1
    bevel = .2
    scale = .6
    
    points = [[0,fullheight-bevel],[0,bevel], [bevel, 0], [fullwidth-bevel, 0], [fullwidth, bevel], [fullwidth, fullheight-bevel], [fullwidth-bevel, fullheight], [bevel, fullheight]]
    points2 = gen_square([fullwidth/2, fullheight/2], fullwidth*scale)
    glyph = {'curves' : [
            {'points': points, 'type':'outer'},
            {'points': points2, 'type':'inner'},]}
    return glyph

def glyphP():
    fullheight = 1; fullwidth = 1
    width = .2
    scale = 1/2
    holescale = 1/8
    
    points = [[0,fullheight],[0,0], [fullwidth, 0], [fullwidth, scale*fullheight], [width, scale*fullheight], [width, fullheight]]
    centerx = fullwidth/2
    centery = scale*fullheight / 2
    points2 = gen_square([centerx, centery], holescale*fullheight)
    glyph = {'curves' : [
            {'points': points, 'type':'outer'},
            {'points': points2, 'type' : 'inner'}]}
    return glyph

def glyphQ():
    fullheight = 1; fullwidth = 1
    width = .2
    scale = 0.5
    tailheight = .15
    
    points = [[0,fullheight-tailheight],[0,0], [fullwidth-tailheight, 0], [fullwidth-tailheight, fullheight-2*tailheight], [fullwidth, fullheight-tailheight]]
    points.extend([[fullwidth-tailheight, fullheight], [fullwidth-2*tailheight, fullheight-tailheight]])
    points2 = gen_square([(fullwidth-tailheight)/2, (fullheight-tailheight)/2], fullwidth*scale)
    glyph = {'curves' : [
            {'points': points, 'type':'outer'},
            {'points': points2, 'type' : 'inner'}]}
    return glyph

def glyphR():
    fullheight = 1; fullwidth = 1
    width = .2
    scale = 0.5
    hpos =.4# leftmost point of hole
    points = [[0,fullheight],[0,0], [fullwidth, 0], [fullwidth, fullheight*scale], [hpos, fullheight*scale]]
    points.extend([[fullwidth, fullheight], [fullwidth-width, fullheight], [hpos/2, fullheight*scale], [width, fullheight]])
    points2 = gen_square([hpos, scale/2], scale/4)
    glyph = {'curves' : [
            {'points': points, 'type':'outer'},
            {'points': points2, 'type' : 'inner'}]}
    return glyph

def glyphS():
    fullheight = 1; fullwidth = 1
    width = .2
    blankdist = (fullheight - 3*width)/2
    vpos = 2*width + blankdist # lower middle
    points = [[0,fullheight],[0,fullheight-width], [fullwidth-width, fullheight-width],[fullwidth-width, vpos], [0, vpos], [0, 0], [fullwidth, 0]]
    points.extend([[fullwidth, width], [width, width], [width, width+blankdist], [fullwidth, width+blankdist], [fullwidth, fullheight]])
    glyph = {'curves' : [
            {'points': points, 'type':'outer'}]}
    return glyph

def glyphT():
    fullheight = 1; fullwidth = 1
    width = .2
    hpos = (fullwidth-width)/2
    points = [[hpos,fullheight],[hpos,width], [0, width],[0, 0], [fullwidth, 0], [fullwidth, width], [hpos+width, width], [hpos+width, fullheight]]
    glyph = {'curves' : [
            {'points': points, 'type':'outer'}]}
    return glyph
def glyphU():
    fullheight = 1; fullwidth = 1
    width = .2
    points = [[0,fullheight],[0,0], [width,0],[width,fullheight-width], [fullwidth-width, fullheight-width], [fullwidth-width, 0], [fullwidth, 0], [fullwidth,fullheight]]
    glyph = {'curves' : [
            {'points': points, 'type':'outer'}]}
    return glyph

def glyphV():
    fullheight = 1; fullwidth = 1
    width = .2
    points = [[0,0], [width,0],[fullwidth/2,fullheight-width], [fullwidth-width, 0], [fullwidth, 0], [fullwidth/2, fullheight]]
    glyph = {'curves' : [
            {'points': points, 'type':'outer'}]}
    return glyph
def glyphW():
    fullheight = 1; fullwidth = 1
    width = .2
    h = .3 #height of the top of the "spike"
    h2 = .3 # vertical distace between top and bottom of spike
    d = .2 # distance between "feet"
    hpos = (fullwidth-d)/2
    points = [[0,0], [width,0],[hpos, h+h2], [hpos+d/2,h], [hpos+d,h+h2],[fullwidth-width, 0],[fullwidth,0]]
    points.extend([[hpos+d,fullheight], [hpos+d/2, h+width], [hpos, fullheight]])
    glyph = {'curves' : [
            {'points': points, 'type':'outer'}]}
    return glyph
def glyphX():
    fullheight = 1; fullwidth = 1
    width = .2
    depth = .45
    points = [[0,0],[width,0],[fullwidth/2, depth], [fullwidth-width, 0], [fullwidth, 0]]
    points.extend([[fullwidth-depth, fullheight/2], [fullwidth, fullheight], [fullwidth-width, fullheight]])
    points.extend([[fullwidth/2, fullheight-depth], [width, fullheight],[0,fullheight]])
    points.extend([[depth, fullheight/2]])
    glyph = {'curves' : [
            {'points': points, 'type':'outer'}]}
    return glyph

def glyphY():
    fullheight = 1; fullwidth = 1
    width = .2
    depth = .3
    points = [[0,0],[width,0],[fullwidth/2, depth], [fullwidth-width, 0], [fullwidth, 0]]
    points.extend([[fullwidth/2, fullheight], [(fullwidth/2)-width, fullheight], [fullwidth/2, depth+width]])
    glyph = {'curves' : [
            {'points': points, 'type':'outer'}]}
    return glyph

def glyphZ():
    fullheight = 1; fullwidth = 1
    width = .2
    points = [[0,0],[fullwidth,0],[width,fullheight-width],[fullwidth,fullheight-width]]
    points.extend([[fullwidth, fullheight], [0, fullheight], [fullwidth-2*width, width],[0,width]])
    glyph = {'curves' : [
            {'points': points, 'type':'outer'}]}
    return glyph

def glyphPERIOD():
    fullheight = 1; fullwidth = 1
    width = .2
    points = [[0,fullheight],[0,fullheight-width],[width,fullheight-width],[width,fullheight]]
    glyph = {'curves' : [
            {'points': points, 'type':'outer'}]}
    return glyph

def glyph1():
    fullheight = 1; fullwidth = 1
    width = .2
    hpos = (fullwidth-width)/2
    points = [[0,fullheight],[0,fullheight-width],[hpos,fullheight-width],[hpos,width], [0, fullheight/2], [hpos, 0],[hpos+width,0], 
              [hpos+width, fullheight-width],[fullwidth,fullheight-width], [fullwidth, fullheight]]
    glyph = {'curves' : [
            {'points': points, 'type':'outer'}]}
    return glyph

def glyph4():
    fullheight = 1; fullwidth = 1
    width = .2
    vpos = (fullheight-width)/3
    points = [[0,0],[width,0],[width,vpos], [fullwidth-width, vpos],[fullwidth-width, 0], [fullwidth, 0],
    [fullwidth,fullheight], [fullwidth-width, fullheight], [fullwidth-width, vpos+width], [0,vpos+width]]
    glyph = {'curves' : [
            {'points': points, 'type':'outer'}]}
    return glyph


def glyph5():
    return glyphS() # screw it

def glyph7():
    fullheight = 1; fullwidth = 1
    width = .3
    points = [[0,0],[fullwidth,0],[width,fullheight],[0,fullheight], [fullwidth-width, width], [0,width]]
    glyph = {'curves' : [
            {'points': points, 'type':'outer'}]}
    return glyph    

letters['A'] = glyphA()
letters['B'] = glyphB()
letters['C'] = glyphC()
letters['D'] = glyphD()
letters['E'] = glyphE()
letters['F'] = glyphF()
letters['G'] = glyphG()
letters['H'] = glyphH()
letters['I'] = glyphI()
letters['J'] = glyphJ()
letters['K'] = glyphK()
letters['L'] = glyphL()
letters['M'] = glyphM()
letters['N'] = glyphN()
letters['O'] = glyphO()
letters['P'] = glyphP()
letters['Q'] = glyphQ()
letters['R'] = glyphR()
letters['S'] = glyphS()
letters['T'] = glyphT()
letters['U'] = glyphU()
letters['V'] = glyphV()
letters['W'] = glyphW()
letters['X'] = glyphX()
letters['Y'] = glyphY()
letters['Z'] = glyphZ()
letters['.'] = glyphPERIOD()
letters['1'] = glyph1()
letters['4'] = glyph4()
letters['5'] = glyph5()
letters['7'] = glyph7()

if __name__ == "__main__":
    imsize = (900, 800)    
    """
    frame = np.zeros((imsize[1],imsize[0],3),dtype=np.uint8)
    renderGlyph(frame, letters['4'], fillcolor=(255,0,0))
    #renderGlyph(frame, letters['4'], closed=False)
    cv2.imshow('stuff',frame)
    cv2.waitKey(1000)
    #sys.exit(1)
    while 1:
        for k in letters:
            frame = np.zeros((imsize[1],imsize[0],3),dtype=np.uint8)
            renderGlyph(frame, letters[k], fillcolor=(255,0,0))
            cv2.imshow('stuff',frame)
            cv2.waitKey(1000)
    """
    string = 'The quick red fox\njumped over the lazy\nbrown dog'
    frame = np.zeros((imsize[1],imsize[0],3),dtype=np.uint8)
    render_string(frame,string)
    cv2.imshow('render string',frame)
    #cv2.waitKey(0)
    string = 'The quick red fox jumped over the lazy brown dog'
    frame = np.zeros((imsize[1],imsize[0],3),dtype=np.uint8)
    render_string_wordwrapped(frame, string)
    cv2.imshow('wordwrap',frame)
    
    string = 'Quick red fox'
    frame = np.zeros((imsize[1],imsize[0],3),dtype=np.uint8)
    circle_center = (imsize[0]/2, imsize[1]/2) # x, y 
    circle_radius = 400
    circle_radius2 = (1/2)*circle_radius
    #glyphWidth = 40; 
    #glyphHeight = 40; line_spacing = 10; spacing = 10; left_margin = 0;top_margin = 10;
    max_angle = 30 # degrees
    # top:
    dtheta = np.pi/180 * max_angle * 2 / len(string)
    theta = np.arange((90+max_angle)*np.pi/180, (90-max_angle)*np.pi/180 - dtheta, -dtheta) # use -dtheta because i want to start at the left
    #arc_len = len(string) * (glyphWidth + spacing) # i think
    #arc_len = np.pi * (circle_radius*2) * (max_angle/360)
    #posx = circle_center[0] - arc_len/2
    #glyphTop = top_margin; glyphLeft = left_margin
    for i in range(0,len(string)):
        clr = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
        if string[i] == ' ':
            pass
        else:
            glyph = letters[string[i].upper()]
            posx = circle_radius * np.cos(theta[i]) + circle_center[0]
            posy = -circle_radius * np.sin(theta[i]) + circle_center[1] 
            posx = int(posx); posy = int(posy)
            nextx = circle_radius2 * np.cos(theta[i+1]) + circle_center[0]
            nexty = -circle_radius2 * np.sin(theta[i+1]) + circle_center[1] 
            nextx = int(nextx); nexty = int(nexty)
            #renderGlyph(frame[posy:posy+glyphHeight, posx:posx+glyphWidth, :], glyph, fillcolor=clr) # TODO - figure out rotated width and height
            renderGlyph(frame[posy:nexty, posx:nextx, :], glyph, fillcolor=clr) # TODO - figure out rotated width and height
            print(i, posx, posy, nextx, nexty)
            if i == 1:  break
        #glyphLeft += glyphWidth + spacing
    
    cv2.imshow('circular',frame)
    cv2.waitKey(0)