#!/usr/bin/env python3


    
def detect_border_collision(center_x,center_y,size_x,size_y,bounding_box):
    # size x/y are max "radii" in that direction
    # bounding box is left, top, width, height of the frame
    # detect if not overlap (ie, object is leaving frame)
    # returns a two-list for collisions with left/right, top/bottom. +1 is collision with right/bottom, -1 is collision with left/top, 0 is no collision
    bleft = bounding_box[0]
    bright = bounding_box[0] + bounding_box[2]
    btop = bounding_box[1]
    bbottom = bounding_box[1] + bounding_box[3]
    collisions = [0,0] # left/right,top/bottom
    if center_x + size_x >= bright:
        collisions[0] = 1
    elif center_x - size_x <= bleft:
        collisions[0] = -1
    if center_y + size_y >= bbottom:
        collisions[1] = 1
    elif center_y - size_y <= btop:
        collisions[1] = -1
    return collisions

def overlaps(x,a,b):
    return a <= x <= b

def detect_object_collision(center_x1,center_y1,size_x1,size_y1,
                            center_x2,center_y2,size_x2,size_y2):
    # size x/y are max "radii" in that direction
    # check if objects overlap. 
    # returns a two-list for collisions with left/right, top/bottom. +1 is collision with right/bottom, -1 is collision with left/top, 0 is no collision
    # collisions[0] == +1 means object 1 is left of object 2
    collisions = [0,0] # left/right,top/bottom
    overlap_x = 0; overlap_y = 0
    # check for overlap in x:
    if overlaps(center_x1-size_x1,center_x2-size_x2,center_x2+size_x2) or \
       overlaps(center_x1,center_x2-size_x2,center_x2+size_x2) or \
       overlaps(center_x1+size_x1,center_x2-size_x2,center_x2+size_x2) or \
       overlaps(center_x2-size_x2,center_x1-size_x1,center_x1+size_x1) or \
       overlaps(center_x2,center_x1-size_x1,center_x1+size_x1) or \
       overlaps(center_x2+size_x2,center_x1-size_x1,center_x1+size_x1):
        overlap_x = 1
    # overlap in y:
    if overlaps(center_y1-size_y1,center_y2-size_y2,center_y2+size_y2) or \
       overlaps(center_y1,center_y2-size_y2,center_y2+size_y2) or \
       overlaps(center_y1+size_y1,center_y2-size_y2,center_y2+size_y2) or \
       overlaps(center_y2-size_y2,center_y1-size_y1,center_y1+size_y1) or \
       overlaps(center_y2,center_y1-size_y1,center_y1+size_y1) or \
       overlaps(center_y2+size_y2,center_y1-size_y1,center_y1+size_y1):
        overlap_y = 1
    
    if overlap_x and overlap_y:
        collisions[0] = 1 if center_x1 <= center_x2 else -1
        collisions[1] = 1 if center_y1 <= center_y2 else -1
    return collisions
        
    """
    lr = -1 # +1 if object 1 is right of object 2
    tb = -1 # +1 if object 1 is lower than object 2
    if center_x1 >= center_x2:
        lr = 1
    if center_t1 >= center_y2:
        tb = 1
        
    if (center_x1 + size_x1) >= (center_x2-size_x2) and (center_x1 + size_x) <= (center_x2 + size_x2):
            collisions[0] = 1
    elif (center_x1 - size_x1) >= (center_x2-size_x2) and (center_x1 - size_x1) <= (center_x2+size_x2):
            collisions[0] = -1
    if center_y + size_y >= btop and center_y + size_y <= bbottom:
            collisions[1] = 1 
    elif center_y - size_y >= btop and center_y - size_y <= bbottom:
            collisions[1] = -1 
    if not np.all(collisions):
        # have to overlap in both dimensions for a collision
        collisions = [0,0]
    """
    return collisions

def detect_object_collision2(center_x1,center_y1,size_1,
                            center_x2,center_y2,size_2):
    # sizes are radii. assume spherical objects, and don't care which direction collision came from
    # check if objects overlap. 
    overlap_x = 0; overlap_y = 0
    # check for overlap in x:
    if overlaps(center_x1-size_1,center_x2-size_2,center_x2+size_2) or \
       overlaps(center_x1,center_x2-size_2,center_x2+size_1) or \
       overlaps(center_x1+size_1,center_x2-size_2,center_x2+size_2) or \
       overlaps(center_x2-size_2,center_x1-size_1,center_x1+size_1) or \
       overlaps(center_x2,center_x1-size_1,center_x1+size_1) or \
       overlaps(center_x2+size_2,center_x1-size_1,center_x1+size_1):
        overlap_x = 1
    # overlap in y:
    if overlaps(center_y1-size_1,center_y2-size_2,center_y2+size_2) or \
       overlaps(center_y1,center_y2-size_2,center_y2+size_2) or \
       overlaps(center_y1+size_1,center_y2-size_2,center_y2+size_2) or \
       overlaps(center_y2-size_2,center_y1-size_1,center_y1+size_1) or \
       overlaps(center_y2,center_y1-size_1,center_y1+size_1) or \
       overlaps(center_y2+size_2,center_y1-size_1,center_y1+size_1):
        overlap_y = 1
    
    collision = overlap_x and overlap_y
    return collision