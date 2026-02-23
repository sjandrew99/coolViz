#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os
from pickleLoad import *
import cv2

class Viewer:
    def __init__(self):
        self.dr = 'precomp'
        self.files = [i for i in os.listdir(self.dr) if i.endswith('.pkl')]
        self.da = np.array([float(i.split('_')[4]) for i in self.files])
        self.db = np.array([float(i.split('_')[6]) for i in self.files])
        self.terms = np.array([int(i.split('_')[8]) for i in self.files])
        
        # TODO - allow these to change:
        self.imsize = (900,800)
        self.k = 3
        
        self.iFile = 0
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('key_press_event',self.on_keypress)
        #fig = plt.figure(); ax = fig.add_axes([0,0,1,1])
        self.draw()
        
    def draw(self):
        self.ax.cla()
        self.fig.suptitle('da: %.05f db: %.05f terms: %03d' % (self.da[self.iFile],self.db[self.iFile],self.terms[self.iFile]))
        x = pickleLoad(self.dr + '/' + self.files[self.iFile])
        self.ax.imshow(x)
        self.fig.canvas.draw()
        cv2.imshow('img',x)
        cv2.waitKey(1)
    
    def on_keypress(self, event):
        bindings = {'a': [self.decrement_a, 'decrement da'],
                    'A': [self.increment_a, 'increment da'],
                    'b': [self.decrement_b, 'decrement db'],
                    'B': [self.increment_b, 'increment db'],
                    't': [self.decrement_terms, 'decrement terms'],
                    'T': [self.increment_terms, 'increment terms'],
                    'c': [self.draw_cv, 'draw current figure with opencv']
                   }
        if event.key == '?':
            for k in bindings:
                print(k + ': ' + bindings[k][1])
        else:
            for k in bindings:
                if event.key == k:
                    bindings[k][0]()
    
    def decrement_a(self):
        da_ = self.da[self.iFile]
        db_ = self.db[self.iFile]
        terms_ = self.terms[self.iFile]
        iPos = np.nonzero((self.db == db_) * (self.terms == terms_))[0] # TODO - check if empty
        #print(iPos)
        iLess = np.nonzero(self.da[iPos] < da_)[0] # TODO - check if empty
        #print(iLess)
        idx = np.argmax(self.da[iPos[iLess]])
        self.iFile = iPos[iLess][idx]
        self.draw()
        
    def increment_a(self):
        da_ = self.da[self.iFile]
        db_ = self.db[self.iFile]
        terms_ = self.terms[self.iFile]
        iPos = np.nonzero((self.db == db_) * (self.terms == terms_))[0] # TODO - check if empty
        iGreater = np.nonzero(self.da[iPos] > da_)[0] # TODO - check if empty
        idx = np.argmin(self.da[iPos[iGreater]])
        self.iFile = iPos[iGreater][idx]
        self.draw()
    def decrement_b(self):
        da_ = self.da[self.iFile]
        db_ = self.db[self.iFile]
        terms_ = self.terms[self.iFile]
        iPos = np.nonzero((self.da == da_) * (self.terms == terms_))[0] # TODO - check if empty
        iLess = np.nonzero(self.db[iPos] < db_)[0] # TODO - check if empty
        idx = np.argmax(self.db[iPos[iLess]])
        self.iFile = iPos[iLess][idx]
        self.draw()
    def increment_b(self):
        da_ = self.da[self.iFile]
        db_ = self.db[self.iFile]
        terms_ = self.terms[self.iFile]
        iPos = np.nonzero((self.da == da_) * (self.terms == terms_))[0] # TODO - check if empty
        iGreater = np.nonzero(self.db[iPos] > db_)[0] # TODO - check if empty
        idx = np.argmin(self.db[iPos[iGreater]])
        self.iFile = iPos[iGreater][idx]
        self.draw()
    def decrement_terms(self):
        da_ = self.da[self.iFile]
        db_ = self.db[self.iFile]
        terms_ = self.terms[self.iFile]
        iPos = np.nonzero((self.da == da_) * (self.db == db_))[0] # TODO - check if empty
        iLess = np.nonzero(self.terms[iPos] < terms_)[0] # TODO - check if empty
        idx = np.argmax(self.terms[iPos[iLess]])
        self.iFile = iPos[iLess][idx]
        self.draw()
    def increment_terms(self):
        da_ = self.da[self.iFile]
        db_ = self.db[self.iFile]
        terms_ = self.terms[self.iFile]
        iPos = np.nonzero((self.da == da_) * (self.db == db_))[0] # TODO - check if empty
        iGreater = np.nonzero(self.terms[iPos] > terms_)[0] # TODO - check if empty
        idx = np.argmin(self.terms[iPos[iGreater]])
        self.iFile = iPos[iGreater][idx]
        self.draw()
    
    def draw_cv(self):
        x = pickleLoad(self.dr + '/' + self.files[self.iFile])
        cv2.imshow('img',x)
        cv2.waitKey(1)
#name_str = '__weierstrass_da_%.05f_db_%.05f_terms_%03d_k_%02d_dw_%.03f_imsize_%d_%d.pkl'


viewer = Viewer()




plt.show()
