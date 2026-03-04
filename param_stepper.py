#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

class ParamStepper:
    def __init__(self,params):
        self.params = params # a dict, each entry is a name and a list, which is a list of parameter values
        self.ptrs = {}
        for i in self.params.keys():
            self.ptrs[i] = 0
            
    def update(self, name=None):
        if name is None:
            rval = []
            for name in self.params.keys():
                rval.append(self.params[name][self.ptrs[name]])
                self.__cycle(name)
        else:
            rval = self.params[name][self.ptrs[name]]
            self.__cycle(name)
        return rval

    def no_update(self, name=None):
        if name is None:
            rval = []
            for name in self.params.keys():
                rval.append(self.params[name][self.ptrs[name]])
        else:
            rval = self.params[name][self.ptrs[name]]
        return rval

    
    def __cycle(self, name):
        self.ptrs[name] += 1
        if self.ptrs[name] >= len(self.params[name]):
            self.ptrs[name] = 0


class ParamStepperDesigner:
    def __init__(self,min,max=None,n=None,name=None,waveform='triangle'):
        # pass a list for the first parameter to get non-linspaced values (max and n are then ignored)
        self.name = name
        if type(min) is list or isinstance(min, np.ndarray):
            x = min
        else:
            x = np.linspace(min,max,n)
        if waveform == 'triangle':
            y = np.flipud(x)[1:] # the first point would be repeated otherwise
            self.params = np.concatenate([x,y])
        elif waveform == 'upramp':
            self.params = x
        elif waveform == 'downramp':
            self.params =  np.flipud(x)

    def plot(self,ax=None):
        if ax is None:
            fig,ax = plt.subplots()
        ax.plot(np.linspace(n), self.params,label=self.name)
        return ax
        