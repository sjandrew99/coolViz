#!/usr/bin/env python3

class CmapStepper:
    def __init__(self):
        self.counter = 0
        self.dcmap = 1
    def update(self):
        if self.counter == 255:
            self.dcmap = -1
        elif self.counter == 0:
            self.dcmap = 1
        self.counter += self.dcmap