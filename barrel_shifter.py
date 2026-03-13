class BarrelShifter:
    def __init__(self,n):
        self.max = n
        self.queue = []
    def push(self,element):
        #self.queue.append(deepcopy(element))
        self.queue.append(element)
        if len(self.queue) > self.max:
            del self.queue[0]
    def isFull(self):
        return len(self.queue) >= self.max

