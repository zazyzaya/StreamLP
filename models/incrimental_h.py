from collections import Counter
from math import log2 

import torch 

'''
Sloppy pre-C++ (?) implimentation of this

H(x) = -sum_n (x_i log (x_i / c)) = -1/c sum_n[x_i log(x_i)] + n/c log(c)

                                   /                                                          \
H(x') if x_i in [0, n] = -1/(c+1) ( sum_n [x_i log(x_i)] - x_i log(x_i) + (x_i + 1)log(x_i +1) )
                                   \                                                          /
                         + n/(c+1) log(c+1)
                                       /                               \
H(x') if x_i not in [0,n] = -1/(c+1)  ( sum_n [x_i log(x_i)] + 1 log(1) )  + (n+1)/c log(c)
                                       \                               /
'''

class StreamEntropy_SingleDim():
    def __init__(self):
        self.xlogx = 0
        self.c = 0
        self.n = 0
        self.counter = Counter()

    def add(self, item):
        if item not in self.counter:
            self.n += 1
        else:
            x_i = self.counter[item]
            self.xlogx += -x_i*log2(x_i) + (x_i+1)*log2(x_i+1)

        self.c += 1 
        self.counter.update([item])

    def get(self):
        return (-(1/self.c) * self.xlogx) + log2(self.c)
    
class StreamEntropy():
    def __init__(self, n_feats):
        self.xlogx = torch.zeros(n_feats)
        self.n = torch.zeros(n_feats)
        self.c = 0
        self.counter = [Counter() for _ in range(n_feats)]

        self.n_feats = n_feats

    def add(self, feat):
        for i in range(self.n_feats):
            item = feat[i].item() 
            
            if item not in self.counter[i]:
                self.n[i] += 1 
            else:
                x_i = self.counter[i][item]
                self.xlogx[i] += -x_i*log2(x_i) + (x_i+1)*log2(x_i+1)
            
            self.counter[i].update([item])
        self.c += 1

    def update(self, old, new, idx):
        '''
        Move element from one set to another
        E.g. if list goes from [1,1,2,1] to [1,2,2,1] (1 is old 2 is new)
        then counter should go from {1:3, 2:1} to {1:2, 2:2}
        Importantly, this does not update the number of bag elements
        it simply removes one and adds one
        '''
        self.counter[idx] += Counter({old: -1, new: 1})

    def get(self):
        if self.c == 0: return torch.tensor([0] * self.n_feats)
        ret = (-(1/self.c) * self.xlogx) + log2(self.c)

        # Issues with floating points
        ret[ret < 1e-7] = 0
        return ret 

if __name__ == '__main__':
    se = StreamEntropy(3)

    # Distr as  x_0 ~ 2/3 -> H=0.92
    #           x_1 ~ 1.  -> H=0
    #           x_2 ~ 1/2 -> H=1
    t = torch.tensor([
        [1,1,1],
        [1,1,0],
        [0,1,1],
        [1,1,0],
        [1,1,1],
        [0,1,0]
    ]).float()

    for i in range(t.size(0)):
        se.add(t[i])

    print('Calculated:', se.get())
    print('Correct:\t', [0.92, 0.0, 1.0])