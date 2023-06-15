from collections import Counter
import math 

import numpy as np 
import torch 

class CategoricalAMC():
    '''
    Calculates the categorical assortative mixing coeficient
    as defined by Newman, 2003:
        https://journals.aps.org/pre/pdf/10.1103/PhysRevE.67.026126
    '''
    def __init__(self, n_classes=256):
        self.e = torch.zeros((n_classes, n_classes))

    def __check_e(self, c):
        if c > self.e.size(0):
            self.e = torch.cat([self.e, torch.zeros(self.e.size(0), self.e.size(1))], dim=0)
            self.e = torch.cat([self.e, torch.zeros(self.e.size(0), self.e.size(1))], dim=1) 
            self.__check_e(c)

    def add_edges(self, edges, ret=True):
        self.__check_e(edges.max())
        uq, cn = edges.unique(dim=1, return_counts=True)
        self.e[uq[0], uq[1]] += cn 

        if ret:
            return self.get_amc()

    def get_amc(self):
        e_norm = self.e / self.e.sum()
        squared_sum = (e_norm @ e_norm).sum()
        return (
            (e_norm.trace() - squared_sum) / 
            (1 - squared_sum)
        )

class ScalarAMC():
    '''
    Calculates the scalar assortative mixing coeficient
    '''
    def __init__(self, n_classes=256):
        self.a = Counter()
        self.b = Counter()
        self.e = Counter()
        self.n = 0

    def add_edges(self, edges, ret=True):
        src,dst = edges.numpy()
        
        self.a.update(src)
        self.b.update(dst)
        self.e.update(zip(src,dst))
        
        self.n += edges.size(1)

        if ret:
            return self.get_amc()

    def get_amc(self):
        numerator = 0 
        mu_a = sum([k*v for k,v in self.a.items()]) / self.n 
        mu_b = sum([k*v for k,v in self.b.items()]) / self.n 

        # Calc covariance
        for (x,y), f_xy in self.e.items():
            f_xy /= self.n 
            numerator += ((x-mu_a) * (y-mu_b) * f_xy)
        
        # Sum of squares
        ss_a = sum([v*((k-mu_a)**2) for k,v in self.a.items()])
        ss_b = sum([v*((k-mu_b)**2) for k,v in self.b.items()])
        
        std_a = math.sqrt(ss_a / self.n)
        std_b = math.sqrt(ss_b / self.n)
        denominator = std_a*std_b 

        return numerator / denominator

class SparseCategoricalAMC(ScalarAMC):
    def get_amc(self):
        max_class = max(
            max(self.a.keys()),
            max(self.b.keys())
        )

        tr = sum(self.e[(i,i)]/self.n for i in range(max_class+1))
        e_squared = sum([self.a[i]*self.b[i] for i in range(max_class+1)]) / (self.n**2)
        
        return (tr - e_squared) / (1 - e_squared)

edges = torch.tensor([
    [0, 1, 2, 0, 1, 2, 2, 3, 4, 1, 1],
    [4, 4, 0, 5, 10, 3, 10, 1, 2, 2, 3]
])

print(CategoricalAMC().add_edges(edges))
print(SparseCategoricalAMC().add_edges(edges))