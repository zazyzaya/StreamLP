import torch 
from torch import nn
from torch_scatter import (
    scatter_add, scatter_mean, 
    scatter_std, scatter_max, 
    scatter_min
)

'''
Implimenting TGBase temporal graph feature extractor from 
    https://epubs.siam.org/doi/pdf/10.1137/1.9781611977172.73

In the paper they beat SoTA on the BTC data set in terms of AUC
They claim that using their feature generator boosts AUC from 
0.955 -> 0.970 on GraphSAGE models (unclear what params)
trained to be binary classifiers 
'''
class TGBase(nn.Module):
    def __init__(self):
        super().__init__()

    def entropy(self, list):
        if list.size(0) == 0:
            return 0.0
        
        _,p = list.unique(return_counts=True)
        p = p.float()
        p /= p.sum()

        ent = -p*torch.log2(p)
        return ent.sum().item()

    @torch.no_grad() # parameter-free 
    def forward(self, ei, ts=None, feats=None, num_nodes=None):
        cols = [self.struct_feats(ei, num_nodes),]
            
        if feats is not None:
            if feats.dim() == 1:
                feats = feats.unsqueeze(-1)
            for i in range(feats.size(1)):
                cols.append(self.edge_feats(feats[:,i], ei, num_nodes))

        if ts is not None:
            cols.append(self.edge_feats(ts, ei, num_nodes))
        
        return torch.cat(cols, dim=1)

    def struct_feats(self, ei, num_nodes):
        if num_nodes is None:
            num_nodes = ei.max()+1
        
        # Used for degree counting    
        x = torch.ones(ei.size(1),1, dtype=torch.float)

        kwargs = dict(dim=0, dim_size=num_nodes)

        # Individual feats 
        in_deg = scatter_add(x, ei[1].unsqueeze(-1), **kwargs)
        out_deg = scatter_add(x, ei[0].unsqueeze(-1), **kwargs)
        tot_deg = in_deg+out_deg 

        structure_cols = [in_deg, out_deg, tot_deg]
        for i,val in enumerate([in_deg, out_deg, tot_deg]):
            if i == 1:
                ei = ei[[1,0]]
            elif i == 2: 
                ei = torch.cat([ei, ei[[1,0]]], dim=1)

            src = val[ei[0]]
            dst = ei[1].unsqueeze(-1)
            args = (src,dst)

            structure_cols += [
                scatter_add(*args,**kwargs),
                scatter_mean(*args, **kwargs),
                scatter_max(*args, **kwargs)[0],
                scatter_min(*args, **kwargs)[0],
                scatter_std(*args, **kwargs)
            ]

            # There's prob a more efficient way to do this but..
            lists = [[] for _ in range(num_nodes)]
            for j in range(ei.size(1)):
                lists[ei[1][j]].append(val[ei[0][j]])
            
            ent=[]
            for l in lists:
                ent.append([self.entropy(torch.tensor(l))])
            ent = torch.tensor(ent)
            structure_cols.append(ent)

        return torch.cat(structure_cols, dim=1)
    
    def edge_feats(self, val, ei, num_nodes):
        if val.dim() == 1:
            val = val.unsqueeze(-1)

        if num_nodes is None:
            num_nodes = ei.max()+1
        kwargs = dict(dim=0, dim_size=num_nodes)

        feat_cols = []
        for i in range(3):
            if i == 1:
                ei = ei[[1,0]]
            elif i == 2: 
                ei = torch.cat([ei, ei[[1,0]]], dim=1)
                val = val.repeat(2,1)

            src = val
            dst = ei[1].unsqueeze(-1)
            args = (src,dst)

            feat_cols += [
                scatter_add(*args,**kwargs),
                scatter_mean(*args, **kwargs),
                scatter_max(*args, **kwargs)[0],
                scatter_min(*args, **kwargs)[0],
                scatter_std(*args, **kwargs)
            ]

            # There's prob a more efficient way to do this but..
            lists = [[] for _ in range(num_nodes)]
            for j in range(ei.size(1)):
                lists[ei[1][j]].append(val[j])
            
            ent=[]
            for l in lists:
                ent.append([self.entropy(torch.tensor(l))])
            ent = torch.tensor(ent)
            feat_cols.append(ent)

        return torch.cat(feat_cols, dim=1)
    

class IncrimentalTGBase():
    def __init__(self, num_nodes, num_feats):
        '''
        TODO allow input precalculated x to initialize
        TODO include entropy (is this even possible?)
        '''
        self.num_nodes = num_nodes
        self.num_feats = num_feats

        args = (num_nodes, num_feats)
        self.S = torch.zeros(*args) # Keeps track of var_i * tot_i
        self.tot = torch.zeros(num_nodes,1)
        
        self.sum = torch.zeros(*args)
        self.min = torch.full(args, float('inf'))
        self.max = torch.full(args, float('-inf'))

    def get_value(self, idxs=None):
        if idxs is None:
            idxs = torch.arange(self.S.size(0))

        # Need to calculate std and mean from observed running totals
        mean = self.sum[idxs] / self.tot[idxs]
        std = torch.sqrt(self.S[idxs] / self.tot[idxs])

        # Concat all features 
        return torch.cat([
            self.sum[idxs],
            self.min[idxs],
            self.max[idxs],
            mean, 
            std 
        ], dim=-1)

    def __add_unseen(self, idx, feat):
        # More or less assumes ids are sequential. So it's unlikely
        # that it will get as edge 1 something like 0->1 then edge 
        # two will be 1,000,000 -> 9,999,999 e.g. (will still work, will just
        # waste a bunch of memory)
        dif = (idx+1) - self.tot.size(0)
        for mat in [self.S, self.tot, self.sum]:
            mat = torch.cat(
                [mat, torch.zeros(dif, mat.size(1))],
                dim=0 
            )
        
        # Variance remains at 0 as only one sample
        self.tot[idx] = 1
        self.sum[idx] = feat 

        # Max and min have to be treated a little differently 
        self.max = torch.cat([
            self.max, torch.full((dif, self.max.size(1)), float('-inf'))
        ])
        self.max[idx] = feat 

        self.min = torch.cat([
            self.min, torch.full((dif, self.min.size(1)), float('inf'))
        ])
        self.min[idx] = feat 

        return self.get_value(idx)


    def new_value(self, idx, feat):
        '''
        Need to update total, count, min/max 
        Other features may be derived from these values later
        TODO entropy
        '''
        # If this is an unseen id 
        if idx >= self.tot.size(0):
            return self.__add_unseen(idx, feat)

        prev_mean = self.sum[idx] / self.tot[idx]

        self.tot[idx] += 1
        self.sum[idx] += feat 
        self.max[idx] = torch.max(self.max[idx], feat)
        self.min[idx] = torch.min(self.min[idx], feat)

        cur_mean = self.sum[idx] / self.tot[idx]

        # Moving standard dev calculated by 
        # S_n = S_{n-1} + (x_n - mu_{n-1})(x_n - mu_n)
        # std_n = sqrt(S_n / n)
        # I don't think torch knows how to FOIL so expanding that,
        # the last term becomes: 
        # (x_n-mu_{n-1})(x_n - mu_n) = x_n^2 - x_n*mu_n - x_n*mu_{n-1} + mu_n*mu_{n-1}
        # which in variables we'll call a       b           c               d 
        a = torch.pow(feat,2)
        b = feat * cur_mean 
        c = feat * prev_mean 
        d = cur_mean * prev_mean
        self.S[idx] += a-b-c+d 

        return self.get_value(idx)
    
if __name__ == '__main__':
    vals = torch.eye(3).repeat(10,1)
    idx = torch.cat([torch.randperm(3) for _ in range(10)])
    itg = IncrimentalTGBase(1, 3)

    for i in range(30):
        itg.new_value(idx[i], vals[i])