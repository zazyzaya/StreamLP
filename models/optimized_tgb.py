from collections import defaultdict
from collections.abc import Iterable

import torch 
from torch import nn
from torch_scatter import (
    scatter_add, scatter_mean, 
    scatter_std, scatter_max
)
from torch_geometric.utils import to_undirected

'''
After testing removing various aggregators from vanilla
TGBase, we found that removing entropy and min improves 
classification accuracy w/ p>0.05 

    WIKI : 
        All aggrs:  0.884 +/- 0.0010
        No min/ent: 0.902 +/- 0.0001
    REDDIT: 
        All aggrs:  0.722 +/- 0.0019
        No min/ent: 0.733 +/- 0.0012

'''
class TGBase():
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
                ei = to_undirected(ei)

            src = val[ei[0]]
            dst = ei[1].unsqueeze(-1)
            args = (src,dst)

            structure_cols += [
                scatter_add(*args,**kwargs),
                scatter_mean(*args, **kwargs),
                scatter_max(*args, **kwargs)[0],
                scatter_std(*args, **kwargs)
            ]

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
                scatter_std(*args, **kwargs)
            ]
            
        return torch.cat(feat_cols, dim=1)
    

class StreamTGBaseEncoder():
    def __init__(self, num_feats, num_nodes):
        self.num_feats = num_feats

        args = (num_nodes, num_feats)
        self.S = torch.zeros(*args) # Keeps track of var_i * tot_i
        self.tot = torch.zeros(num_nodes,1)
        
        self.sum = torch.zeros(*args)
        self.max = torch.full(args, float('-inf'))

    def get_value(self, idxs=None):
        if idxs is None:
            idxs = torch.arange(self.S.size(0))

        if not isinstance(idxs, Iterable) or (isinstance(idxs, torch.Tensor) and idxs.dim()==0):
            idxs = [idxs]

        # Need to calculate std and mean from observed running totals
        mean = (self.sum[idxs] / self.tot[idxs]).nan_to_num(0)
        std = torch.sqrt(self.S[idxs] / self.tot[idxs]).nan_to_num(0)

        to_tensor = [
            self.sum[idxs],
            mean, 
            self.max[idxs],
            std 
        ]

        # Concat all features 
        return torch.cat(to_tensor, dim=-1).nan_to_num(0,0,0)

    def __getitem__(self, idx):
        return self.get_value(idx)

    def __add_unseen(self, idx):
        # More or less assumes ids are sequential. So it's unlikely
        # that it will get as edge 1 something like 0->1 then edge 
        # two will be 1,000,000 -> 9,999,999 e.g. (will still work, will just
        # waste a bunch of memory)
        dif = (idx+1) - self.tot.size(0)

        for key in ['S', 'tot', 'sum']:
            # Can't pass by reference w/o doing this?
            self.__dict__[key] = torch.cat([
                self.__dict__[key], 
                torch.zeros(dif, self.__dict__[key].size(1))
            ], dim=0)

        # Inf values are slipping into the data. Not sure how the OG
        # authors delt with this, it appears to still be in their code
        # but I'm just initializing to 0 
        # UPDATE: changed get_value to have .nan_to_num to overwrite inf values to 0
        self.max = torch.cat([
            self.max, 
            torch.full((dif, self.max.size(1)), float('-inf'))
        ])

    def add(self, idx, feat):
        '''
        Need to update total, count, min/max 
        Other features may be derived from these values later
        TODO entropy
        '''
        if (isinstance(idx, torch.Tensor) and idx.dim() > 0) \
            or isinstance(idx, list):
            check = max(idx) # type: ignore
        else:
            check = idx 
            idx = [idx]

        # If this is an unseen id 
        if check >= self.tot.size(0):
            self.__add_unseen(check)

        # In case hasn't been seen before
        prev_mean = (self.sum[idx] / self.tot[idx]).nan_to_num(0)

        self.tot[idx] += 1
        self.sum[idx] += feat 
        self.max[idx] = torch.max(self.max[idx], feat)

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

class BatchTGBaseEncoder(StreamTGBaseEncoder):
    '''
    Note:   the field S now denotes M2 in Welford's method of calculating running variance
            (which, I'm pretty sure is the same quantity... M2/n = S/n = variance)
    See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    and https://stackoverflow.com/questions/56402955/whats-the-formula-for-welfords-algorithm-for-variance-std-with-batch-updates
    '''
    def add(self, idxs, feats):    
        check = max(idx) # type: ignore

        # If this is an unseen id 
        if check >= self.tot.size(0):
            self.__add_unseen(check)

        prev_mean = (self.sum[idxs] / self.tot[idxs]).nan_to_num(0)

        n_updates = scatter_add(torch.ones(idxs.size(0),1), idxs, dim=0)
        self.tot[idxs] += n_updates 
        self.sum[idxs] += scatter_add(feats, idxs, dim=0)
        self.max[idxs] = torch.max(self.max[idxs], scatter_max(feats, idxs, dim=0)[0])

        cur_mean = self.sum[idxs] / self.tot[idxs]

        '''
        Using Welford's method to calculate running std
        batched version. Expanding the term for M2 (here denoted S) 
        in the original method, we find that 
       
            S_n = S_{n-1} + Sum(
                [ updates - mu_{n-1} * len(updates) ] * 
                [ updates - mu_n * len(updates) ]
            )
        
        Let 
            updates = [x_1, x_2, ..., x_m], 
            mu_{n-1}len(updates) = a_1, 
            mu_n len(updates) = a_2. 

        Then 
            S_n    = S_{n-1} + (x_1 a_1 x_1 a_2) + ... + (x_m a_1 x_m a_2)
                    = (x_1^2 + ... x_m^2) a_1 a_2
        
        We can quickly find the sum of squares for a given list of updates using 
        scatter_add (feats**2). 
        '''

        a_1 = prev_mean * n_updates 
        a_2 = cur_mean * n_updates
        sum_of_squares = scatter_add(torch.pow(feats, 2), idxs, dim=0)
        self.S[idxs] += sum_of_squares * a_1 * a_2 


class StreamTGBase():
    def __init__(self, n_feats, n_nodes=128):
        # Edge feature aggregators
        self.in_aggr = StreamTGBaseEncoder(n_feats, n_nodes)
        self.out_aggr = StreamTGBaseEncoder(n_feats, n_nodes)
        self.bi_aggr = StreamTGBaseEncoder(n_feats, n_nodes)

        # Time feature aggregators 
        # (Could technically append to feat aggrs, but seems like
        # that'd take more time)
        self.in_ts = StreamTGBaseEncoder(1, n_nodes)
        self.out_ts = StreamTGBaseEncoder(1, n_nodes)
        self.bi_ts = StreamTGBaseEncoder(1, n_nodes)

        # Just counts in/out/bi edges from this node
        self.local = torch.zeros(n_nodes, 3)
        self.last_seen = torch.zeros(n_nodes, 3)
        self.IN = 0
        self.OUT = 1
        self.BI = 2

        # Neighbors as of this time
        self.neighbors = defaultdict(set)
        self.n_feats = n_feats

    def __add_if_unseen(self, idx):
        if idx < self.local.size(0):
            return 

        # More or less assumes ids are sequential. So it's unlikely
        # that it will get as edge 1 something like 0->1 then edge 
        # two will be 1,000,000 -> 9,999,999 e.g. (will still work, will just
        # waste a bunch of memory)
        dif = (idx+1) - self.local.size(0)
        self.local = torch.cat([
            self.local,
            torch.zeros(dif, self.local.size(1))
        ], dim=0)

        self.local = torch.cat([
            self.last_seen,
            torch.zeros(dif)
        ])

    def add_batch(self, src_dst, ts, edge_feat, return_value=False):
        src,dst = src_dst 
        self.__add_if_unseen(max(src,dst))

        # Update ts aggrs
        in_ts = ts - self.last_seen[src, self.IN]
        out_ts = ts - self.last_seen[dst, self.OUT]
        bi_ts = ts - self.last_seen[src_dst, self.BI]
        
        self.in_ts.add(dst, in_ts.unsqueeze(-1))
        self.out_ts.add(src, out_ts.unsqueeze(-1))
        self.bi_ts.add(src_dst, bi_ts.unsqueeze(-1))

        self.last_seen[src, self.IN] = ts 
        self.last_seen[dst, self.OUT] = ts 
        self.last_seen[src_dst, self.BI] = ts

        # Update degree counts
        self.local[dst, self.IN] += 1
        self.local[src, self.OUT] += 1
        self.local[src_dst, self.BI] += 1 

        # Add edge feature info 
        self.out_aggr.add(src, edge_feat)
        self.in_aggr.add(dst, edge_feat)
        self.bi_aggr.add(src_dst, edge_feat)

        # Add edge to list
        self.neighbors[src.item()].add(dst)
        self.neighbors[dst.item()].add(src)

        if return_value: 
            return self.get(src)

    def calc_structural(self, idxs=None):
        # TODO This can be seriously revamped since we're
        # giving add_batch an edge index. I think we can incorporate this
        # into the add_batch function 

        if idxs is None:
            idxs = list(range(self.local.size(0)))

        if isinstance(idxs, torch.Tensor):
            if idxs.dim()==0:
                idxs = [idxs.item()]
            else:
                idxs = [int(i) for i in idxs]
        elif not isinstance(idxs, Iterable):
            idxs = [idxs]

        src,dst = [],[]
        for i,k in enumerate(idxs):
            v = self.neighbors[k]

            # Ensure indexed from 0-len(idx) 
            # so output mat[i] correlates to node idx[i]
            src += [i]*len(v)
            dst += list(v) 
        
        ei = torch.tensor([src,dst])

        num_nodes = len(idxs)
        kwargs = dict(dim=0, dim_size=num_nodes)

        dst = ei[0].unsqueeze(-1)
        src = self.local[ei[1]]
        args = (src,dst)

        structure_cols = [
            scatter_add(*args,**kwargs),
            scatter_mean(*args, **kwargs),
            scatter_max(*args, **kwargs)[0],
            scatter_std(*args, **kwargs)
        ]

        if len(idxs) > 1:
            return torch.cat(structure_cols, dim=1)
        return torch.cat(structure_cols, dim=1).squeeze()

    def get(self, idx=None):
        if idx is None:
            idx = torch.arange(self.local.size(0))

        ret = torch.cat([
            self.local[idx],
            self.calc_structural(idx), 
            self.in_aggr.get_value(idx),
            self.in_ts.get_value(idx),
            self.out_aggr.get_value(idx),
            self.out_ts.get_value(idx),
            self.bi_aggr.get_value(idx),
            self.bi_ts.get_value(idx)
        ], dim=-1) 

        return ret 

class BatchTGBase(StreamTGBase):
    def __init__(self, n_feats, n_nodes=128):
        super().__init__(n_feats, n_nodes)

        # Edge feature aggregators
        self.in_aggr = BatchTGBaseEncoder(n_feats, n_nodes)
        self.out_aggr = BatchTGBaseEncoder(n_feats, n_nodes)
        self.bi_aggr = BatchTGBaseEncoder(n_feats, n_nodes)

        # Time feature aggregators 
        # (Could technically append to feat aggrs, but seems like
        # that'd take more time)
        self.in_ts = BatchTGBaseEncoder(1, n_nodes)
        self.out_ts = BatchTGBaseEncoder(1, n_nodes)
        self.bi_ts = BatchTGBaseEncoder(1, n_nodes)