from collections import defaultdict
from collections.abc import Iterable

import torch 
from torch_scatter import (
    scatter_add, scatter_mean, 
    scatter_std, scatter_max, scatter_min
)

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
class StreamTGBaseEncoder():
    def __init__(self, num_feats, num_nodes):
        self.num_feats = num_feats

        args = (num_nodes, num_feats)
        self.S = torch.zeros(*args) # Keeps track of var_i * tot_i
        self.tot = torch.zeros(num_nodes,1)
        
        self.sum = torch.zeros(*args)
        self.max = torch.full(args, float('-inf'))
        self.min = torch.full(args, float('inf'))
        self.n_nodes = num_nodes

    def get_value(self, idxs=None):
        if idxs is None:
            idxs = torch.arange(self.S.size(0))

        if not isinstance(idxs, Iterable) or (isinstance(idxs, torch.Tensor) and idxs.dim()==0):
            idxs = [idxs]

        # Need to calculate std and mean from observed running totals
        mean = (self.sum[idxs] / self.tot[idxs]).nan_to_num(0)
        std = torch.sqrt(self.S[idxs] / self.tot[idxs]).nan_to_num(0)

        to_tensor = [
            #self.sum[idxs],
            mean, 
            self.max[idxs],
            self.min[idxs],
            std 
        ]

        # Concat all features 
        return torch.cat(to_tensor, dim=-1).nan_to_num(0,0,0)

    def __getitem__(self, idx):
        return self.get_value(idx)

    def _add_unseen(self, idx):
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

        self.min = torch.cat([
            self.min, 
            torch.full((dif, self.min.size(1)), float('inf'))
        ])

        self.n_nodes = self.tot.size(0)

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
            self._add_unseen(check)

        # In case hasn't been seen before
        prev_mean = (self.sum[idx] / self.tot[idx]).nan_to_num(0)

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

class BatchTGBaseEncoder(StreamTGBaseEncoder):
    '''
    Note:   the field S now denotes M2 in Welford's method of calculating running variance
            (which, I'm pretty sure is the same quantity... M2/n = S/n = variance)
    See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    and https://stackoverflow.com/questions/56402955/whats-the-formula-for-welfords-algorithm-for-variance-std-with-batch-updates
    '''
    def add(self, idxs, feats):    
        check = idxs.max() 

        # If this is an unseen id 
        if check >= self.tot.size(0):
            self._add_unseen(check)

        # Ensure we only touch memory we absolutely need to 
        unique,reindex = idxs.unique(return_inverse=True)

        prev_mean = (self.sum[unique] / self.tot[unique]).nan_to_num(0)
        n_updates = scatter_add(torch.ones(idxs.size(0),1), reindex, dim=0)
        self.tot[unique] += n_updates 

        self.sum[unique] += scatter_add(feats, reindex, dim=0)
        self.max[unique] = torch.max(self.max[unique], scatter_max(feats, reindex, dim=0)[0])
        self.min[unique] = torch.min(self.min[unique], scatter_min(feats, reindex, dim=0)[0])

        cur_mean = self.sum[unique] / self.tot[unique]

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
        sum_of_squares = scatter_add(torch.pow(feats, 2), reindex, dim=0)
        self.S[unique] += sum_of_squares * a_1 * a_2 


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
        self.n_nodes = n_nodes 

    def _add_if_unseen(self, idx):
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

        self.n_nodes = self.local.size(0)

    def add_edge(self, src_dst, ts, edge_feat, return_value=False):
        src,dst = src_dst 
        self._add_if_unseen(max(src,dst))

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
            #scatter_add(*args,**kwargs),
            scatter_mean(*args, **kwargs),
            scatter_max(*args, **kwargs)[0],
            scatter_min(*args, **kwargs)[0],
            scatter_std(*args, **kwargs)
        ]

        if len(idxs) > 1:
            return torch.cat(structure_cols, dim=1)
        return torch.cat(structure_cols, dim=1).squeeze()

    def get(self, idx=None):
        if idx is None:
            idx = torch.arange(self.local.size(0))

        to_tensor = [
            self.local[idx],
            self.calc_structural(idx), 
            self.in_aggr.get_value(idx),
            self.in_ts.get_value(idx),
            self.out_aggr.get_value(idx),
            self.out_ts.get_value(idx),
            self.bi_aggr.get_value(idx),
            self.bi_ts.get_value(idx)
        ]

        ret = torch.cat(to_tensor, dim=-1) 

        return ret 

class BatchTGBase(StreamTGBase):
    def __init__(self, n_feats, n_nodes=128, undirected_neighborhood_aggr=True):
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

        self.neighbors = set() 
        self.undirected_neighborhood_aggr = undirected_neighborhood_aggr

    @torch.no_grad()
    def add_batch(self, ei, ts, edge_feat, return_value=False):
        self._add_if_unseen(ei.max())

        # Need to incorporate repeats within the batch, 
        # so have to iterate over whole ei 
        last_seen_in = []
        last_seen_out = []
        last_seen_bi_src = []
        last_seen_bi_dst = []
        for i in range(ei.size(1)):
            t = ts[i]
            src,dst = ei[:, i]
            
            # Get deltas
            last_seen_in.append(t - self.last_seen[dst, self.IN])
            last_seen_out.append(t - self.last_seen[src, self.OUT])
            last_seen_bi_src.append(t - self.last_seen[src, self.BI])
            last_seen_bi_dst.append(t - self.last_seen[dst, self.BI])

            # Update last seen 
            self.last_seen[dst, self.IN] = t 
            self.last_seen[src, self.OUT] = t 
            self.last_seen[src, self.BI] = self.last_seen[dst, self.BI] = t

            # While we're looping, also want to keep track of 
            # all unique edges we've seen for structural features
            self.neighbors.add(
                (src.item(), dst.item())
            )
            if self.undirected_neighborhood_aggr:
                self.neighbors.add(
                    (dst.item(), src.item())
                )


        self.in_ts.add(ei[1], torch.tensor(last_seen_in).unsqueeze(-1))
        self.out_ts.add(ei[0], torch.tensor(last_seen_out).unsqueeze(-1))
        self.bi_ts.add(
            ei.reshape(ei.size(1)*2), 
            torch.tensor(last_seen_bi_src + last_seen_bi_dst).unsqueeze(-1)
        )

        self.local[:, self.IN] += scatter_add(
            torch.ones(ei.size(1)), 
            ei[1], dim=0, dim_size=self.n_nodes
        )
        self.local[:, self.OUT] += scatter_add(
            torch.ones(ei.size(1)),
            ei[0], dim=0, dim_size=self.n_nodes
        )
        self.local[:, self.BI] += scatter_add(
            torch.ones(ei.size(1)*2),
            ei.reshape(ei.size(1)*2), dim_size=self.n_nodes
        )

        # Add edge feature info 
        self.out_aggr.add(ei[0], edge_feat)
        self.in_aggr.add(ei[1], edge_feat)
        self.bi_aggr.add(ei.reshape(ei.size(1)*2), edge_feat.repeat(2,1))

        if return_value:
            return self.get()

    def calc_structural(self, idxs=None):
        ei = torch.tensor([*zip(*self.neighbors)])

        args = (self.local[ei[0]], ei[1])
        kwargs = dict(dim=0, dim_size=self.n_nodes)
        return torch.cat([
            #scatter_add(*args, **kwargs),
            scatter_mean(*args, **kwargs), 
            scatter_max(*args, **kwargs)[0],
            scatter_min(*args, **kwargs)[0],
            scatter_std(*args, **kwargs)
        ], dim=1)
    
if __name__ == '__main__':
    ei = torch.tensor([
        [0,1,1,2,3],
        [1,2,3,3,0]
    ])
    ts = torch.arange(5, dtype=torch.float)
    ex = torch.tensor([
        [0.1],
        [0.2],
        [0.3],
        [0.4],
        [0.5]
    ])

    tg = BatchTGBase(1, 4)
    e1 = tg.add_batch(ei, ts, ex, return_value=True)
    print(e1)

    '''
    Expected: 
    Node 0: 
        i/o/b   
                    [1,1,2]
        Neighbors
            sum     [3,3,6]
            mean    [1.5,1.5,3]
            max     [2,2,3],
            std     [0.5,0.5,0]
        In feats
                    [0.5,0.5,0.5, 0]
        In ts
                    [5,5,5,0]
        Out feats
                    [0.1,0.1,0.1,0]
        Out ts
                    [0,0,0,0]
        Bi feats
                    [0.6,0.3,0.5,0.2]
        Bi ts  
                    [3,6,5,2]
    
    Actual: 
        1.0000, 1.0000, 2.0000
        
        3.0000, 3.0000, 6.0000, sum
        1.5000, 1.5000, 3.0000, mean
        2.0000, 2.0000, 3.0000, max
        0.7071, 0.7071, 0.0000, <- std wrong for small counts
        
        0.5000, 0.5000, 0.5000, 0.0000, 
        4.0000, 4.0000, 4.0000, 0.0000, 
        
        0.1000, 0.1000, 0.1000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 
        
        0.6000, 0.3000, 0.5000, 0.0000, 
        4.0000, 2.0000, 4.0000, 0.0000
    '''

    e2 = tg.add_batch(ei, ts, ex, return_value=True)
    print(e2)
