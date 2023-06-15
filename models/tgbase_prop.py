from collections.abc import Iterable
import torch

from .tgbase import StreamTGBaseEncoder
from .incrimental_h import StreamEntropy

class CustomTGBaseEncoder():
    def __init__(self, num_feats, num_nodes, sum=True, mean=True, min=True, max=True, std=True, entropy=True):
        self.num_feats = num_feats

        self.has_sum=sum; self.has_mean=mean; self.has_std=std
        self.has_min=min; self.has_max=max; self.has_entropy=entropy
        
        self.aggr_keys = []

        args = (num_nodes, num_feats)
        if std:
            self.S = torch.zeros(*args) # Keeps track of var_i * tot_i
            self.aggr_keys.append('S')

        if mean or std:
            self.tot = torch.zeros(num_nodes,1)
            self.aggr_keys.append('tot')
        
        if mean or sum or std:
            self.sum = torch.zeros(*args)
            self.aggr_keys.append('sum')

        if min:
            self.min = torch.full(args, float('inf'))
        
        if max: 
            self.max = torch.full(args, float('-inf'))

        if entropy:
            self.ent = [
                StreamEntropy(num_feats)
                for _ in range(num_nodes)
            ]

    def get_value(self, idxs=None):
        if idxs is None:
            idxs = torch.arange(self.S.size(0))

        if not isinstance(idxs, Iterable) or (isinstance(idxs, torch.Tensor) and idxs.dim()==0):
            idxs = [idxs]

        to_tensor = []

        if self.has_mean:
            to_tensor.append((self.sum[idxs] / self.tot[idxs]).nan_to_num(0))
        if self.has_std:
            to_tensor.append(torch.sqrt(self.S[idxs] / self.tot[idxs]).nan_to_num(0))
        if self.has_sum:
            to_tensor.append(self.sum[idxs])        
        if self.has_max: 
            to_tensor.append(self.max[idxs])
        if self.has_min:
            to_tensor.append(self.min[idxs])
        
        if self.has_entropy:
            ent = []
            for i in idxs:
                ent.append(self.ent[i].get())

            # Not sure how to maket this cleaner but need
            # the resulting tensor to be dim()==1 if len(idxs)==1
            if len(idxs) != 1:
                to_tensor.append(torch.stack(ent))
            else:
                to_tensor.append(ent[0])

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

        for key in self.aggr_keys:
            # Can't pass by reference w/o doing this?
            self.__dict__[key] = torch.cat([
                self.__dict__[key], 
                torch.zeros(dif, self.__dict__[key].size(1))
            ], dim=0)

        # Inf values are slipping into the data. Not sure how the OG
        # authors delt with this, it appears to still be in their code
        # but I'm just initializing to 0 
        # UPDATE: changed get_value to have .nan_to_num to overwrite inf values to 0
        if self.has_max:
            self.max = torch.cat([
                self.max, 
                torch.full((dif, self.max.size(1)), float('-inf'))
            ])
        
        if self.has_min:
            self.min = torch.cat([
                self.min, 
                torch.full((dif, self.min.size(1)), float('inf'))
            ])

        if self.has_entropy:
            self.ent += [
                StreamEntropy(self.num_feats)
                for _ in range(dif)
            ]


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
        prev_mean = 0
        if self.has_std:
            prev_mean = (self.sum[idx] / self.tot[idx]).nan_to_num(0)

        if self.has_std or self.has_mean or self.has_sum:
            self.tot[idx] += 1
            self.sum[idx] += feat 

        if self.has_max:
            self.max[idx] = torch.max(self.max[idx], feat)
        if self.has_min:
            self.min[idx] = torch.min(self.min[idx], feat)

        # Moving standard dev calculated by 
        # S_n = S_{n-1} + (x_n - mu_{n-1})(x_n - mu_n)
        # std_n = sqrt(S_n / n)
        # I don't think torch knows how to FOIL so expanding that,
        # the last term becomes: 
        # (x_n-mu_{n-1})(x_n - mu_n) = x_n^2 - x_n*mu_n - x_n*mu_{n-1} + mu_n*mu_{n-1}
        # which in variables we'll call a       b           c               d 
        if self.has_std:
            cur_mean = self.sum[idx] / self.tot[idx]

            a = torch.pow(feat,2)
            b = feat * cur_mean 
            c = feat * prev_mean 
            d = cur_mean * prev_mean
            self.S[idx] += a-b-c+d 

        if self.has_entropy:
            [self.ent[i].add(feat) for i in idx]


class SimpleMeanAggr():
    def __init__(self, num_feats, num_nodes, decay=0.):
        self.sum = torch.zeros(num_nodes, num_feats)
        self.c = torch.zeros(num_nodes,1)
        self.decay_rate = 1-decay 
    
    def __add_unseen(self, idx):
        # More or less assumes ids are sequential. So it's unlikely
        # that it will get as edge 1 something like 0->1 then edge 
        # two will be 1,000,000 -> 9,999,999 e.g. (will still work, will just
        # waste a bunch of memory)
        dif = (idx+1) - self.sum.size(0)
        self.sum = torch.cat([
            self.sum, 
            torch.zeros(dif, self.sum.size(1))
        ], dim=0)

    def add(self, idx, feat):
        if (isinstance(idx, torch.Tensor) and idx.dim() > 0) \
            or isinstance(idx, list):
            check = max(idx) # type: ignore
        else:
            check = idx 
            idx = [idx]

        # If this is an unseen id 
        if check >= self.sum.size(0):
            self.__add_unseen(check)

        # Slowly removes effect of older values
        if self.decay_rate != 1:
            self.sum[idx] *= self.decay_rate 
            self.c[idx] *= self.decay_rate

        self.sum[idx] += feat 
        self.c[idx] += 1

    def get_value(self, idxs=None):
        if idxs is None:
            idxs = torch.arange(self.sum.size(0))

        if not isinstance(idxs, Iterable) or (isinstance(idxs, torch.Tensor) and idxs.dim()==0):
            idxs = [idxs]

        return (self.sum[idxs] / self.c[idxs]).nan_to_num(0)

    def __getitem__(self, idxs):
        return self.get_value(idxs)


class ConvolutionalTGBase():
    def __init__(self, n_feats, hops=3, n_nodes=128, entropy=False, decay=1.):
        # Edge feature aggregators -- Correlate to node features
        self.in_aggr = StreamTGBaseEncoder(n_feats, n_nodes, entropy=entropy)
        self.out_aggr = StreamTGBaseEncoder(n_feats, n_nodes, entropy=entropy)
        self.bi_aggr = StreamTGBaseEncoder(n_feats, n_nodes, entropy=entropy)
    
        # Time feature aggregators 
        # (Could technically append to feat aggrs, but seems like
        # that'd take more time)
        self.in_ts = StreamTGBaseEncoder(1, n_nodes, entropy=entropy)
        self.out_ts = StreamTGBaseEncoder(1, n_nodes, entropy=entropy)
        self.bi_ts = StreamTGBaseEncoder(1, n_nodes, entropy=entropy)

        # Propagators
        self.one_hop_feats = self.__get_x(0).size(1)
        self.in_prop = SimpleMeanAggr(self.one_hop_feats * (hops-1), n_nodes, decay=decay)
        self.out_prop = SimpleMeanAggr(self.one_hop_feats * (hops-1), n_nodes, decay=decay)
        self.bi_prop = SimpleMeanAggr(self.one_hop_feats * (hops-1), n_nodes, decay=decay)

        # Include timestamp info
        self.last_seen = torch.zeros(n_nodes, 3)
        self.IN = 0
        self.OUT = 1
        self.BI = 2

    def __get_x(self, idx):
        return torch.cat([
            self.in_ts[idx], self.out_ts[idx], self.bi_ts[idx],
            self.in_aggr[idx], self.out_aggr[idx], self.bi_aggr[idx]
        ], dim=-1)

    def get_value(self, idx):
        return torch.cat([
            self.__get_x(idx),
            self.in_prop[idx],
            self.out_prop[idx],
            self.bi_prop[idx]
        ])

    def __getitem__(self, idx):
        return self.get_value(idx)

    def add_edge(self, src_dst, ts, edge_feat, return_value=False):
        src,dst = src_dst 

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

        # Add edge feature info 
        self.out_aggr.add(src, edge_feat)
        self.in_aggr.add(dst, edge_feat)
        self.bi_aggr.add(src_dst, edge_feat)

        # Propagate message to neighbors
        src_x_dst_x = self.__get_x(src_dst)
        src_x,dst_x = src_x_dst_x
        dst_src = src_dst[[1,0]]

        self.in_prop.add(dst, torch.cat([src_x, self.in_prop[src][:-self.one_hop_feats]], dim=-1))
        self.out_prop.add(src, torch.cat([dst_x, self.out_prop[dst][:-self.one_hop_feats]], dim=-1))
        self.bi_prop.add(dst_src, torch.cat([src_x_dst_x, self.bi_prop[src_dst][:,:-self.one_hop_feats]], dim=-1))

        if return_value:
            return self.get_value(src)
        
    
if __name__ == '__main__':
    tgb = ConvolutionalTGBase(2)
    ei = torch.tensor([
        [0,1,2],
        [1,2,3]
    ]).T
    x = torch.tensor([
        [0.1,0.1],
        [0.2,0.2],
        [0.3,0.3]
    ])

    ret = [tgb.add_edge(ei[i], i, x[i], return_value=True) for i in range(ei.size(0))]
    print(ret)