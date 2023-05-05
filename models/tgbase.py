from collections import defaultdict
from collections.abc import Iterable

import torch 
from torch import nn
from torch_scatter import (
    scatter_add, scatter_mean, 
    scatter_std, scatter_max, 
    scatter_min
)

from .incrimental_h import StreamEntropy

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
    

class StreamTGBaseEncoder():
    def __init__(self, num_feats, num_nodes, entropy=True):
        '''
        TODO include entropy (is this even possible?)
        '''
        self.num_feats = num_feats

        args = (num_nodes, num_feats)
        self.S = torch.zeros(*args) # Keeps track of var_i * tot_i
        self.tot = torch.zeros(num_nodes,1)
        
        self.sum = torch.zeros(*args)
        self.min = torch.zeros(*args) #, float('inf'))
        self.max = torch.zeros(*args) #, float('-inf'))

        self.calc_entropy = entropy
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

        # Need to calculate std and mean from observed running totals
        mean = (self.sum[idxs] / self.tot[idxs]).nan_to_num(0)
        std = torch.sqrt(self.S[idxs] / self.tot[idxs]).nan_to_num(0)

        to_tensor = [
            self.sum[idxs],
            mean, 
            self.max[idxs],
            self.min[idxs],
            std 
        ]

        if self.calc_entropy:
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
        return torch.cat(to_tensor, dim=-1)

    def __add_unseen(self, idx):
        # More or less assumes ids are sequential. So it's unlikely
        # that it will get as edge 1 something like 0->1 then edge 
        # two will be 1,000,000 -> 9,999,999 e.g. (will still work, will just
        # waste a bunch of memory)
        dif = (idx+1) - self.tot.size(0)

        for key in ['S', 'tot', 'sum', 'max', 'min']:
            # Can't pass by reference w/o doing this?
            self.__dict__[key] = torch.cat([
                self.__dict__[key], 
                torch.zeros(dif, self.__dict__[key].size(1))
            ], dim=0)

        '''
        # Inf values are slipping into the data. Not sure how the OG
        # authors delt with this, it appears to still be in their code
        # but I'm just initializing to 0 
        self.max = torch.cat([
            self.max, 
            torch.full((dif, self.max.size(1)), float('-inf'))
        ])
        self.min = torch.cat([
            self.min, 
            torch.full((dif, self.min.size(1)), float('inf'))
        ])
        '''

        if self.calc_entropy:
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

        if self.calc_entropy:
            [self.ent[i].add(feat) for i in idx]


    def increment(self, idx, value, feat_id=-1):
        '''
        Given list [a,b,c] that we are keeping track of aggregations for
        the result of incriment(a) will make the aggregations as if 
        the list had been updated to [a+1, b,c] 
        
        NOTE: 
            This is as opposed to add(a) which would change the list to [a,b,c,a]
            Only used for neighborhood info if a node gets a new edge
            Feature must represent discrete counting variable, not continuous one
        '''
        # If this is an unseen id 
        if idx >= self.tot.size(0):
            return self.__add_unseen(idx)

        # Total does not change, still have n samples
        #   self.tot[idx] += 1
        # Min doesn't change either. Min(x, x+1) always x
        #   self.min[idx] = torch.min(self.min[idx], feat) 

        self.sum[idx, feat_id] += 1 
        self.max[idx, feat_id] = torch.max(self.max[idx, feat_id], value)
        
        # This follows from 
        # S = var*n = sum(x_i^2) - n*mu^2
        # S' - S =  [ sum(x^2) - (x'-1)^2 + (x')^2 - n*(mu'^2) ]
        #           - [ sum(x^2) - n*mu^2]
        #
        #           / 1-2*sum(x)+1  \
        #  = 2x-1+ | --------------  |
        #           \       n       /
        self.S[idx, feat_id] = \
            self.S[idx, feat_id] + \
            (2*value-1) + \
            ((1-2*self.sum[idx, feat_id])/self.tot[idx])
        
        if self.calc_entropy:
            self.ent[idx].update(value-1, value, feat_id)

    
class StreamTGBase():
    def __init__(self, n_feats, n_nodes=128, entropy=True):
        # Edge feature aggregators
        self.in_aggr = StreamTGBaseEncoder(n_feats, n_nodes, entropy=entropy)
        self.out_aggr = StreamTGBaseEncoder(n_feats, n_nodes, entropy=entropy)
        self.bi_aggr = StreamTGBaseEncoder(n_feats, n_nodes, entropy=entropy)

        # Time feature aggregators 
        # (Could technically append to feat aggrs, but seems like
        # that'd take more time)
        self.in_ts = StreamTGBaseEncoder(1, n_nodes, entropy=entropy)
        self.out_ts = StreamTGBaseEncoder(1, n_nodes, entropy=entropy)
        self.bi_ts = StreamTGBaseEncoder(1, n_nodes, entropy=entropy)

        # Just counts in/out/bi edges from this node
        self.local = torch.zeros(n_nodes, 3)
        self.last_seen = torch.zeros(n_nodes, 3)
        self.IN = 0
        self.OUT = 1
        self.BI = 2

        # Neighbors as of this time
        self.neighbors = defaultdict(set)
        self.entropy = entropy
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

    def add_edge(self, src_dst, ts, edge_feat, return_value=False):
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
            scatter_min(*args, **kwargs)[0],
            scatter_std(*args, **kwargs)
        ]

        if self.entropy:
            # There's prob a more efficient way to do this but..
            lists = [[] for _ in range(num_nodes)]
            for j in range(ei.size(1)):
                lists[ei[0][j]].append(
                    self.local[ei[1][j]]
                )
            
            ent=[     
                self.__column_wise_entropy(torch.stack(l))
                for l in lists
            ]
            ent = torch.stack(ent)
            structure_cols.append(ent)

        if len(idxs) > 1:
            return torch.cat(structure_cols, dim=1)
        return torch.cat(structure_cols, dim=1).squeeze()
    
    def __column_wise_entropy(self, mat):
        ent = []
        for i in range(mat.size(1)):
            col = mat[:,i].unique(return_counts=True)[1]
            c = col.sum()
            ent.append(-((col/c) * torch.log2(col/c)).sum())
        
        return torch.tensor(ent)

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

def mask_feat(names, size, directions=['in','out','bi'], has_entropy=True):
    '''
    Returns a mask of only the specified features/directions for each node
    '''
    
    allowed = [
        'local', 'structural', 'times', 'sum', 'mean', 
        'max', 'min', 'std', 'entropy'
    ]
    for name in names:
        assert name in allowed, '%s not in allowed list %s' % (name, str(allowed))

    m = torch.zeros(size, dtype=torch.bool)
    
    neighborhood = 18 if has_entropy else 15
    per_feat = 6 if has_entropy else 5

    # Need to derive edge feats given feature vec looks like:
    # 3 + 3*6 || 3*5 + [f * (6 || 5) + 6 || 5] * 3
    # where || denotes the left side if ent the right otherwise
    edge_feats = size-3    # local
    edge_feats -= neighborhood
    edge_feats /= 3             # in/out/both
    edge_feats -= per_feat      # timestamps
    edge_feats /= per_feat      # features

    # Non-variable length part (other than entropy)
    node_feats = 3
    edge_start = 3 + 3*per_feat
    edge_size = edge_feats*per_feat

    fn_map = {
        'sum':0, 'mean': 1, 'max':2,
        'min':3, 'std': 4, 'entropy':5
    }

    if 'local' in names:
        m[:node_feats] = True 
        names.remove('local')

    structural=False
    if 'structural' in names:
        structural=True 
        names.remove('structural')

    ts=False 
    if 'times' in names:
        ts = True 
        names.remove('times')

    # User can just input times/structural and get only
    # those features for every function
    no_edges = False
    if (ts or structural) and names == []:
        names = [k for k in fn_map.keys()]
        if not has_entropy:
            names.remove('entropy')
        no_edges = True

    dir_map = {'in':0, 'out':1, 'bi':2}
    directions = [dir_map[d] for d in directions]

    for name in names:
        for i in directions:
            if not no_edges:
                    # Local feats       # Which fn             #ts        #direction
                st = edge_start + fn_map[name]*edge_feats + per_feat*i + edge_size*i 
                en = st + edge_feats
                m[int(st):int(en)] = True 

            if ts: 
                # Kind of convoluted... skip edge start, skip edge features 
                # and any time features we've already read, then we're pointed at 
                # the right place
                loc = int(edge_start + edge_size*(i+1) + per_feat*i + fn_map[name])
                m[loc] = True 

            if structural:
                loc = int(node_feats + 3*fn_map[name] + i) 
                m[loc] = True 

    return m




if __name__ == '__main__':
    mask(['structural'], 78, directions=['bi'])