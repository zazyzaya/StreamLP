import torch 
from torch import nn 
from torch_scatter import (
    scatter_add, scatter_mean, 
    scatter_std, scatter_max, scatter_min
)

class FeatureSampler(nn.Module):
    def _tdelta(self, ts): 
        deltas = []
        for t in ts: 
            if not t.size(0):
                deltas.append(torch.tensor([]))
                continue

            deltas.append(
                torch.cat([
                    torch.tensor([0.]),
                    t[1:] - t[:-1]
                ])
            )

        return deltas 

    def aggregate(self, feats): 
        idx = sum([[i]*len(feats[i]) for i in range(feats)], [])
        idx = torch.tensor(idx) 

        feats = torch.cat(feats, dim=0)
        args = (feats, idx)
        kwargs = dict(dim=0, dim_size=len(feats))
        return torch.cat([
            scatter_mean(*args, **kwargs), # type: ignore
            scatter_std(*args, **kwargs),
            scatter_max(*args, **kwargs),
            scatter_min(*args, **kwargs)
        ], dim=1)

    def forward(self, attr_accessor, idx, t): 
        '''
        Assumes input graph is AttributeAccesser object defined in /csr.py and we
        want embeddings for idxs in idx from times prior to t
        '''
        src_t,src_attr = attr_accessor.get_src(idx, t)
        dst_t,dst_attr = attr_accessor.get_src(idx, t)
        bi_t,bi_attr = attr_accessor.get_src(idx, t)

        src_dt = self._tdelta(src_t)
        dst_dt = self._tdelta(dst_t)
        bi_dt = self._tdelta(bi_t)
        