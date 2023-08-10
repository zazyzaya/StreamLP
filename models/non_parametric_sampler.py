import torch 
from torch import nn 
from torch_scatter import (
    scatter_mean, scatter_std, 
    scatter_max, scatter_min
)

class FeatureSampler(nn.Module):
    def _tdelta(self, ts): 
        deltas = []
        for t in ts: 
            if not t.size(0):
                deltas.append(torch.empty((0,1)))
                continue

            deltas.append(
                torch.cat([
                    torch.tensor([0.]),
                    t[1:] - t[:-1]
                ]).unsqueeze(-1)
            )

        return deltas 

    def aggregate(self, feats): 
        idx = sum([[i]*len(feats[i]) for i in range(len(feats))], [])
        idx = torch.tensor(idx) 

        dim_size = len(feats)
        feats = torch.cat(feats, dim=0)

        args = (feats, idx)
        kwargs = dict(dim=0, dim_size=dim_size)
        
        return torch.cat([
            scatter_mean(*args, **kwargs), # type: ignore
            scatter_std(*args, **kwargs),
            scatter_max(*args, **kwargs)[0],
            scatter_min(*args, **kwargs)[0]
        ], dim=1)

    def __full_agg_one(self, ts, feats):
        ts = self._tdelta(ts)
        complete = [
            torch.cat([feats[i], ts[i]], dim=1) 
            for i in range(len(feats))
        ] 
        return self.aggregate(complete)


    def forward(self, attr_accessor, idx, t): 
        '''
        Using AttributeAccesser object defined in /csr.py and we
        want embeddings for idxs in idx from times prior to t
        '''
        return torch.cat([
            self.__full_agg_one(*attr_accessor.get_src(idx, t)),
            self.__full_agg_one(*attr_accessor.get_dst(idx, t)),
            self.__full_agg_one(*attr_accessor.get_bi(idx, t))
        ], dim=1)


if __name__ == '__main__':
    from torch_geometric.data import Data 
    import sys 
    sys.path.append('..')
    from csr import EdgeAttrAccesser 

    g = Data(
        edge_index = torch.tensor([
            [0,0,1,1,2,2],
            [1,2,2,3,1,3]
        ]), 
        ts = torch.tensor([
            0,1,2,3,4,5
        ]),
        edge_attr = torch.tensor([
            [0,0,1],
            [0,1,1],
            [1,0,0],
            [1,1,0],
            [1,1,1],
            [0,1,0]
        ])
    )
    eaa = EdgeAttrAccesser(g) 

    fs = FeatureSampler() 
    print(fs.forward(eaa, torch.tensor([0,1,2,3]), torch.tensor([2,3,4,5])))