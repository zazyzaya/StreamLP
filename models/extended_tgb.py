import torch 

from .optimized_tgb import BatchTGBase, StreamTGBase

class PropStreamTGBase(StreamTGBase): 
    def get(self, idx=None):
        if idx is None:
            idx = torch.arange(self.local.size(0))

        # Remove neighborhood embedding
        to_tensor = [
            self.local[idx], 
            self.in_aggr.get_value(idx),
            self.in_ts.get_value(idx),
            self.out_aggr.get_value(idx),
            self.out_ts.get_value(idx),
            self.bi_aggr.get_value(idx),
            self.bi_ts.get_value(idx)
        ]

        ret = torch.cat(to_tensor, dim=-1) 
        return ret 
    
    def add_edge(self, src_dst, ts, edge_feat, return_value=False):
        # Update node features
        super().add_edge(src_dst, ts, edge_feat, return_value=False)
