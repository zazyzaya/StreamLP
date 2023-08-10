from collections import defaultdict
from collections.abc import Iterable

import torch 
from tqdm import tqdm 

class EdgeAttrAccesser:
    def __init__(self, g): 
        self.ts = g.ts 
        self.edge_attr = g.edge_attr 

        (
            self.src_idx, self.src_ptr,  
            self.dst_idx, self.dst_ptr,
            self.both_idx, self.both_ptr
        ) = self.ei_to_csr(g.edge_index)

    def ei_to_csr(self, ei):
        src = defaultdict(list)
        dst = defaultdict(list)
        both = defaultdict(list)

        # Sort into sets of src -> dst 
        for i in tqdm(range(ei.size(1)), desc='Organizing'):
            s,d = ei[:,i]
            src[s.item()].append(i) 
            dst[d.item()].append(i)
            both[s.item()].append(i)
            both[d.item()].append(i)

        src_ptr = [0]
        dst_ptr = [0]
        both_ptr = [0]

        single_src = []
        single_dst = []
        single_both = []
        for idx in tqdm(range(ei.max()+1), desc='Torchifying'):
            s = src.pop(idx, [])
            src_end = src_ptr[-1] + len(s)
            single_src += s 

            d = dst.pop(idx, [])
            dst_end = dst_ptr[-1] + len(d)
            single_dst += d 

            b = both.pop(idx, [])
            both_end = both_ptr[-1] + len(b)
            single_both += b 

            src_ptr.append(src_end)
            dst_ptr.append(dst_end)
            both_ptr.append(both_end)

        # Compress into single list 
        src = torch.tensor(single_src)
        dst = torch.tensor(single_dst) 
        both = torch.tensor(single_both)

        return \
            src, torch.tensor(src_ptr), \
            dst, torch.tensor(dst_ptr), \
            both, torch.tensor(both_ptr) \
        
    def get(self, nids, max_time, ptr, idx): 
        rets = []
        
        for i in range(len(nids)): 
            st = ptr[nids[i]]
            en = ptr[nids[i]+1]
            vals = idx[st:en]
            
            ts = self.ts[vals]
            mask = (ts < max_time[i])

            ts = ts[mask]
            ew = self.edge_attr[vals][mask]

            rets.append((ts,ew))

        return zip(*rets)

    def get_src(self, idx, max_time): 
        return self.get(idx, max_time, self.src_ptr, self.src_idx)
    def get_dst(self, idx, max_time): 
        return self.get(idx, max_time, self.dst_ptr, self.dst_idx)
    def get_bi(self, idx, max_time): 
        return self.get(idx, max_time, self.both_ptr, self.both_idx)
    
if __name__ == '__main__':
    g = torch.load('mixer-datasets/precalculated/wikipedia.pt')
    eaa = EdgeAttrAccesser(g)