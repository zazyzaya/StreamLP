from collections import defaultdict
import os 

import torch 
from torch_geometric.data import Data

class CSRData():
    def __init__(self, ei, et, ew):
        self.ptr, self.index, self.time, self.weight = self.to_csr(ei,et,ew)
        
    def to_csr(self, ei, et, ew):
        get_empty = lambda : {'ei':[], 'ts': [], 'ew':[]}
        neighbors = defaultdict(get_empty)

        for i in range(ei.size(1)):
            src,dst = ei[:,i]
            t = et[i]; w = ew[i]

            d = neighbors[dst.item()]
            d['ei'].append(src)
            d['ts'].append(t)
            d['ew'].append(w)

        n_nodes = ei.max()+1
        ptr, eis, ets, ews = [0],[],[],[]
        for i in range(n_nodes):
            d = neighbors[i]

            ptr.append(len(d['ei'])+ptr[-1])
            eis += d['ei']
            ets += d['ts']
            ews += d['ew']

        ptr = torch.tensor(ptr)
        eis = torch.tensor(eis)
        ets = torch.tensor(ets)
        ews = torch.tensor(ews)

        return ptr, eis, ets, ews
    
    def get(self, i):
        st, en = self.ptr[i], self.ptr[i+1]
        return self.index[st:en], self.time[st:en], self.weight[st:en]
    
    def __getitem__(self, i):
        return self.get(i)


def find_state_changes(edge_index):
    '''
    Find any time a node switches from being a dst node to a src node
    and thus propagates different messages than it did last time it was 
    a src node, and needs updating.
    '''
    is_src=torch.zeros(edge_index.max()+1, dtype=torch.bool)
    state_change_ts = [0]

    for i, (src,dst) in enumerate(edge_index.T):
        is_src[src] = True 
        if is_src[dst]:
            state_change_ts.append(i)
            is_src[dst] = False 

    state_change_ts.append(edge_index.size(1))
    return state_change_ts

def get_compressed_edge_ts(edge_index):
    '''
    Number edges by which state change this is rather than actual timestamp
    or edge count. 
    '''
    state_change_ts = find_state_changes(edge_index)
    ts = torch.zeros(edge_index.size(1), dtype=torch.long)

    for i in range(len(state_change_ts)-1):
        st = state_change_ts[i]
        en = state_change_ts[i+1]
        ts[st:en] = i 

    return ts

def get_node_ts(edge_index, compressed_ts):
    '''
    Get the timestamp of the last edge all nodes were dst nodes 

        compressed_ts == get_compressed_edge_ts(edge_index)
    '''

    n_nodes = edge_index.max()+1
    ts = torch.zeros(n_nodes, dtype=torch.long)

    # Only care abt nodes that are dst at some point
    # otherwise their embedding will always just be a fn
    # of their original features
    dsts = edge_index[1]
    unique = set([d.item() for d in dsts.unique()])
    
    # Find the last timestamp where this node appeared
    i = -1
    while unique:
        if (node := dsts[i].item()) in unique:
            ts[node] = compressed_ts[i]
            unique.remove(node)
        
        i -= 1

    return ts 


def to_weighted_ei(edge_index):
    '''
    Save space. Change consecutive edges from v->u into single, weighted ones
    '''
    # Self-loops get added later. Remove for now
    ei = edge_index[:, edge_index[0]!=edge_index[1]]

    # Some datasets are not zero-indexed 
    if ei.min() == 1:
        ei -= 1

    # Just learned about this fn. So cool!
    ei,ew = ei.unique_consecutive(dim=1, return_counts=True)
    return ei, ew

def build_data_obj(ei, x=None):
    ei, ew = to_weighted_ei(ei) 
    ts = get_compressed_edge_ts(ei)
    node_ts = get_node_ts(ei, ts)

    if x is None: 
        x = torch.eye(ei.max()+1)

    csr =  CSRData(ei, ts, ew)
    data = Data(
        x,
        csr_ei=csr,
        node_ts=node_ts
    )    
    return data  


CTDNE_FNAMES = {
    'forum': ('fb-forum.edges',','),
    'contacts': ('ia-contacts_hypertext2009.edges',','),
    'enron': ('ia-enron-employees.edges',' '),
    'radoslaw': ('ia-radoslaw-email.edges',' '),
    'btc': ('soc-sign-bitcoinotc.edges', ','),
    'wiki': ('soc-wiki-elec.edges', ' ')
}

def load_ctdne(name, force=False):
    home = 'ctdne-datasets/'
    out_f = home + f'precalculated/{name}.pt'

    if os.path.exists(out_f) and not force:
        return torch.load(out_f)

    fname, sep = CTDNE_FNAMES[name]
    fname = home+fname
    f = open(fname, 'r')
    line = f.readline()

    tokens = line.split(sep)
    has_e_feats = len(tokens) == 4

    src,dst,ts = [],[],[]
    while(line):
        tokens = line.split(sep) 
        (s,d),t = tokens[0:2], tokens[-1]
        
        # For now just skip negative edges? Only applies to wiki really 
        if has_e_feats and tokens[2] == '-1':
            line = f.readline() 
            continue 

        src.append(int(s))
        dst.append(int(d))
        ts.append(int(t))

        line = f.readline() 
    
    ei = torch.tensor([src,dst])
    ts = torch.tensor(ts)

    # Pretty sure order is guaranteed, but to be safe
    order = ts.sort().indices
    ei = ei[:, order]

    g = build_data_obj(ei)
    torch.save(g, out_f)
    return g 

if __name__ == '__main__':
    load_ctdne('enron')