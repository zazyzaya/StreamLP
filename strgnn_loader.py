import os 

import torch
from torch_geometric.data import Data 
from tqdm import tqdm 


HOME = 'StrGNN_Data/'

def load_uci(force=False):
    if os.path.exists(HOME + 'uci.pt') and not force:
        return torch.load(HOME+'uci.pt')

    f = open(HOME+'uci.txt', 'r')

    f.readline()
    meta = f.readline()[2:-1].split(' ')

    n_edges = int(meta[0])
    n_nodes = int(meta[1]) 

    src = []
    dst = []
    weights = []
    ts = []

    line = f.readline()
    prog = tqdm(total=n_edges)
    while line: 
        s,d,w,t = line.split(' ')
        
        src.append(int(s))
        dst.append(int(d))
        weights.append(int(w))
        ts.append(int(t))

        line = f.readline() 
        prog.update() 

    f.close() 
    prog.close() 

    ei = torch.tensor([src,dst])-1 # Idx starts at 1 for some reason 
    ew = torch.tensor(weights, dtype=torch.float)
    ts = torch.tensor(ts, dtype=torch.float)

    # Norm to start at 0 
    ts = ts-ts[0]

    data = Data(
        edge_index=ei, 
        edge_attr=ew, 
        ts=ts, 
        num_nodes=n_nodes-1
    )

    torch.save(data, HOME+'uci.pt')
    return data 