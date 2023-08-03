import os 

import torch
from torch_geometric.data import Data 
from tqdm import tqdm 


HOME = 'StrGNN_Data/'

def load_network_repo(fname='uci', force=False):
    if os.path.exists(f'{HOME}/{fname}.pt') and not force:
        return torch.load(f'{HOME}/{fname}.pt')

    f = open(f'{HOME}/{fname}.txt', 'r')

    f.readline()
    line = f.readline()

    if line.startswith('%'):
        meta = line[2:-1].split(' ')
        n_edges = int(meta[0])
        line = f.readline()
    else:
        n_edges = None 
        n_nodes = None 

    src = []
    dst = []
    weights = []
    ts = []

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
    ew = torch.tensor(weights, dtype=torch.float).unsqueeze(-1)
    ts = torch.tensor(ts, dtype=torch.float)

    # Not guaranteed to be in order
    ts, reidx = ts.sort()
    ei = ei[:, reidx]
    ew = ew[reidx]

    # Norm to start at 0 
    ts = ts-ts[0]

    # Norm ts to be in terms of days
    ts = ts / (60*60*24)

    n_nodes = ei.max()+1
    data = Data(
        edge_index=ei, 
        edge_attr=ew, 
        ts=ts, 
        num_nodes=n_nodes
    )

    torch.save(data, f'{HOME}/{fname}.pt')
    return data 