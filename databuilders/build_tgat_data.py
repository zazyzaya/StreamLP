from joblib import Parallel, delayed 
import torch 
from torch_geometric.data import Data 
from tqdm import tqdm 

HOME = '/home/ead/iking5/code/StreamLP/mixer-datasets/'

def build_from_csv(name, tr_ratio=0.7):
    fname = HOME+name+'.csv'

    def get_edge_job(f):
        src,dst,ts,ys = [],[],[],[]
        f.readline() # Skip header 
        line = f.readline()

        prog = tqdm(desc='Edges')
        while(line):
            s,d,t,y,_ = line.split(',',4)
            src.append(int(s))
            dst.append(int(d))
            ts.append(float(t))
            ys.append(int(y))

            prog.update()
            line = f.readline()

        src = torch.tensor(src)
        dst = torch.tensor(dst)
        ts = torch.tensor(ts)
        ys = torch.tensor(ys)

        # Need nodes to be unique. Since graph was 
        # bipartite they give srcs and dsts their own
        # set of unique labels, we're just undoing that
        dst += src.max()+1

        # Norm so min is 0
        ts -= ts.min()
        ts, idx = ts.sort() # Make sure ascending temporal order
        
        # Reorder other stuff too 
        ei = torch.stack([src,dst])
        ei = ei[:, idx]
        ys = ys[idx]

        f.close()
        prog.close()
        return ei,ts,ys,idx
    
    def get_feat_job(f):
        feats = []
        f.readline() # Skip header
        line = f.readline()

        prog = tqdm(desc='Features')
        while (line):
            # Trim off tailing \n, and skip 
            # edge index data
            feat = line[:-1].split(",")[4:]
            feat = [float(f) for f in feat]
            feats.append(feat)

            line = f.readline()
            prog.update()

        f.close()
        prog.close()

        return torch.tensor(feats)
    
    def delegate(i):
        f = open(fname, 'r')
        if i == 0:
            return get_edge_job(f)
        else: 
            return get_feat_job(f)
    
    # I think stripping features takes long enough that it's probably worth it to 
    # just delegate it to its own thread (saves like 2 seconds.. eh)
    #                                       This comment shuts up PyLance --v
    (ei,ts,ys,idx),feats = Parallel(2, prefer='processes', batch_size=2)( # type: ignore
        delayed(delegate)(i) for i in range(2)
    )

    feats = feats[idx]

    n_edges = ei.size(1)
    n_nodes = ei.max()

    out_f = HOME+f'precalculated/{name}.pt'
    g = Data(
        edge_index=ei, 
        edge_attr=feats, 
        ts=ts, 
        ys=ys, 
        num_nodes=n_nodes, 
        num_edges=n_edges 
    )

    torch.save(g, out_f)
    return g 

if __name__ == '__main__':
    build_from_csv('reddit')
    build_from_csv('wikipedia')