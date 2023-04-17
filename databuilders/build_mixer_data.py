from collections import defaultdict
import os 
import pickle
import socket 

from joblib import Parallel, delayed
import torch 
from torch_geometric.data import Data 
from torch_geometric.nn import MessagePassing
from tqdm import tqdm 

HOME = os.path.dirname(__file__) + '/../mixer-datasets/'

if socket.gethostname() == 'orion.ece.seas.gwu.edu':
    HOME = '/mnt/raid0_ssd_8tb/isaiah/mixer-datasets/'

class MixerCSR():
    def __init__(self, ei, ts, feats, ys, va,te, name, node_feats=None):
        self.end_tr = self.start_va = va 
        self.end_va = self.start_te = te 

        self.ptr, self.index, \
        self.time, self.efeats, \
        self.dyn_y = self.to_csr(ei,ts,feats,ys)

        self.name = name 
        self.num_nodes = self.ptr.size(0)-1
        self.node_feats = torch.zeros(self.num_nodes, 1)

    def to(self, device):
        self.time = self.time.to(device)
        self.efeats = self.efeats.to(device)
        self.node_feats = self.node_feats.to(device)

    def to_csr(self, ei, ts, feats, ys):
        get_empty = lambda : {
            'ei':[], 'ts': [], 'feat':[], 'y':[]
        }
        neighbors = defaultdict(get_empty)

        for i in tqdm(range(ei.size(1)), desc='scattering'):
            src,dst = ei[:,i]
            t = ts[i]; w = feats[i]; y = ys[i]

            d = neighbors[dst.item()]
            d['ei'].append(src)
            d['ts'].append(t)
            d['feat'].append(w)
            d['y'].append(y)

        n_nodes = ei.max()+1
        ptr, eis, ets, feats, ys, = [0],[],[],[],[]
        for i in tqdm(range(n_nodes), desc='gathering'):
            d = neighbors[i]

            ptr.append(len(d['ei'])+ptr[-1])
            
            ei = torch.tensor(d['ei'])
            et = torch.tensor(d['ts'])
            y = torch.tensor(d['y'])
            feat = torch.stack(d['feat'])

            # Put in order of time
            et, order = et.sort()
            ei = ei[order]
            feat = feat[order]
            y = y[order]
            
            eis.append(ei)
            ets.append(et)
            feats.append(feat)
            ys.append(y)
            
        # Consolidate into single vectors
        ptr = torch.tensor(ptr)
        eis = torch.cat(eis)
        ets = torch.cat(ets)
        feats = torch.cat(feats, dim=0)
        ys = torch.cat(ys)

        return ptr, eis, ets, feats, ys
    
    def __getitem__(self, nid):
        st, en = self.ptr[nid], self.ptr[nid+1]
        return \
            self.time[st:en].unsqueeze(-1), \
            self.efeats[st:en], \
            self.index[st:en], \
            self.dyn_y[st:en]

    def get_edge_feats(self, nid):
        # Don't actually care about the neighbor's indices
        st, en = self.ptr[nid], self.ptr[nid+1]
        return self.time[st:en].unsqueeze(-1), self.efeats[st:en] 

    def sample_one(self, nid, t, k):
        # Edge features
        ts, feats = self.get_edge_feats(nid)
        earlier = (ts < t).squeeze(-1)

        # Returns most recent k edge features that occured 
        # before time t 
        return ts[earlier][-k:], feats[earlier][-k:]
    
    def sample(self, nids, t, k=20, compressed=True):
        # Somehow, using threads/procs only slows this down
        # communication is slow enough that it's worth it to
        # just do this all at once I guess
        samples = [
            self.sample_one(nid, t, k)
            for nid in nids
        ]

        if compressed:
            # Concat samples, and return pointers to where each 
            # datapoint belongs (kind of like sparse-packing an RNN)
            return \
                *self.compress(samples), \
                *self.index_sample(samples)
        
        return samples 
    
    def subsample(self, idx,feats,samples):
        subset = [samples[i] for i in idx]
        
        return feats[idx], \
            *self.compress(subset), \
            *self.index_sample(subset)

    def compress(self, samples):
        ts,feat = zip(*samples)
        ts = torch.cat(ts, dim=0)
        feat = torch.cat(feat, dim=0)

        return ts,feat

    def index_sample(self, sample):
        nid, idx = [],[]
        for i in range(len(sample)):
            samp_size = sample[i][0].size(0)
            
            nid += [i]*samp_size
            idx += list(range(samp_size))

        return torch.tensor(nid), torch.tensor(idx)


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

        # Make undirected
        ei = torch.stack([
            torch.cat([src,dst], dim=0),
            torch.cat([dst,src], dim=0)
        ])

        # Norm so min is 0
        ts -= ts.min()
        # Repeat so src->dst and dst->src match
        ts = ts.repeat(2)

        # Only src nodes have labels. Fill w empty vals
        # for dst nodes 
        ys = torch.cat([
            torch.full(ys.size(), -1),
            ys
        ])

        f.close()
        prog.close()
        return ei,ts,ys
    
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

        # Repeat so src->dst and dst->src match
        return torch.tensor(feats).repeat(2,1)
    
    def delegate(i):
        f = open(fname, 'r')
        if i == 0:
            return get_edge_job(f)
        else: 
            return get_feat_job(f)
    
    # I think stripping features takes long enough that it's probably worth it to 
    # just delegate it to its own thread (saves like 2 seconds.. eh)
    #                                       This comment shuts up PyLance --v
    (ei,ts,ys),feats = Parallel(2, prefer='processes', batch_size=2)( # type: ignore
        delayed(delegate)(i) for i in range(2)
    )

    n_edges = ei.size(1) // 2
    va = int(n_edges*tr_ratio)
    te = va + int(n_edges * (1-tr_ratio)/2)

    va = ts[va]
    te = ts[te]

    # Node features are a one-hop mean aggr of neighbors. But ensure
    # edges from future timesteps do not contribute to node reprs 
    mp = MessagePassing(aggr='mean')
    tr_x = mp.propagate(ei[:,ts<va], x=torch.eye(ei.max()+1))
    te_x = mp.propagate(ei[:,ts<te], x=torch.eye(ei.max()+1))

    out_csr = HOME+f'precalculated/{name}_csr.pt'
    csr = MixerCSR(
        ei, ts, feats, 
        ys, va, te, 
        os.path.abspath(out_csr)
    )

    out_csr = HOME+f'precalculated/{name}_csr.pt'
    with open(out_csr, 'wb') as f:
        pickle.dump(csr, f)

    torch.save({'ei': ei, 'ts': ts}, HOME+f'precalculated/{name}_ei.pt')
    torch.save({'tr': tr_x, 'te': te_x}, HOME+f'precalculated/{name}_x.pt')
    
    return csr, ei, ts

def get_dataset(name):
    out_csr = HOME+f'precalculated/{name}_csr.pt'

    # Better to do this from here or pickle gets mad
    if os.path.exists(out_csr):
        with open(out_csr, 'rb') as f:
            csr = pickle.load(f)
        et = torch.load(out_csr.replace('_csr', '_ei'))

        return csr, et['ei'], et['ts']
        
    return build_from_csv(name)

if __name__ == '__main__':
    build_from_csv('wikipedia')
    build_from_csv('reddit')