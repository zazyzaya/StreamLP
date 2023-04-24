from collections import defaultdict
import os 

from joblib import Parallel, delayed
import torch 
from torch_geometric.data import Data
from tqdm import tqdm 

MIXER_HOME = os.path.dirname(__file__) + '/../mixer-datasets/'

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

def build_data_obj(ei, lp_test_data=0.7, x=None):
    ei, ew = to_weighted_ei(ei) 

    va_edges = int(ei.size(1)*lp_test_data)
    te_edges = int(ei.size(1)*(lp_test_data + (1-lp_test_data)/2)) 
    tr,va,te = ei[:,:va_edges], ei[:,va_edges:te_edges], ei[:, te_edges:]

    ts = get_compressed_edge_ts(tr)
    node_ts = get_node_ts(tr, ts)

    if x is None: 
        x = torch.eye(ei.max()+1)

    csr =  CSRData(tr, ts, ew)
    data = Data(
        x,
        csr_ei=csr,
        node_ts=node_ts,
        tr_edge_index=tr, 
        va_edge_index=va,
        te_edge_index=te
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
    home = '../ctdne-datasets/'
    out_f = home + f'precalculated/{name}.pt'

    if os.path.exists(out_f) and not force:
        return torch.load(out_f)

    fname, sep = CTDNE_FNAMES[name]
    fname = home+fname
    f = open(fname, 'r')
    
    line = f.readline()
    while line.startswith('%'):
        line = f.readline()

    tokens = line.split(sep)
    has_e_feats = len(tokens) == 4

    src,dst,ts = [],[],[]
    while(line):
        tokens = line.split(sep) 
        (s,d),t = tokens[0:2], tokens[-1].strip()
        
        # For now just skip negative edges? Only applies to wiki really 
        if has_e_feats and tokens[2] == '-1':
            line = f.readline() 
            continue 

        src.append(int(s))
        dst.append(int(d))
        ts.append(float(t))

        line = f.readline() 
    
    f.close()
    ei = torch.tensor([src,dst])
    ts = torch.tensor(ts, dtype=torch.long)

    # Pretty sure order is guaranteed, but to be safe
    order = ts.sort().indices
    ei = ei[:, order]
    g = build_data_obj(ei)

    torch.save(g, out_f)
    return g 

def load_mixer(name):
    fname = MIXER_HOME+name+'.csv'
    out_f = MIXER_HOME + f'precalculated/{name}_stream.pt'

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
        ei = torch.stack([src,dst])

        # Norm so min is 0
        ts -= ts.min()

        # Only src nodes have labels. Fill w empty vals
        # for dst nodes 
        ys = torch.cat([
            ys,
            torch.full(ys.size(), -1),
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
    (ei,ts,ys),feats = Parallel(2, prefer='processes', batch_size=2)( # type: ignore
        delayed(delegate)(i) for i in range(2)
    )

    order = ts.sort().indices
    ei = ei[:, order]
    g = build_data_obj(ei)

    torch.save(g, out_f)
    return g 

if __name__ == '__main__':
    [
        load_ctdne(name, force=True) 
        for name in CTDNE_FNAMES.keys()
    ]