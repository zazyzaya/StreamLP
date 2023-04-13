import torch 
from torch import nn 
from torch_scatter import scatter_mean, scatter

def find_state_changes(edge_index):
    is_src=torch.zeros(edge_index.max()+1, dtype=torch.bool)
    state_change_ts = [0]

    for i, (src,dst) in enumerate(edge_index.T):
        is_src[src] = True 
        if is_src[dst]:
            state_change_ts.append(i)
            is_src[dst] = False 

    return state_change_ts

def sequential_prop(x, edge_index, state_changes, cur_x=None):
    if cur_x is None:
        cur_x = torch.zeros(x.size())

    for i in range(len(state_changes)-1):
        st,en = state_changes[i:i+1]
        ei = edge_index[:, st:en]
        scatter(x[ei[0]], ei[1], dim=0, out=cur_x)
        return cur_x
    
class FlowGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layers=2, act=nn.ReLU, aggr='mean'):
        super().__init__()

        def gcn_constructor(in_d, out_d):
            return nn.Sequential(
                nn.Linear(in_d, out_d),
                nn.BatchNorm1d(out_d)
            )
        self.gcns = nn.ModuleList(
            [gcn_constructor(in_dim, hidden_dim)] + 
            [gcn_constructor(hidden_dim, hidden_dim) for _ in range(layers-2)] +
            [gcn_constructor(hidden_dim, out_dim)]
        )
        self.n_layers = layers

        self.activation = act()
        self.aggr = {
            'mean': scatter_mean,
            'sum': scatter 
        }[aggr]

    def forward(self, g, batch=None):
        if batch is None:
            batch = torch.arange(g.x.size(0))
        
        return self.recursive_temporal_prop(
            self.n_layers, 
            batch,
            g.node_ts[batch],
            g 
        )

    def recursive_temporal_prop(self, layer, batch, batch_ts, g):
        '''
        Given an edge index in csr format (idx[ptr[0] : ptr[1]] == N(0))
        and timestamp list of same, generate a time constrained embedding
        of every node in batch s.t. only edges < node_t carry messages to that node
        '''
        if layer == 0: 
            return g.x[batch]
        
        # Get potential neigbors plus self {N_{t'<t}(v_t) U v_t}
        # TODO make this more efficient
        n_ptr = []
        n_batch = []
        n_ts = []

        for i in range(batch.size(0)):
            b = batch[i]
            t = batch_ts[i]

            neighbors, n_times, _ = g.csr_ei[b]
            n_mask = n_times <= t
            
            temporal_neighbors = neighbors[n_mask]

            n_batch.append(temporal_neighbors)
            n_ts.append(n_times[n_mask])
            n_ptr += [i] * temporal_neighbors.size(0)

        # Add self loops 
        n_ptr += list(range(batch.size(0)))
        n_batch.append(batch)
        n_ts.append(g.node_ts[batch])

        # Cat everything together
        n_ptr = torch.tensor(n_ptr)
        n_batch = torch.cat(n_batch, dim=0)
        n_ts = torch.cat(n_ts, dim=0)

        # Remove duplicate node,ts tuples
        (n_batch,n_ts), dupe_idx = torch.stack(
            [n_batch, n_ts]
        ).unique(dim=1, return_inverse=True)

        # Get neighbors' embeddings
        n_embs = self.recursive_temporal_prop(
            layer-1, n_batch, n_ts, g
        )

        # Aggregate
        # TODO I feel like there's a more efficient way to do this
        # scatter. Like indexing n_embs and then reindexing into the 
        # n_ptr seems like it could be consolidated, but I'm not sure
        x = self.aggr(n_embs[dupe_idx], n_ptr, dim=0)
        return self.activation(self.gcns[layer-1](x))


class L2Decoder(nn.Module):
    def forward(self, x1, x2):
        return (x1-x2).pow(2).sum(dim=1, keepdim=True)

class Dot(nn.Module):
    def forward(self, x1, x2):
        return (x1 * x2).sum(dim=1, keepdim=True)

class HadNet(nn.Module):
    def __init__(self, dim):
        self.net = nn.Linear(dim, 1)
    def forward(self, x1, x2):
        return self.net(x1 * x2)

class DeepHadNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim,dim),
            nn.Linear(dim, 1)
        )
    
    def forward(self, x1, x2):
        return self.net(x1 * x2)

class FlowGNN_LP(FlowGNN):
    def __init__(self, in_dim, hidden_dim, out_dim, layers=2, dec='hadnet', act=nn.ReLU, aggr='mean'):
        super().__init__(in_dim, hidden_dim, out_dim, layers, act, aggr)

        self.decode_net = {
            'l2': L2Decoder(),
            'dot': Dot(),
            'hadnet': HadNet(out_dim),
            'deephad': DeepHadNet(out_dim)
        }[dec]
        self.loss_fn = nn.BCEWithLogitsLoss()


    def forward(self, g, target, batch=None):
        batch, target = target.unique(return_inverse=True)
        embs = super().forward(g, batch)

        src,dst = target
        n_src,n_dst = torch.randint(0, target.max(), (2, target.size(1)))

        pos = self.predict(embs, src, dst, activation=False)
        neg = self.predict(embs, n_src, n_dst, activation=False)
        
        preds = torch.cat([pos,neg], dim=0)
        target = torch.zeros(preds.size())
        target[:pos.size(0)] = 1. 

        return self.loss_fn(preds, target)


    def predict(self, embs, src, dst, activation=True):
        pred = self.decode_net(embs[src], embs[dst])
        
        if activation:
            return torch.sigmoid(pred)
        else:
            return pred 
        

    def lp(self, g, target):
        embs = super().forward(g, torch.arange(target.max()+1))
        return self.predict(embs, target[0], target[1])