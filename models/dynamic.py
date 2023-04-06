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
        self.gcns = nn.ModuleList(
            [nn.Linear(in_dim, hidden_dim)] + 
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(layers-2)] +
            [nn.Linear(hidden_dim, out_dim)]
        )

        self.activation = act()
        self.aggr = {
            'mean': scatter_mean,
            'sum': scatter 
        }[aggr]

    def recursive_temporal_prop(self, layer, batch, times, x, ptr, idx, ts):
        '''
        Given an edge index in csr format (idx[ptr[0] : ptr[1]] == N(0))
        and timestamp list of same, generate a time constrained embedding
        of every node in batch s.t. only edges < node_t carry messages to that node
        '''
        if layer == 0: 
            return x[batch]
        
        # Get potential neigbors plus self {N_{t'<t}(v_t) U v_t}
        # TODO make this more efficient
        n_ptr = []
        n_batch = []
        n_ts = []

        for i in range(batch.size(0)):
            b = batch[i]
            t = times[i]

            neighbors = idx[ptr[b]:ptr[b+1]]
            n_times = ts[ptr[b]:ptr[b+1]]
            n_mask = n_times <= times[i]
            
            temporal_neighbors = neighbors[n_mask]

            n_batch.append(temporal_neighbors)
            n_ts.append(n_times[n_mask])
            n_ptr += [i] * temporal_neighbors.size(0)

        # Add self loops 
        n_ptr += list(range(batch.size(0)))
        n_batch.append(batch)
        n_ts.append(times)

        # Cat everything together
        n_ptr = torch.tensor(n_ptr)
        n_batch = torch.cat(n_batch, dim=0)
        n_ts = torch.cat(n_ts, dim=0)

        # Get neighbors' embeddings
        n_embs = self.recursive_temporal_prop(
            layer-1, n_batch, n_ts, x, ptr, idx, ts
        )

        # Aggregate
        x = self.aggr(n_embs, n_ptr, dim=0)
        return self.activation(self.gcns[layer-1](x))
    
if __name__ == '__main__':
    # edge_index: [
    #   [0,0,3,0],
    #   [1,2,0,4]
    # ]
    # 0 ->1  3->0->4
    #   \-->2
    # we expect f(1) == f(2) != f(4) as it has been touched by 
    # v_0 after it has been "poisoned" by v_3 
    x   = torch.tensor([
        [1,0,0], # 0 
        [0,1,0], # 1
        [0,1,0], # 2
        [0,0,1], # 3
        [0,1,0], # 4
    ]).float()
    
    idx = torch.tensor([3,0,0,0])
    ts  = torch.tensor([2,0,1,4])
    ptr = torch.tensor([0,1,2,3,3,4])
    
    gnn = FlowGNN(3, 16, 1, act=nn.Sigmoid)
    x = gnn.recursive_temporal_prop(
        2, torch.arange(5), 
        torch.tensor([3,0,1,4,4]), 
        x, ptr, idx, ts
    )
    print(x)

    '''
    Output: 
    tensor([[0.4886],
            [0.4903],
            [0.4903],
            [0.4858],
            [0.4877]], grad_fn=<SigmoidBackward0>)
    
    So we find f(1) == f(2) != f(4)
    Confirmation that this works. 
    '''