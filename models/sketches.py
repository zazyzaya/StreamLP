import torch 
from torch import nn 
from torch_scatter import scatter_mean, scatter

'''
Sketching out ideas to try to improve temporal prop
''' 
# Will be params in the class eventually. Just making them
# globals while messing with algorithm
aggr = scatter_mean
gcns = [nn.Linear(3, 16), nn.Linear(16,1)]
activation = nn.Sigmoid()

def recursive_temporal_prop_naiive(layer, batch, times, x, ptr, idx, ts):
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
        n_mask = n_times <= t
        
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
    n_embs = recursive_temporal_prop_naiive(
        layer-1, n_batch, n_ts, x, ptr, idx, ts
    )

    # Aggregate
    x = aggr(n_embs, n_ptr, dim=0)
    return activation(gcns[layer-1](x))

def recursive_temporal_prop_no_repeats(layer, batch, times, x, ptr, idx, ts):
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
        n_mask = n_times <= t
        
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

    # Remove duplicate node,ts tuples
    (n_batch,n_ts), dupe_idx = torch.stack(
        [n_batch, n_ts]
    ).unique(dim=1, return_inverse=True)

    # Get neighbors' embeddings
    n_embs = recursive_temporal_prop_no_repeats(
        layer-1, n_batch, n_ts, x, ptr, idx, ts
    )

    # Aggregate
    x = aggr(n_embs[dupe_idx], n_ptr, dim=0)
    return activation(gcns[layer-1](x))



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
    
    # Need to find way to do this programatically. Times broken into 
    # count of when src node goes to dst. (Happens once at ei[:,2])
    ts  = torch.tensor([1,0,0,1]) 
    ptr = torch.tensor([0,1,2,3,3,4])
    
    x = recursive_temporal_prop_no_repeats(
        2, torch.arange(5), 
        torch.tensor([1,0,0,1,1]), 
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