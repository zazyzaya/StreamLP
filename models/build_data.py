import torch 
from torch_geometric.data import Data

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

    return state_change_ts

def get_compressed_edge_ts(edge_index):
    '''
    Number edges by which state change this is rather than actual timestamp
    or edge count. 
    '''
    state_change_ts = find_state_changes(edge_index)
    ts = torch.zeros(edge_index.size(1))

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

    n_nodes = edge_index.max()
    ts = torch.zeros(n_nodes+1)

    # Only care abt nodes that are dst at some point
    # otherwise their embedding will always just be a fn
    # of their original features
    dsts = edge_index[1]
    unique = set(list(dsts.unique()))
    
    # Find the last timestamp where this node appeared
    i = -1
    while unique:
        if dsts[i] in unique:
            ts[dsts[i]] = compressed_ts[i]
            unique.remove(dsts[i])
        
        i -= 1

    return ts 


def to_weighted_ei(edge_index):
    '''
    Save space. Change consecutive edges from v->u into single, weighted ones
    '''
    # Just learned about this fn. So cool!
    ei,ew = edge_index.unique_consecutive(dim=1, return_counts=True)
    return ei, ew