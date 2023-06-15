import torch 
from torch import nn 
from torch_scatter import segment_add_csr

class FiLM(nn.Module):
    '''
    Implementing feature-wise linear modulation (no clue
    how they got the acronym FiLM...)
    '''
    def __init__(self, dim, device='cpu'):
        super().__init__()

        self.e_theta = nn.Parameter(
            torch.empty(1, dim, device=device)
        )
        nn.init.kaiming_normal_(self.e_theta)

        self.alpha = nn.Sequential(
            nn.Linear(dim, dim), 
            nn.LeakyReLU()
        )
        self.beta = nn.Sequential(
            nn.Linear(dim,dim),
            nn.LeakyReLU()
        )

    def forward(self, x):
        alpha = self.alpha(x)
        beta = self.beta(x)

        params = (self.e_theta)*(1+alpha) + beta 
        return torch.sigmoid(
            (params * x).sum(dim=1, keepdim=True)
        )
    
def sample_temporal_neighbors(batch, csr, t):
    neighbors_unfiltered, times_unfiltered = zip(*csr[batch])
    neigbhors, times = [],[]
    for i in range(len(neighbors_unfiltered)):
        mask = times_unfiltered[i] <= t [i]
        neigbhors.append(neighbors_unfiltered[i][mask])
        times.append(times_unfiltered[i][mask].max())

    return neigbhors, torch.tensor(times)

class TemporalGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layers=2):
        # TODO make able to handle multiple layers
        self.node = nn.ParameterList([
            nn.Linear(in_dim, hidden_dim),
            nn.Linear(hidden_dim, out_dim)
        ])
        self.neighborhood = nn.ParameterList([
            nn.Linear(in_dim, hidden_dim),
            nn.Linear(hidden_dim, out_dim)
        ])

        self.t_kernel = nn.Parameter(torch.rand(1))

    def kernel(self, t):
        return torch.exp(
            -self.t_kernel * t
        )

    def forward(self, batch, ts, x, csr, layer=-1):
        if layer == -1:
            layer = len(self.node)

        if layer == 0:
            return x[batch] 

        neighbors, n_times = sample_temporal_neighbors(batch, csr, ts)
        neighbors += list(batch.split(1))
        sizes = [n.size(0) for n in neighbors]

        # Build mapping of which nodes need to be reduced where
        sizes = [n.size(0) for n in neighbors]
        idxptr = [0]
        for s in sizes:
            idxptr.append(s+idxptr[-1])
        idxptr = torch.tensor(idxptr)

        # Get embedding from previous layer
        neighbors = torch.cat(neighbors)
        h_prev = self.forward(neighbors, n_times, x, csr, layer-1)
        h_prev = segment_add_csr(h_prev, idxptr)
        h0, hn = h_prev.split(batch.size(-1))

        # Get neighbor embeddings weighted by time
        hn *= self.kernel(ts-n_times)
        hn = self.neighborhood[layer-1](hn)

        # Get embedding of self 
        h0 = self.node[layer-1](h0)

        # Add to aggr'd neighbor embedding
        h = torch.relu(h0 + hn)

        return h
    

class TREND(nn.Module):
    def __init__(self, in_dim, hidden=16, out=128, layers=2):
        super().__init__()
        
        self.gnn = TemporalGNN(in_dim, hidden, out, layers=layers)
        self.conditional = FiLM(out*2)
        self.dynamics = nn.Sequential(
            nn.Linear(out, 1), 
            nn.ReLU()
        )

    def forward(self, x, csr, ei, ts, counts):
        batch = ei.view(ei.size(1)*2, 1)
        h = self.gnn(batch, ts.repeat(2), x, csr)

        lamb = torch.cat(h.split(ei.size(1)), dim=1)
        lamb = self.conditional(lamb)
        dyn = self.dynamics(h) 

        