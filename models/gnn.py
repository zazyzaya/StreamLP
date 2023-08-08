import torch 
from torch import nn 
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_remaining_self_loops, to_undirected

class DropEdge(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p 

    def forward(self, ei):
        if self.training:
            rnd = torch.rand(ei.size(1))
            return ei[:, rnd>self.p]
        
        return ei 


class GCN(nn.Module):
    def __init__(self, in_dim, hidden, jknet=True, latent=32):
        super().__init__() 
        self.in_gcn = GCNConv(in_dim, hidden)
        self.hidden = GCNConv(hidden, hidden)
        self.out = GCNConv(hidden, hidden)

        self.drop_edge = DropEdge()

        # Doing in the style of jumping knowledge net
        self.jknet = jknet 
        if jknet:
            self.net = nn.Sequential(
                nn.Linear(hidden*3, hidden),
                nn.ReLU(),
                nn.Linear(hidden, latent),
                nn.ReLU()
            )
        else: 
            self.net = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, latent),
                nn.ReLU()
            )

        self.pred = nn.Linear(latent, 1)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, x, edges, neg): 
        z = self.embed(x, edges)
        pos = self.pred(z[edges[0]] * z[edges[1]])
        neg = self.pred(z[neg[0]] * z[neg[1]])

        return (
            self.bce(pos, torch.ones(pos.size(0),1)) + 
            self.bce(neg, torch.zeros(neg.size(0),1))
        )

    def embed(self, x, ei):
        ei = add_remaining_self_loops(ei)[0]
        ei = to_undirected(ei)

        x1 = torch.relu(self.in_gcn(x, self.drop_edge(ei)))
        x2 = torch.relu(self.hidden(x1, self.drop_edge(ei)))
        x3 = torch.relu(self.out(x1, self.drop_edge(ei)))

        if self.jknet:
            x = torch.cat([x1,x2,x3], dim=1)
        else: 
            x = x3 

        return self.net(x) 

    def inference(self, z, edges):
        return torch.sigmoid(
            self.pred(z[edges[0]] * z[edges[1]])
        )