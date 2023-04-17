import numpy as np 
import torch 
from torch import nn 
from torch_geometric.nn import MessagePassing

'''
Implimenting GraphMixer from
    https://arxiv.org/pdf/2302.11636.pdf

Their src is available here:
    https://github.com/CongWeilin/GraphMixer

but has a lot of bells and whistles for testing/ablation
that we don't really need
'''

class TimeEncode(nn.Module):
    """
    out = linear(time_scatter): 1-->time_dims
    out = cos(out)

    Copied straight from their git repo
    """
    def __init__(self, dim, device='cpu'):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim, device=device)
        self.device = device 
        self.reset_parameters()
    
    def reset_parameters(self, ):
        w = torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.dim, dtype=np.float32)).reshape(self.dim, -1).to(self.device)
        self.w.weight = nn.Parameter(w)
        self.w.bias = nn.Parameter(torch.zeros(self.dim, device=self.device))

        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False
    
    @torch.no_grad()
    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1))))
        return output
    

class MLPMixer(nn.Module):
    def __init__(self, dim, context_size, col_expansion_factor=0.5, row_expansion_factor=4, device='cpu'):
        '''
        Expects batch of B x context_size x dim matrices. 
        Will apply the MLP-Mixer as described in GraphMixer, and originally in
            https://proceedings.neurips.cc/paper_files/paper/2021/file/cba0a4ee5ccd02fda0fe3f9a3e7b89fe-Paper.pdf

        Default hyperparams taken from GraphMixer src
        '''
        super().__init__()

        def mlp_factory(in_d, hidden, out_d):
            return nn.Sequential(
                nn.LayerNorm(in_d, device=device),
                nn.Linear(in_d, hidden, device=device),
                nn.GELU(), 
                nn.Linear(hidden, out_d, device=device)
            )
        
        self.col_nn = mlp_factory(
            context_size, 
            int(col_expansion_factor*context_size),
            context_size
        )

        self.row_nn = mlp_factory(
            dim, int(row_expansion_factor*dim), dim
        )

    def forward(self, x):
        '''
        Expects B x K x d input where K is fixed length list of 
        d-dimensional edge features 
        '''
        x = x + self.col_nn(x.permute(0,2,1)).permute(0,2,1)
        x = x + self.row_nn(x)
        return x 
    
class EdgeFeatEncoder(nn.Module):
    def __init__(self, edge_feats, time_feats, out_dim, device='cpu'):
        super().__init__()

        self.ts = TimeEncode(time_feats, device=device)
        self.lin = nn.Linear(edge_feats + time_feats, out_dim, device=device)

    def forward(self, et, ef):
        et = self.ts(et)
        feats = torch.cat([ef, et], dim=-1)
        return self.lin(feats)


class GraphMixer(nn.Module):
    def __init__(
        self, edge_feats, hidden=100, time_dim=100, out_dim=100, 
        layers=1, K=20, device='cpu'
    ):
        super().__init__()

        self.edge_feats = EdgeFeatEncoder(edge_feats, time_dim, hidden, device=device)
        self.mixers = nn.ModuleList(
            [MLPMixer(hidden, K, device=device) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(hidden, device=device)
        self.out_proj = nn.Linear(hidden, out_dim, device=device)
        
        self.K = K
        self.hidden = hidden 
        self.out = out_dim
        self.device=device
    
    def forward(self, et, ef, nid, idx, num_nodes):
        '''
        Given a list of E edges, labeled by node id (assumed unique 0-N)
        and their position in the list of edges for that node (0-K) do forward
        pass of edge features through mlp-mixer
        '''
        # E x d
        x = self.edge_feats(et, ef)
        
        split = torch.zeros(self.K * (num_nodes), x.size(1), device=self.device)
        positions = nid*self.K + idx
        split[positions] += x 

        # B x K x d
        x = split.view(num_nodes, self.K, self.hidden)
        for mixer in self.mixers:
            x = mixer(x)

        x = self.norm(x)

        # B x d
        x = x.mean(dim=1)

        # B x out 
        return self.out_proj(x)
    

class GraphMixer_LP(nn.Module):
    def __init__(self, edge_feats, node_feats, device='cpu', lp_hidden=100, **mixer_kwargs):
        super().__init__()

        self.edge_emb = GraphMixer(edge_feats, device=device, **mixer_kwargs)
        self.src_mlp = nn.Linear(node_feats + self.edge_emb.out, lp_hidden, device=device)
        self.dst_mlp = nn.Linear(node_feats + self.edge_emb.out, lp_hidden, device=device)
        self.proj_out = nn.Linear(lp_hidden*2, 1, device=device)

        self.node_feats = node_feats
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.device = device 
        self.mp = MessagePassing(aggr='mean')

    def embed(self, graph, target, t, node_x):
        neg = torch.randint(0, graph.num_nodes, target.size())
        num_neg = neg.size(1)

        target = torch.cat([target, neg], dim=1)
        batch, target = target.unique(return_inverse=True)

        edge_ts, edge_feats, nid, idx = graph.sample(
            batch, t, self.edge_emb.K
        )

        return torch.cat([
            self.edge_emb(edge_ts, edge_feats, nid, idx, batch.size(0)),
            node_x[batch]
        ], dim=1), target, num_neg

    def lp(self, src, dst, zs):
        src = self.src_mlp(zs[src])
        dst = self.dst_mlp(zs[dst])
        return self.proj_out(torch.cat([src,dst], dim=1))
        
    def forward(self, graph, target, t, last_T, pred=False):
        node_x = self.mp.propagate(
            last_T, size=(graph.num_nodes,self.node_feats), 
            x=graph.node_feats
        )
        
        zs, target, num_neg = self.embed(graph, target, t, node_x)
        target, n_target = target[:,:-num_neg], target[:,-num_neg:]

        pos = self.lp(target[0], target[1], zs)
        neg = self.lp(n_target[0], n_target[1], zs)
        preds = torch.cat([pos,neg], dim=0)

        labels = torch.zeros(pos.size(0)*2,1, device=self.device)
        labels[:pos.size(0)] = 1

        if pred:
            return preds, labels

        loss = self.loss_fn(preds, labels)
        return loss 