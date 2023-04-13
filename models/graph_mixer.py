import numpy as np 
import torch 
from torch import nn 

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
    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        self.reset_parameters()
    
    def reset_parameters(self, ):
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.dim, dtype=np.float32))).reshape(self.dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(self.dim))

        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False
    
    @torch.no_grad()
    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1))))
        return output
    

class MLPMixer(nn.Module):
    def __init__(self, dim, context_size, col_expansion_factor=0.5, row_expansion_factor=4):
        '''
        Expects batch of B x context_size x dim matrices. 
        Will apply the MLP-Mixer as described in GraphMixer, and originally in
            https://proceedings.neurips.cc/paper_files/paper/2021/file/cba0a4ee5ccd02fda0fe3f9a3e7b89fe-Paper.pdf

        Default hyperparams taken from GraphMixer src
        '''
        super().__init__()

        def mlp_factory(in_d, hidden, out_d):
            return nn.Sequential(
                nn.LayerNorm(in_d),
                nn.Linear(in_d, hidden),
                nn.GELU(), 
                nn.Linear(hidden, out_d)
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
        x = x + self.col_nn(x.transpose(0,2,1)).transpose(0,2,1)
        x = x + self.row_nn(x)
        return x 
    
class EdgeFeatEncoder(nn.Module):
    def __init__(self, edge_feats, time_feats, out_dim):
        super().__init__()

        self.ts = TimeEncode(time_feats)
        self.lin = nn.Linear(edge_feats + time_feats, out_dim)

    def forward(self, ef, et):
        et = self.ts(et)
        feats = torch.cat([ef, et], dim=-1)
        return self.lin(feats)


class GraphMixer(nn.Module):
    def __init__(
        self, edge_feats, hidden=100, time_dim=100, out_dim=100, layers=1, K=20
    ):
        super().__init__()

        self.edge_feats = EdgeFeatEncoder(edge_feats, time_dim, hidden)
        self.mixers = nn.ModuleList(
            [MLPMixer(hidden, K) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, out_dim)
        
        self.K = K
        self.hidden = hidden 
        self.out = out_dim

    
    def forward(self, ef, et, nid, idx):
        '''
        Given a list of E edges, labeled by node id (assumed unique 0-N)
        and their position in the list of edges for that node (0-K) do forward
        pass of edge features through mlp-mixer
        '''
        # E x d
        x = self.edge_feats(ef, et)
        
        split = torch.zeros(self.K * (nid.max()+1), x.size(1))
        positions = nid*self.K + idx
        split[positions] += x 

        # B x K x d
        x = split.view(nid.max()+1, self.K, self.hidden)
        for mixer in self.mixers:
            x = mixer(x)

        x = self.norm(x)

        # B x d
        x = x.mean(dim=1)

        # B x out 
        return self.out_proj(x)
    

class GraphMixer_LP(nn.Module):
    def __init__(self, edge_feats, node_feats, lp_hidden=100, **mixer_kwargs):
        super().__init__()

        self.edge_emb = GraphMixer(edge_feats, **mixer_kwargs)
        self.src_mlp = nn.Linear(node_feats + self.edge_emb.out, lp_hidden)
        self.dst_mlp = nn.Linear(node_feats + self.edge_emb.out, lp_hidden)
        self.proj_out = nn.Linear(lp_hidden, 1)

        self.loss_fn = nn.BCELoss()

    def embed(self, graph, target, t, presampled=None):
        batch, target = target.unique(return_inverse=True)
        
        if presampled is None:
            node_feats, edge_feats, edge_ts, nid, idx = sample(
                graph, batch, t, self.edge_emb.K
            )
        else:
            node_feats, edge_feats, edge_ts, nid, idx = presampled

        return torch.cat([
            self.edge_emb(edge_feats, edge_ts, nid, idx),
            node_feats
        ], dim=1), target 

    def lp(self, src, dst, zs):
        src = self.src_mlp(zs[src])
        dst = self.dst_mlp(zs[dst])
        return self.proj_out(src+dst)
        
    def forward(self, graph, target, t, presampled=None):
        zs, target = self.embed(graph, target, t, presampled=presampled)
        
        n_edges = target.size(1)
        n_target = torch.stack([
            target[0, torch.randperm(n_edges)], 
            target[1, torch.randperm(n_edges)], 
        ], dim=0)

        pos = self.lp(target[0], target[1], zs)
        neg = self.lp(n_target[0], n_target[1], zs)
        preds = torch.cat([pos,neg], dim=0)

        labels = torch.zeros(pos.size(1)*2)
        labels[:pos.size(1)] = 1
        loss = self.loss_fn(labels, preds)

        return loss 

# TODO 
def sample(graph, nodes, t, K):
    '''
        samples graph pull find node/edge feats out of csr matrix
        given a timestamp that they must all come before

        returns: 
            node_feats
            edge_feats
            edge_ts
            nid - unique id associated with edge feature from [0,|nodes|) 
            idx - unique id from [0, K) 
    '''
    return [None]