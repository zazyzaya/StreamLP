import os 
import sys 
from types import SimpleNamespace
from typing import Any

import pandas as pd 
from sklearn.metrics import (
    accuracy_score as acc_score, average_precision_score as ap_score
)
import torch 
from torch.optim import Adam 
from torch_geometric.data import Data

from models.long_term_tgb import BatchTGBase, StreamTGBase
from models.gnn import GCN
from databuilders.build_mixer_data import build_from_csv

torch.set_num_threads(32)
HOME = 'mixer-datasets/precalculated/'
HP = SimpleNamespace(
    hidden=128, jknet=False,
    lr=0.001, epochs=1000
)

PATIENCE = 100 

'''
Comparing to Xu et al., 2021 TGAT paper


Transductive
            Acc         Ap
Reddit:     92.92 (0.3) 98.12 (0.2)
Wiki:       88.14 (0.2) 95.34 (0.1)

Inductive
            Acc         Ap
Reddit:     90.73 (0.2) 96.62 (0.3)
Wiki:       85.35 (0.2) 93.99 (0.3)
'''

class RandomEdgeSampler():
    '''
    Need to be kind of careful here. For one, these graphs are bipartite 
    so it would be too easy to allow src nodes to be put in the dst area.
    For two, when doing inductive LP need to make sure that only allowed nodes
    can be sampled. 
    '''
    def __init__(self, src,dst):
        self.srcs = src.unique()
        self.dsts = dst.unique()

    def sample(self, size):
        return torch.stack([
            self.srcs[torch.multinomial(torch.ones(self.srcs.size(0)), size, replacement=True)],
            self.dsts[torch.multinomial(torch.ones(self.dsts.size(0)), size, replacement=True)]
        ])

    def __call__(self, size):
        return self.sample(size)

def generate_batch_data(fname, force=False):
    '''
    Returns 70,15,15 split of data. 
    If inductive only returns portions of ei with unseen nodes
    '''
    if os.path.exists(HOME+fname+'-split.pt') and not force:
        return torch.load(HOME+fname+'-split.pt')

    g = torch.load(HOME+fname+'.pt')
    g.ts /= (60*60*24)

    tgb = BatchTGBase(g.edge_attr.size(1), g.num_nodes+1)

    end_tr = int(g.num_edges * 0.7)
    end_va = int(g.num_edges * 0.15) + end_tr 

    tr_ei = g.edge_index[:, :end_tr]
    va_ei = g.edge_index[:, end_tr:end_va]
    te_ei = g.edge_index[:, end_va:]

    print("Building training X")
    tr_x = tgb.add_batch(
        tr_ei,
        g.ts[:end_tr], 
        g.edge_attr[:end_tr],
        return_value=True 
    )

    print("Building val X")
    va_x = tgb.add_batch(
        va_ei,
        g.ts[end_tr:end_va], 
        g.edge_attr[end_tr:end_va],
        return_value=True 
    )

    print("Building test X")
    te_x = tgb.add_batch(
        te_ei, 
        g.ts[end_va:], 
        g.edge_attr[end_va:],
        return_value=True 
    )

    tr_observed = g.edge_index[:, :end_tr].unique().unsqueeze(-1)
    
    va_src_observed = (va_ei[0] == tr_observed).sum(dim=0) 
    va_dst_observed = (va_ei[1] == tr_observed).sum(dim=0) 
    
    te_src_observed = (te_ei[0] == tr_observed).sum(dim=0) 
    te_dst_observed = (te_ei[1] == tr_observed).sum(dim=0) 

    va_ind_mask = ~torch.logical_or(va_src_observed, va_dst_observed)
    te_ind_mask = ~torch.logical_or(te_src_observed, te_dst_observed)
    va_trans_mask = torch.logical_and(va_src_observed, va_dst_observed)
    te_trans_mask = torch.logical_and(te_src_observed, te_dst_observed)

    out = (
        Data(tr_x, edge_index=tr_ei), 
        Data(va_x, edge_index=va_ei, ind_mask=va_ind_mask, trans_mask=va_trans_mask),
        Data(te_x, edge_index=te_ei, ind_mask=te_ind_mask, trans_mask=te_trans_mask)
    )

    torch.save(out, HOME+fname+'-split.pt')
    return out 


@torch.no_grad()
def eval(g,model, tr_edges, real_edges,fake_edges):
    edges = torch.cat([tr_edges, g.edge_index], dim=1)
    z = model.embed(g.x, edges)

    labels = torch.zeros(real_edges.size(1) + fake_edges.size(1),1)
    labels[:real_edges.size(1)] = 1
    edges = torch.cat([real_edges, fake_edges], dim=1)
    preds = model.inference(z, edges)
    y_hat = (preds > 0.5).long()

    acc = acc_score(labels, y_hat)
    ap = ap_score(labels, preds)

    return acc, ap 


def train(hp, tr,va,te): 
    model = GCN(tr.x.size(1), hp.hidden, jknet=hp.jknet)
    opt = Adam(model.parameters(), lr=hp.lr)

    best_ind = (0, None, 0)
    best_tran = (0, None, 0)
    early_stopping = [False, False]

    tran_sampler = RandomEdgeSampler(
        *torch.cat([
            tr.edge_index, 
            va.edge_index, 
            te.edge_index
        ], dim=1)
    )
    tr_sampler = RandomEdgeSampler(*tr.edge_index)
    va_ind_sampler = RandomEdgeSampler(*va.edge_index)
    te_ind_sampler = RandomEdgeSampler(*te.edge_index)

    e = 0 
    while True:
        opt.zero_grad()
        model.train() 
        loss = model.forward(tr.x, tr.edge_index, tr_sampler(tr.edge_index.size(1)))
        loss.backward()
        opt.step() 

        print(f"[{e}] Loss: {loss.item():0.4f}")

        model.eval() 
        va_tran = eval(va, model, tr.edge_index, va.edge_index[:, va.trans_mask], tran_sampler(va.trans_mask.sum()))
        print(f"\t  T-VAcc: {va_tran[0]:0.4f}  T-VAP: {va_tran[1]:0.4f}")
        va_ind = eval(va, model, tr.edge_index, va.edge_index[:, va.ind_mask], va_ind_sampler(va.trans_mask.sum()))
        print(f"\t  I-VAcc: {va_ind[0]:0.4f}  I-VAP: {va_ind[1]:0.4f}")

        te_tran = eval(te, model, tr.edge_index, te.edge_index[:, te.trans_mask], tran_sampler(te.trans_mask.sum()))
        print(f"\t  T-TAcc: {te_tran[0]:0.4f}  T-TAP: {te_tran[1]:0.4f}")
        te_ind = eval(te, model, tr.edge_index, te.edge_index[:, te.ind_mask], te_ind_sampler(te.trans_mask.sum()))
        print(f"\t  I-TAcc: {te_ind[0]:0.4f}  I-TAP: {te_ind[1]:0.4f}")

        # Stop tracking best metric after 10 epochs of no improvement
        # Early stop if both inductive and transductive stopped improving
        if va_tran[1] > best_tran[0] and not early_stopping[0]: 
            best_tran = (va_tran[1], te_tran, e)
        else:
            if e-best_tran[2] >= PATIENCE: 
                early_stopping[0] = True 

        if va_ind[1] > best_ind[0] and not early_stopping[1]:
            best_ind = (va_ind[1], te_ind, e)
        else:
            if e-best_ind[2] >= PATIENCE: 
                early_stopping[1] = True 

        if early_stopping[0] and early_stopping[1]:
            print("Early stopping!")
            break 

        e += 1

    print("Best:")
    print("Transductive")
    print(f"\tAcc: {best_tran[1][0]:0.4f}  AP: {best_tran[1][1]:0.4f}")
    print("Inductive")
    print(f"\tAcc: {best_ind[1][0]:0.4f}  AP: {best_ind[1][1]:0.4f}")

    return best_tran, best_ind

if __name__ == '__main__':
    #train(HP, tr,va,te)

    '''
    Baseline: hidden=128, 3-layer, jknet=True
    Best:
    Transductive
            Acc: 0.9325  AP: 0.9743
    Inductive
            Acc: 0.9001  AP: 0.4011
    '''
    for fname in ['wikipedia', 'reddit']:
        tr,va,te = generate_batch_data(fname)

        results = []
        for jknet in [True, False]: 
            for hidden in [64, 128, 256, 512]:
                HP.hidden = hidden
                HP.jknet = jknet
                (_, (tr_acc, tr_ap), tr_e), (_, (ind_acc, ind_ap), ind_e) = train(HP, tr,va,te)
                results.append({
                    'tr_acc': tr_acc, 
                    'tr_ap': tr_ap, 
                    'tr_e': tr_e, 
                    'ind_acc': ind_acc, 
                    'ind_ap': ind_ap,
                    'ind_e': ind_e,
                    'hidden': hidden,
                    'jk': jknet,
                })

                # Do inside every loop in case of crash
                r = pd.DataFrame(results)
                
                with open(f'results/tg-to-gnn/{fname}.csv', 'w+') as f:
                    r.to_csv(f)