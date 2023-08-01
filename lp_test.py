import json 
from random import randint 
import sys 

import numpy as np 
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    roc_auc_score as auc_score, average_precision_score as ap_score,
    f1_score, precision_score, recall_score, accuracy_score)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM
import torch 
from torch import nn 
from torch_geometric.data import Data 
from torch_geometric.nn import GCNConv, Sequential
from torch.optim import Adam 
from tqdm import tqdm 

from strgnn_loader import load_network_repo
from models.optimized_tgb import StreamTGBase, BatchTGBase

SECONDS_PER_DAY = 60*60*24

def insert_anoms(g, percent=1): 
    n_edges = g.edge_index.size(1)
    end_tr = n_edges // 2 

    tr = Data(
        edge_index=g.edge_index[:, :end_tr],
        edge_attr =g.edge_attr[:end_tr],
        ts = g.ts[:end_tr],
        num_nodes = g.num_nodes 
    )

    te = Data(
        edge_index=g.edge_index[:, end_tr:],
        edge_attr =g.edge_attr[end_tr:],
        ts = g.ts[end_tr:],
        num_nodes = g.num_nodes 
    )

    n_nodes = g.num_nodes-1
    n_anoms = int(te.edge_index.size(1) * (percent/100))
    y = torch.zeros(te.edge_index.size(1))
    for _ in range(n_anoms): 
        # Generate random edge
        idx = randint(0, te.edge_index.size(1)-1)
        edge = torch.tensor([
            [randint(0, n_nodes)], [randint(0, n_nodes)]
        ])
        ew = te.edge_attr[idx:idx+2].mean(0, keepdim=True)
        ts = te.ts[idx:idx+2].mean(0, keepdim=True)

        # Insert it
        te.edge_index = torch.cat([
            te.edge_index[:, :idx], edge, te.edge_index[:, idx:]
        ], dim=1) # type: ignore 
        te.edge_attr = torch.cat([
            te.edge_attr[:idx], ew, te.edge_attr[idx:]
        ]) # type: ignore
        te.ts = torch.cat([
            te.ts[:idx], ts, te.ts[idx:]
        ])
        y = torch.cat([
            y[:idx], torch.tensor([1]), y[idx:]
        ])

    te.y = y 
    return tr, te     

def generate_tg(tr,te, indi=False):
    tgb = StreamTGBase(1, tr.num_nodes+1)
    tr_edges = []
    te_edges = []

    for i in tqdm(range(tr.edge_index.size(1)), desc='Tr'): 
        tr_edges.append(
            tgb.add_edge(
                tr.edge_index[:, i],
                tr.ts[i] // SECONDS_PER_DAY, 
                tr.edge_attr[i], 
                return_value=True 
            )
        )

        if indi: 
            tgb = StreamTGBase(1, tr.num_nodes+1)

    for i in tqdm(range(te.edge_index.size(1)), desc='Te'): 
        te_edges.append(
            tgb.add_edge(
                te.edge_index[:, i],
                te.ts[i] // SECONDS_PER_DAY, 
                te.edge_attr[i], 
                return_value=True 
            )
        )

        if indi: 
            tgb = StreamTGBase(1, tr.num_nodes+1)

    return torch.stack(tr_edges), torch.stack(te_edges), te.y

def generate_fold_daily_tg(g, tgb=None):
    '''
    Generates node features for each timestep of input
    graph. Optionally, continue using aggregated features 
    from older TGB instance (as when going from tr to te)
    '''
    if tgb is None: 
        tgb = BatchTGBase(1, g.num_nodes+1) 

    gs = []

    st_day = int((g.ts[0] // SECONDS_PER_DAY).item())
    en_day = int((g.ts[-1] // SECONDS_PER_DAY).item())
    for d in tqdm(range(st_day, en_day+1)): 
        mask = (
            g.ts >= d * SECONDS_PER_DAY
        ).logical_and(
            g.ts < (d+1) * SECONDS_PER_DAY
        )

        if mask.sum() == 0: 
            continue 

        if 'y' in g.keys: 
            y = g.y[mask]
        else: 
            y = None 

        g_t = Data(
            x = tgb.add_batch(
                    g.edge_index[:, mask],
                    g.ts[mask], 
                    g.edge_attr[mask], 
                    return_value=True
            ),
            edge_index=g.edge_index[:, mask],
            ts=g.ts[mask], 
            edge_attr=g.edge_attr[mask],
            y=y 
        )
        gs.append(g_t)

    return tgb, gs

def generate_daily_tg(tr,te):
    '''
    Given tr graph and te graph, generate dynamic 
    node features for each fold 
    '''
    tgb, tr_x = generate_fold_daily_tg(tr)
    _, te_x = generate_fold_daily_tg(te, tgb=tgb)

    feat_fixer = Pipeline(steps=[
        ('var', VarianceThreshold(0.)), 
        ('scale', RobustScaler())
    ])

    x = torch.cat([t.x for t in tr_x], dim=0)
    feat_fixer.fit(x) 

    for t in tr_x: 
        t.x = torch.from_numpy(feat_fixer.transform(t.x))
    for t in te_x: 
        t.x = torch.from_numpy(feat_fixer.transform(t.x))

    return tr_x, te_x 

def eval_model(model, tr_x, te_x, y): 
    model.fit(tr_x)
    y_hat = model.predict(te_x)
    preds = model.decision_function(te_x)

    # Make it so benign == 0; anom == 1
    y_hat[y_hat == 1] = 0 
    y_hat[y_hat == -1] = 1

    np.savez('tmp', y_hat=y_hat, preds=preds)

    return {
        'pr': precision_score(y, y_hat),
        're': recall_score(y,y_hat),
        'f1': f1_score(y,y_hat),
        'ac': accuracy_score(y,y_hat),
        'auc': auc_score(y, preds),
        'ap': ap_score(y, preds)
    }

def oc_svm(tr_x, te_x, y):
    model = Pipeline(steps=[
        ('var', VarianceThreshold(0.)), 
        ('scale', RobustScaler()),
        ('oc-svm', OneClassSVM(cache_size=2048))
    ])
    return eval_model(model, tr_x, te_x, y)

def lof(tr_x, te_x, y, knn=20): 
    '''
    Best values: 
        LOF (knn=2): (when time in terms of days)
    {
        "pr": 0.0084809717181155,
        "re": 0.7892976588628763,
        "f1": 0.016781625542202947,
        "ac": 0.08481980342191482,
        "auc": 0.7101053917497124,
        "ap": 0.02125996772125157
    }
    '''
    model = Pipeline(steps=[
        ('var', VarianceThreshold(0.)), 
        ('scale', RobustScaler()),
        ('lof', LocalOutlierFactor(n_neighbors=knn, novelty=True))
    ])
    return eval_model(model, tr_x, te_x, y)

def dnn(tr_x, te_x, y, hidden): 
    feat_fixer = Pipeline(steps=[
        ('var', VarianceThreshold(0.)), 
        ('scale', RobustScaler())
    ])
    tr_x = torch.from_numpy(feat_fixer.fit_transform(tr_x))
    te_x = torch.from_numpy(feat_fixer.transform(te_x))

    class DNN(nn.Module):
        def __init__(self, hidden):
            super().__init__() 

            def block(i,o): 
                return nn.Sequential(
                    nn.Linear(i,o),
                    nn.Dropout(),
                    nn.ReLU(), 
                )

            self.src = nn.Sequential(
                block(tr_x.size(1)//2, hidden*2),
                block(hidden*2, hidden)
            )

            self.dst = nn.Sequential(
                block(tr_x.size(1)//2, hidden*2),
                block(hidden*2, hidden)
            )

            self.net = nn.Sequential(
                nn.Linear(hidden, 1)
            )

            self.bce = nn.BCEWithLogitsLoss()

        def forward(self, x, labels): 
            preds = self.inference(x)
            return self.bce(preds, labels)

        def inference(self, x):
            src = x[:, :x.size(1)//2]
            dst = x[:, x.size(1)//2:]

            src = self.src(src); dst = self.dst(dst)
            x = src*dst 
            return self.net(x)
        
    model = DNN(hidden)
    opt = Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    labels = torch.zeros(tr_x.size(0)*2,1) 
    labels[tr_x.size(0):] = 1 

    print()
    best = (float('inf'), None)
    for e in range(2000):
        fake_src = tr_x[torch.randperm(tr_x.size(0)), :tr_x.size(1)//2]
        fake_dst = tr_x[torch.randperm(tr_x.size(0)), tr_x.size(1)//2:]
        fake_x = torch.cat([fake_src, fake_dst], dim=1)

        opt.zero_grad()
        model.train()
        loss = model(torch.cat([tr_x, fake_x], dim=0), labels)
        loss.backward()
        opt.step()

        print(f"\r[{e}] Loss: {loss.item():0.4f}", end='')

        with torch.no_grad():
            model.eval()
            preds = torch.sigmoid(model.inference(te_x))
            auc = auc_score(y, preds)
            ap = ap_score(y, preds)

            print(f"\tAUC: {auc:0.4f}  AP: {ap:0.4f}", end='')
            if loss.item() < best[0]: 
                best = (loss.item(), (auc,ap))
    print()

    # Best meaning epoch where model had lowest loss (on tr data)
    print(f"Best AUC: {best[1][0]}\nBest AP: {best[1][1]}")

def dnn_pipeline(fname, hidden=32): 
    g = load_network_repo(fname=fname)
    tr, te = insert_anoms(g)
    tr, te, y = generate_tg(tr, te)
    torch.save({'tr':tr, 'te':te, 'y':y}, f'StrGNN_Data/{fname}-tgb.pt')
    
    #data = torch.load(f'StrGNN_Data/{fname}-tgb.pt')
    #tr = data['tr']; te = data['te']; y = data['y']

    dnn(tr, te, y, hidden)

def gcn(tr_gs, te_gs): 
    '''
    Best AUC: 0.8314484898633745 (StrGNN: 0.8179)
    Best AP: 0.08409933645323109
    '''
    class GCN(nn.Module):
        def __init__(self, hidden, latent=32):
            super().__init__() 
            self.in_gcn = GCNConv(tr_gs[0].x.size(1), hidden)
            #self.hidden = GCNConv(hidden, hidden)
            self.out = GCNConv(hidden, hidden)

            # Doing in the style of jumping knowledge net
            self.net = nn.Sequential(
                nn.Linear(hidden*2, hidden),
                nn.ReLU(),
                nn.Linear(hidden, latent),
                nn.ReLU()
            )

            self.pred = nn.Linear(latent, 1)

            self.bce = nn.BCEWithLogitsLoss()

        def forward(self, x, ei, edges, labels): 
            x1 = torch.relu(self.in_gcn(x, ei))
            #x2 = torch.relu(self.hidden(x1, ei))
            x3 = torch.relu(self.out(x1, ei))

            x = torch.cat([x1,x3], dim=1)
            z = self.net(x) 
            preds = self.pred(z[edges[0]] * z[edges[1]])

            return self.bce(preds, labels)

        def inference(self, x, ei, edges):
            x1 = torch.relu(self.in_gcn(x, ei))
            #x2 = torch.relu(self.hidden(x1, ei))
            x3 = torch.relu(self.out(x1, ei))

            x = torch.cat([x1,x3], dim=1)
            z = self.net(x)

            return torch.sigmoid(
                self.pred(z[edges[0]] * z[edges[1]])
            )
        
    model = GCN(64)
    opt = Adam(model.parameters(), lr=0.01)

    print()
    best = (float('inf'), None)

    # Prepend last tr embedding to test so they can use the 
    # node features from the previous timestamps
    te_gs = [tr_gs[-1]] + te_gs 

    for e in range(2000):
        model.train()
        for i in range(1, len(tr_gs)):
            x = tr_gs[i].x            # Use previous encoding as features
            ei = tr_gs[i].edge_index    # To predict on current edges

            fake_edges = torch.randint(0, tr_gs[0].num_nodes, (ei.size()))
            edges = torch.cat([ei, fake_edges], dim=1)
            labels = torch.zeros((edges.size(1),1))
            labels[ei.size(1):] = 1

            opt.zero_grad()
            loss = model(x, ei, edges, labels)
            loss.backward()
            opt.step()

        print(f"\r[{e}] Loss: {loss.item():0.4E}", end='')

        with torch.no_grad():
            model.eval()
            preds = []
            y = []

            for i in range(1, len(te_gs)): 
                x = te_gs[i].x 
                ei = te_gs[i].edge_index
                preds.append(model.inference(x, ei, ei))
                y.append(te_gs[i].y)

            y = torch.cat(y)
            preds = torch.cat(preds) 

            auc = auc_score(y, preds)
            ap = ap_score(y, preds)

            print(f"\tAUC: {auc:0.4f}  AP: {ap:0.4f}", end='')
            if loss.item() < best[0]: 
                best = (loss.item(), (auc,ap))
    print()

    # Best meaning epoch where model had lowest loss (on tr data)
    print(f"Best AUC: {best[1][0]}\nBest AP: {best[1][1]}")


def gnn_pipeline(fname): 
    g = load_network_repo(fname=fname, force=True)
    tr, te = insert_anoms(g) 
    tr_gs, te_gs = generate_daily_tg(tr, te)

    gcn(tr_gs, te_gs)

if __name__ == '__main__':
    dnn_pipeline(sys.argv[1], hidden=int(sys.argv[2]))