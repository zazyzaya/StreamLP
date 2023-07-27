import json 
from random import randint 

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
from torch.optim import Adam 
from tqdm import tqdm 

from strgnn_loader import load_uci
from models.optimized_tgb import StreamTGBase

def insert_anoms(g, percent=1): 
    n_edges = g.edge_index.size(1)
    end_tr = n_edges // 2 

    tr = Data(
        edge_index=g.edge_index[:, :end_tr],
        edge_attr =g.edge_attr[:end_tr].unsqueeze(-1),
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
    te.edge_attr = te.edge_attr.unsqueeze(-1)
    return tr, te     

def generate_tg(tr,te):
    tgb = StreamTGBase(1, tr.num_nodes+1)
    tr_edges = []
    te_edges = []

    for i in tqdm(range(tr.edge_index.size(1)), desc='Tr'): 
        tr_edges.append(
            tgb.add_edge(
                tr.edge_index[:, i],
                tr.ts[i] // (60*60*24), 
                tr.edge_attr[i], 
                return_value=True 
            )
        )

    for i in tqdm(range(te.edge_index.size(1)), desc='Te'): 
        te_edges.append(
            tgb.add_edge(
                te.edge_index[:, i],
                te.ts[i] // (60*60*24), 
                te.edge_attr[i], 
                return_value=True 
            )
        )

    return torch.stack(tr_edges), torch.stack(te_edges), te.y

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

def dnn(tr_x, te_x, y): 
    '''
    Best AUC: 0.8314484898633745 (StrGNN: 0.8179)
    Best AP: 0.08409933645323109
    '''
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
        
    model = DNN(32)
    opt = Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    labels = torch.zeros(tr_x.size(0)*2,1) 
    labels[tr_x.size(0):] = 1 

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

        print(f"[{e}] Loss: {loss.item():0.4f}", end='')

        with torch.no_grad():
            model.eval()
            preds = torch.sigmoid(model.inference(te_x))
            auc = auc_score(y, preds)
            ap = ap_score(y, preds)

            print(f"\tAUC: {auc:0.4f}  AP: {ap:0.4f}")
            if loss.item() < best[0]: 
                best = (loss.item(), (auc,ap))

    # Best meaning epoch where model had lowest loss (on tr data)
    print(f"Best AUC: {best[1][0]}\nBest AP: {best[1][1]}")

if __name__ == '__main__':
    #g = torch.load('StrGNN_Data/uci.pt')
    #tr, te = insert_anoms(g)
    #tr, te, y = generate_tg(tr, te)
    #torch.save({'tr':tr, 'te':te, 'y':y}, 'StrGNN_Data/uci-tgb.pt')
    
    data = torch.load('StrGNN_Data/uci-tgb.pt')
    tr = data['tr']; te = data['te']; y = data['y']

    #for i in [100, 50, 25, 10, 5, 2, 1]:
    #    print(f"LOF ({i}): \n ", json.dumps(lof(tr, te, y, i), indent=1))
    #print("OC-SVM: \n ", json.dumps(oc_svm(tr, te, y)))

    dnn(tr, te, y)