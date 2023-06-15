from argparse import ArgumentParser
from copy import deepcopy
import math 
import os 
from types import SimpleNamespace

import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import torch 
from torch.optim import Adam 
from tqdm import tqdm 

from models.mlp import BinaryClassifier
from models.tgbase import mask_feat

torch.set_num_threads(16)
DATA_HOME = os.path.dirname(__file__) + '/mixer-datasets/precalculated/tgbase/processed/'
HYPERPARAMS = SimpleNamespace(
    epochs=10, bs=100, patience=float('inf'), lr=3e-4
)
        
def train(hp, model, tr_x, tr_y, va_x, va_y, te_x, te_y):
    opt = Adam(model.parameters(), lr=hp.lr)

    best = dict(epoch=-1, auc=0, sd=dict())
    for e in range(hp.epochs):
        for batch in tqdm(range(math.ceil(tr_x.size(0)/hp.bs))):
            bx = tr_x[batch*hp.bs : (batch+1)*hp.bs]
            by = tr_y[batch*hp.bs : (batch+1)*hp.bs]
            
            model.train()
            opt.zero_grad()
            loss = model(bx, by)
            loss.backward()
            opt.step()

        val = eval(model, va_x, va_y)
        print(f"[{e}] loss: {loss.item():0.4f}\n\tVal AUC: {val:0.4f}")
        print(f"\tTest AUC {eval(model, te_x, te_y):0.4f}")

        if val > best['auc']: # type: ignore
            best = dict(epoch=e, auc=val, sd=deepcopy(model.state_dict()))
        if best['epoch'] < e-hp.patience: 
            break 

    return best 

@torch.no_grad()
def eval(model, x,y):
    model.eval()
    y_hat = model.pred(x)
    return roc_auc_score(y, y_hat)

def missing_feature_importance_test(hp, tr_x, tr_y, va_x, va_y, te_x, te_y, no_val=False):
    # Scale data (same as in paper's code though doesn't seem to matter)
    mm = MinMaxScaler()
    tr_x = torch.from_numpy(mm.fit_transform(tr_x)).float()
    va_x = torch.from_numpy(mm.transform(va_x)).float()
    te_x = torch.from_numpy(mm.transform(te_x)).float()

    # Bias to care more about rare 1-labeled rows
    weight0 = tr_y.sum()/tr_y.size(0)
    weight1 = 1-weight0
    weights = torch.tensor([weight1])

    stats = dict()

    # Control group: do one with no missing features:
    aucs = []
    for _ in range(10):
        model = BinaryClassifier(tr_x.size(1), class_weights=weights)
        best = train(
            hp, model, 
            tr_x, tr_y, 
            va_x, va_y, 
            te_x, te_y
        )
        if not no_val:
            model.load_state_dict(best['sd'])
        aucs.append(eval(model, te_x, te_y))
    stats['nothing'] = aucs

    feats = ['times', 'entropy', 'local', 'structural', 'sum', 'mean', 'max', 'min', 'std']
    for f_str in feats:
        print(f_str)

        mask = ~mask_feat([f_str], tr_x.size(1))
        aucs = []

        for _ in range(10):
            model = BinaryClassifier(tr_x[:,mask].size(1), class_weights=weights)
            best = train(
                hp, model, 
                tr_x[:,mask], tr_y, 
                va_x[:,mask], va_y, 
                te_x[:,mask], te_y
            )
            if not no_val:
                model.load_state_dict(best['sd'])
            aucs.append(eval(model, te_x[:,mask], te_y))

        stats[f_str] = aucs

    for i in range(len(feats)):
        for j in range(i+1,len(feats)):
            # Every unique combo (where (a,b) == (b,a) so is not unique)
            f1 = feats[i]
            f2 = feats[j]
            f_str = f1+'-'+f2
            print(f_str)

            mask = ~mask_feat([f1,f2], tr_x.size(1))
            aucs = []

            for _ in range(10):
                model = BinaryClassifier(tr_x[:,mask].size(1), class_weights=weights)
                best = train(
                    hp, model, 
                    tr_x[:,mask], tr_y, 
                    va_x[:,mask], va_y, 
                    te_x[:,mask], te_y
                )
                if not no_val:
                    model.load_state_dict(best['sd'])
                aucs.append(eval(model, te_x[:,mask], te_y))

            stats[f_str] = aucs  

    df = pd.DataFrame(stats)
    out_f = 'results/tgstream/missing_feat_test.csv'
    if no_val:
        out_f = out_f.replace('.csv', '_no_val.csv')
    with open(out_f, 'a') as f:
        f.write(df.to_csv())
        f.write(df.mean().to_csv())
        f.write(df.sem().to_csv())

def single_feature_importance_test(hp, tr_x, tr_y, va_x, va_y, te_x, te_y, no_val=False):
    # Scale data (same as in paper's code though doesn't seem to matter)
    mm = MinMaxScaler()
    tr_x = torch.from_numpy(mm.fit_transform(tr_x)).float()
    va_x = torch.from_numpy(mm.transform(va_x)).float()
    te_x = torch.from_numpy(mm.transform(te_x)).float()

    # Bias to care more about rare 1-labeled rows
    weight0 = tr_y.sum()/tr_y.size(0)
    weight1 = 1-weight0
    weights = torch.tensor([weight1])

    stats = dict()

    feats = ['entropy', 'sum', 'mean', 'max', 'min', 'std']
    for f_str in feats:
        print(f_str)

        mask = mask_feat([f_str], tr_x.size(1))
        aucs = []

        for _ in range(10):
            model = BinaryClassifier(tr_x[:,mask].size(1), class_weights=weights)
            best = train(
                hp, model, 
                tr_x[:,mask], tr_y, 
                va_x[:,mask], va_y, 
                te_x[:,mask], te_y
            )
            if not no_val:
                model.load_state_dict(best['sd'])
            aucs.append(eval(model, te_x[:,mask], te_y))

        stats[f_str] = aucs

    df = pd.DataFrame(stats)
    out_f = 'results/tgstream/single_feat_test.csv'
    if no_val:
        out_f = out_f.replace('.csv', '_no_val.csv')

    with open(out_f, 'a') as f:
        f.write(df.to_csv())
        f.write(df.mean().to_csv())
        f.write(df.sem().to_csv())

def random_forest(fname):
    fname = DATA_HOME + fname + '.pt'
    x,y = torch.load(fname)
    y = y.float()

    # Chronological split as in the paper
    # 85 : 15
    tr = int(x.size(0)*0.85)

    tr_x = x[:tr];   tr_y = y[:tr]
    te_x = x[tr:];   te_y = y[tr:]

    aucs = []
    for i in range(10):
        print(f"Fitting ({i+1}/10)...", end='\t', flush=True)
        rf = RandomForestClassifier(n_jobs=32)
        rf.fit(tr_x, tr_y)
        preds = rf.predict_proba(te_x)[:,1]

        auc = roc_auc_score(te_y, preds)
        print(auc, '\n')
        aucs.append(auc)

    df = pd.DataFrame({'auc':aucs})
    print(df.mean())
    print(df.sem())


def main(hp, fname, no_val=False):
    fname = DATA_HOME + fname + '.pt'

    x,y = torch.load(fname)
    y = y.float().unsqueeze(-1)

    # Chronological split as in the paper
    # 70 : 15 : 15
    tr = int(x.size(0)*0.70)
    va = int(x.size(0)*0.85)

    if not no_val:
        tr_x = x[:tr];   tr_y = y[:tr]
        va_x = x[tr:va]; va_y = y[tr:va]
        te_x = x[va:];   te_y = y[va:]
    else:
        tr_x = x[:va];   tr_y = y[:va]
        va_x = x[tr:va]; va_y = y[tr:va]
        te_x = x[va:];   te_y = y[va:]

    #missing_feature_importance_test(hp, tr_x, tr_y, va_x, va_y, te_x, te_y, no_val=no_val)
    #single_feature_importance_test(hp, tr_x, tr_y, va_x, va_y, te_x, te_y, no_val=no_val)
    # Bias to care more about rare 1-labeled rows
    weight0 = tr_y.sum()/tr_y.size(0)
    weight1 = 1-weight0
    weights = torch.tensor([weight1])

    stats = dict()
    aucs = []
    for _ in range(10):
        model = BinaryClassifier(tr_x.size(1), class_weights=weights)
        best = train(
            hp, model, 
            tr_x, tr_y, 
            va_x, va_y, 
            te_x, te_y
        )
        if not no_val:
            model.load_state_dict(best['sd'])
        aucs.append(eval(model, te_x, te_y))
    
    stats['nothing'] = aucs
    df = pd.DataFrame(stats)
    out_f = 'results/tgstream/tbgase.csv'
    if no_val:
        out_f = out_f.replace('.csv', '_no_val.csv')

    with open(out_f, 'a') as f:
        f.write(str(hp)+'\n')
        f.write(df.to_csv())
        f.write(df.mean().to_csv())
        f.write(df.sem().to_csv())

if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-d', '--dataset', default='wikipedia')
    args.add_argument('--rf', action='store_true')
    args.add_argument('--no-val', action='store_true')
    args = args.parse_args()

    if args.rf:
        random_forest(args.dataset)
    else:
        main(HYPERPARAMS, args.dataset, no_val=args.no_val)