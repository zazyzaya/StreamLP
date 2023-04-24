from copy import deepcopy
import os 
import time 
from types import SimpleNamespace

import numpy as np 
from sklearn.metrics import roc_auc_score as auc_score
import torch 
from torch.optim import Adam
from torch_geometric.data import Data

from databuilders.build_ctdne_data import load_ctdne, CTDNE_FNAMES
from databuilders.build_mixer_data import get_dataset, MixerCSR
from models.dynamic import FlowGNN_LP

torch.set_num_threads(8)
HYPERPARAMS = SimpleNamespace(
    hidden=256, out=100, layers=3, decoder='deephad',
    lr=0.01, epochs=200, batch_size=64
)

def train(hp, model, data):
    opt = Adam(model.parameters(), lr=hp.lr)

    for e in range(hp.epochs):
        #sample = torch.randperm(data.tr_edge_index.size(1))
        ei = data.tr_edge_index #[:, sample[:hp.batch_size]]

        st = time.time()
        model.train()
        opt.zero_grad()
        loss = model(data, ei)
        loss.backward()
        opt.step()

        print("[%d] Loss: %0.03f  (%0.2f)" % (e, loss.item(), time.time()-st))

        if (e+1) % 10 == 0:
            test(model, data, verbose=True)

    return model 

@torch.no_grad()
def test(model, data, verbose=True):
    pos = data.te_edge_index
    neg = torch.randint(0, data.te_edge_index.max(), pos.size())

    pos = model.lp(data, pos)
    neg = model.lp(data, neg)

    preds = torch.cat([pos, neg], dim=0)
    targets = torch.zeros(preds.size())
    targets[:pos.size(0)] = 1. 

    auc = auc_score(targets, preds)
    if verbose:
        print("AUC: %0.4f" % auc)

    return auc


def main(hp, dataset, loader, trials=5):
    data = loader(dataset, force=True)

    if loader == get_dataset:
        csr, ei, ts = data 
        data = Data(
            x = csr.node_feats, 
            edge_index = ei, 
            csr_ei = csr 
        )

    stats = []
    for _ in range(trials):
        print(dataset)
        model = FlowGNN_LP(
            data.x.size(1), hp.hidden, hp.out, 
            layers=hp.layers, dec=hp.decoder
        )

        train(hp, model, data)
        stats.append(test(model, data))

    stats = np.array(stats)
    mean = np.mean(stats)
    sem = np.std(stats) / (trials ** 0.5)

    out_f = 'results/%s.csv' % dataset
    
    # Write header if need be
    if not os.path.exists(out_f):
        with open(out_f, 'w+') as f:
            f.write('hidden,out,layers,decoder,mean,stderr\n')

    with open(out_f, 'a') as f:
        out_str = f'{hp.hidden},{hp.out},{hp.layers},{hp.decoder},{mean},{sem}\n'
        f.write(out_str)

    print(mean,'(+/-)',sem)

if __name__ == '__main__':
    for dec in ['deephad', 'had', 'dot', 'l2']:
        for layers in [2,3]:
            hp = deepcopy(HYPERPARAMS)
            hp.decoder = dec 
            hp.layers = layers 

            for data in ['wikipedia', 'reddit']:
                main(hp, data, get_dataset)