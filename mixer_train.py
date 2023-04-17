from copy import deepcopy
from types import SimpleNamespace

from sklearn.metrics import average_precision_score as ap_score
import torch 
from torch.optim import Adam 
from tqdm import tqdm 

from databuilders.build_mixer_data import get_dataset, MixerCSR
from models.graph_mixer import GraphMixer_LP

torch.set_num_threads(8)
DEVICE = 3
HYPERPARAMS = SimpleNamespace(
    lr=0.0001, wd=1e-6, batch_size=600, T=2000, # Same as paper 
    epochs=200, patience=20,     # from their github repo
)

# From paper Appendix A
k_map = {
    'reddit': 10, 
    'lastfm': 10,
    'mooc': 20,
    'wikipedia': 30,
    'gdelt': 30
}

def sample_node_feats(ei, ts, t0, st, en):
    pass # TODO 

def train(hp, model, csr, ei, ts):
    tr_ei = ei[:, ts<csr.end_tr]

    n_batches = tr_ei.size(1) // hp.batch_size
    n_batches = n_batches + 1 if tr_ei.size(1) % hp.batch_size else n_batches
    #perm = torch.randperm(tr_ei.size(1))

    best = {'epoch': -1, 'val_ap': 0}
    stats = []
    opt = Adam(model.parameters(), lr=hp.lr, weight_decay=hp.wd)
    for e in range(hp.epochs):
        model.train()

        for b in tqdm(range(1,n_batches)):
            st = b*hp.batch_size 
            en = st+hp.batch_size 
            target = tr_ei[:, st:en]
            t = ts[st]
            
            opt.zero_grad()
            last_T = ei[:, max(0,st-hp.T):st].to(DEVICE)
            loss = model(csr, target, t, last_T)
            loss.backward()
            opt.step()

            del last_T

        print(f"[{e}] {loss.item():.4f}")
        tr_ap, val_ap = validate(model, csr, ei, ts)
        te_ap  = test(model, csr, ei, ts)

        stats.append([tr_ap, val_ap, te_ap])

        if val_ap > best['val_ap']:
            best = dict(
                epoch = e, 
                val_ap = val_ap, 
                te_ap = te_ap, 
                sd = deepcopy(model.state_dict())
            )

        if e > best['epoch']+hp.patience: 
            break 

    return torch.tensor(stats).T, best

@torch.no_grad()
def validate(model, csr, ei, ts):
    model.eval()
    tr_ei = ei[:, (ts < csr.end_tr)]
    last_T = tr_ei[:, -2000:].to(DEVICE)
    preds, targets = model(csr, tr_ei, csr.end_tr, last_T, pred=True)

    preds = preds.cpu()
    targets = targets.cpu()
    tr_ap = ap_score(targets, preds)
    print("\tTr  AP: ", tr_ap)

    va_ei = ei[:, (ts < csr.end_va).logical_and(ts > csr.end_tr)]
    preds, targets = model(csr, va_ei, csr.end_tr, last_T, pred=True)

    preds = preds.cpu()
    targets = targets.cpu()
    va_ap = ap_score(targets, preds)
    print("\tVal AP: ", va_ap)
    return tr_ap, va_ap 

@torch.no_grad()
def test(model, csr, ei, ts):
    model.eval()
    te_ei = ei[:, (ts > csr.start_te)]
    start_T = ei[:, (ts < csr.start_te)][:, -2000:].to(DEVICE)
    preds, targets = model(csr, te_ei, csr.end_va, start_T, pred=True)

    preds = preds.cpu()
    targets = targets.cpu()
    ap = ap_score(targets, preds)
    print("\tTe  AP: ", ap)
    return ap 

def main(dataset, i):
    csr, ei, ts = get_dataset(dataset)

    # No need to eval on both direction of edges(?)
    ei = ei[:, :ei.size(1) // 2]
    ts = ts[:ei.size(1)]

    if csr.node_feats.size(1) == 1:
        csr.node_feats = torch.eye(csr.num_nodes)

    csr.to(DEVICE)
    model = GraphMixer_LP(
        csr.efeats.size(1), 
        csr.node_feats.size(1), 
        device=DEVICE,
        K=k_map[dataset]
    )

    stats, best = train(HYPERPARAMS, model, csr, ei, ts)
    torch.save(
        dict(stats=stats, best=best), 
        f'results/mixer/{dataset}_i.pt'
    )

[main('wikipedia', i) for i in range(5)]
[main('reddit', i) for i in range(5)]