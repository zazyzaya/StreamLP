from types import SimpleNamespace

from sklearn.metrics import average_precision_score as ap_score
import torch 
from torch.optim import Adam 
from tqdm import tqdm 

from databuilders.build_mixer_data import get_dataset, MixerCSR
from models.graph_mixer import GraphMixer_LP

torch.set_num_threads(16)
HYPERPARAMS = SimpleNamespace(
    lr=0.0001, wd=1e-6, batch_size=6000,  # Same as paper 
    epochs=200                  # from their github repo
)

# From paper Appendix A
k_map = {
    'reddit': 10, 
    'lastfm': 10,
    'mooc': 20,
    'wikipedia': 30,
    'gdelt': 30
}

def train(hp, model, csr, ei, ts):
    tr_ei = ei[:, ts<csr.end_tr] 

    n_batches = tr_ei.size(1) // hp.batch_size
    n_batches = n_batches + 1 if tr_ei.size(1) % hp.batch_size else n_batches
    perm = torch.randperm(tr_ei.size(1))

    opt = Adam(model.parameters(), lr=hp.lr, weight_decay=hp.wd)
    for e in range(hp.epochs):
        opt.zero_grad()
        for b in tqdm(range(n_batches)):
            st = b*hp.batch_size 
            en = st+hp.batch_size 
            target = tr_ei[:, perm[st:en]]

            #opt.zero_grad()
            loss = model(csr, target, csr.end_tr)
            loss.backward()
            #opt.step()

            #print(f"[{e}-{b}] {loss.item():.4f}")
        opt.step()
        validate(model, csr, ei, ts)

@torch.no_grad()
def validate(model, csr, ei, ts):
    va_ei = ei[:, (ts < csr.end_va).logical_and(ts > csr.end_tr)]
    preds, targets = model(csr, va_ei, csr.end_va, pred=True)

    print("\tVal AP: ", ap_score(targets, preds))

dataset = 'reddit'
csr, ei, ts = get_dataset(dataset)
csr.load_train_feats()
model = GraphMixer_LP(csr.efeats.size(1), csr.node_feats.size(1))

train(HYPERPARAMS, model, csr, ei, ts)
