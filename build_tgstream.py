from argparse import ArgumentParser
import os 
from types import SimpleNamespace

import torch 
from tqdm import tqdm 

from models.tgbase import StreamTGBase

RAW_HOME = os.path.dirname(__file__) + '/mixer-datasets/'
DATA_HOME = os.path.dirname(__file__) + '/mixer-datasets/precalculated/tgbase/'
HYPERPARAMS = SimpleNamespace(
    epochs=10, bs=100, patience=float('inf'), lr=3e-4
)

def preprocess(fname, force=False):
    '''
    Assumes file is formatted as the reddit/wiki ones are 
    just rows of edges in a csv with columns
    src,dst,ts,y,feat
    '''
    outf = DATA_HOME + 'raw/' + fname + '.pt'
    fname = RAW_HOME + fname + '.csv'

    if os.path.exists(outf) and not force:
        return torch.load(outf)

    f = open(fname, 'r')

    # skip header 
    f.readline()
    line = f.readline()

    features = []
    ts = []
    ei = []
    ys = []

    prog = tqdm(desc='converting from string')
    while line:
        src,dst,t,y,tokens = line.strip().split(',', 4)
        feat = [float(f) for f in tokens.split(',')]

        ei.append([int(src),int(dst)])
        features.append(feat)
        ts.append(float(t))
        ys.append(int(y))

        line = f.readline()
        prog.update()
    
    ret = torch.tensor(ei), torch.tensor(features), torch.tensor(ts), torch.tensor(ys)
    torch.save(ret, outf)

    return ret 

@torch.no_grad()
def build_dataset(fname, no_h=True, force=False, force_tg=False, suffix=''):
    outf = DATA_HOME + 'processed/' + fname + suffix + '.pt'

    if os.path.exists(outf) and not force and not force_tg:
        return torch.load(outf)

    ei, feats, ts, ys = preprocess(fname, force=force)
    tg = StreamTGBase(feats.size(1), n_nodes=10_000, entropy=not no_h)

    rows = []
    for i in tqdm(range(feats.size(0))):
        feat = tg.add_edge(
            ei[i],
            ts[i],
            feats[i],
            return_value=True
        )

        rows.append(feat)

    if no_h:
        outf = outf.replace('.pt', '_no_h.pt')

    xs = torch.stack(rows)
    torch.save((xs,ys), outf)
    return xs, ys


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-d', '--dataset', default='wikipedia')
    args.add_argument('-e', '--entropy', action='store_false')
    args.add_argument('-f','--force', action='store_true')
    args.add_argument('-t','--force-tg', action='store_true')
    args.add_argument('-s', '--suffix', default='')

    args = args.parse_args()
    print(args)

    build_dataset(
        args.dataset,
        no_h=args.entropy,
        force=args.force,
        force_tg=args.force_tg,
        suffix=args.suffix
    )
