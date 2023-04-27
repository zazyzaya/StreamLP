import os 

import torch 
from tqdm import tqdm 

from models.tgbase import StreamTGBase

def preprocess(fname, force=False):
    '''
    Assumes file is formatted as the reddit/wiki ones are 
    just rows of edges in a csv with columns
    src,dst,ts,y,feat
    '''
    stem = os.path.dirname(fname)
    basename = os.path.basename(fname)
    outf = stem + '/precalculated/tgbase/raw/' + basename.replace('.csv', '.pt')

    if os.path.exists(outf) or force:
        return torch.load(outf)

    f = open(fname, 'r')

    # skip header 
    f.readline()
    line = f.readline()

    features = []
    ei = []
    ys = []

    prog = tqdm(desc='converting from string')
    while line:
        src,dst,ts,y,tokens = line.strip().split(',', 4)
        feat = [float(f) for f in tokens.split(',')]
        feat += [float(ts)]

        ei.append([int(src),int(dst)])
        features.append(feat)
        ys.append(int(y))

        line = f.readline()
        prog.update()
    
    ret = torch.tensor(ei), torch.tensor(features), torch.tensor(ys)
    torch.save(ret, outf)

    return ret 

@torch.no_grad()
def build_dataset(fname):
    stem = os.path.dirname(fname)
    basename = os.path.basename(fname)
    outf = stem + '/precalculated/tgbase/processed/' + basename.replace('.csv', '.pt')

    ei, feats, _ = preprocess(fname)
    tg = StreamTGBase(feats.size(1), n_nodes=10_000, entropy=False)

    rows = []
    for i in tqdm(range(feats.size(0))):
        feat = tg.add_edge(
            ei[i],
            feats[i],
            return_value=True
        )

        rows.append(feat)

    torch.save(torch.stack(rows), outf)
        

if __name__ == '__main__':
    build_dataset('mixer-datasets/wikipedia.csv')