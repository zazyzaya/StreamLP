from argparse import ArgumentParser
import glob 
from math import ceil

from joblib import Parallel, delayed
import torch 
from tqdm import tqdm 

from models.optimized_tgb import BatchTGBase
from models.mlp import BinaryClassifier

NUM_NODES = 29985
NUM_ETYPES = 49 + 3 
RAW = '/home/ead/iking5/code/process_lanl/torch_files/'
TG = '/home/ead/iking5/code/StreamLP/lanl-data/'

GRANULARITY = 60*15

def build_data_indi():
    all_files = glob.glob(RAW + '*.pt')
    n_files = len(all_files)

    batches_per_file = ceil(3600 / GRANULARITY)

    def build_single(idx):
        tgb = BatchTGBase(NUM_ETYPES, NUM_NODES)

        g = torch.load(RAW+str(idx)+'.pt')
        st = g.ts[0].item()
        en = st + GRANULARITY

        for _ in tqdm(range(batches_per_file), desc=f'{idx+1}/{n_files}'):
            mask = (g.ts >= st).logical_and(g.ts < en)
            ei = g.edge_index[:, mask]
            ts = g.ts[mask]
            ew = g.edge_attr[mask]

            tgb.add_batch(
                ei, ts, ew, 
            )

            st = en 
            en += GRANULARITY

        z = tgb.get()
        del tgb 
        return z
    
    node_embs = Parallel(n_jobs=16, prefer='processes', )(
        delayed(build_single)(i) for i in range(64)
    )

    torch.save(torch.stack(node_embs), TG+'tgbase_embeddings_indi.pt')



def build_data():
    tgb = BatchTGBase(NUM_ETYPES, NUM_NODES)

    all_files = glob.glob(RAW + '*.pt')
    n_files = len(all_files)

    batches_per_file = ceil(3600 / GRANULARITY)
    node_embs = []
    prog = tqdm(total=n_files*(batches_per_file+1))

    for i in range(n_files):
        prog.update()
        g = torch.load(RAW+str(i)+'.pt')
        st = g.ts[0].item()
        en = st + GRANULARITY

        for batch in range(batches_per_file):
            prog.desc = f'{i}: ({batch}/{batches_per_file})'
            prog.update() 
            
            mask = (g.ts >= st).logical_and(g.ts < en)
            ei = g.edge_index[:, mask]
            ts = g.ts[mask]
            ew = g.edge_attr[mask]

            z = tgb.add_batch(
                ei, ts, ew, 
                return_value=True
            )
            node_embs.append(z)

            st = en 
            en += GRANULARITY

        prog.desc = f'Loading file {i+1}'
        torch.save(torch.stack(node_embs), TG+'tgbase_embeddings.pt')

    torch.save(torch.stack(node_embs), TG+'tgbase_embeddings.pt')

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('-i', '--indipendant', action='store_true')
    args = ap.parse_args()

    if args.indipendant:
        build_data_indi()
    else:
        build_data()