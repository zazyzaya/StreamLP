from argparse import ArgumentParser
import glob 
from math import ceil

from joblib import Parallel, delayed
import torch 
from tqdm import tqdm 

from models.optimized_tgb import BatchTGBase
from models.mlp import BinaryClassifier

NUM_NODES = 29980
QUANT = 3
NUM_ETYPES = 49 + QUANT
RAW = '/mnt/raid1_ssd_4tb/datasets/LANL15/torch_data/'
TG = '/home/ead/iking5/code/StreamLP/lanl-data/'

GRANULARITY = 60*15

def build_data_indi():
    all_files = glob.glob(RAW + '*.pt')
    n_files = len(all_files)

    batches_per_file = ceil(3600 / GRANULARITY)

    def build_single(idx):
        tgb = BatchTGBase(NUM_ETYPES, NUM_NODES)

        try: 
            g = torch.load(RAW+str(idx)+'.pt')
        except FileNotFoundError: 
            print(f"No file found for {idx}.pt")
            torch.save(tgb.get(), f'{TG}indipendant/{idx}.pt')
            return 

        st = g.ts[0].item()
        en = st + GRANULARITY
        
        # Forgot to do this in preproc
        g.edge_attr[:, -QUANT:] = torch.log(g.edge_attr[:, -QUANT:])

        for _ in tqdm(range(batches_per_file), desc=f'{idx+1}/{n_files}'):
            mask = (g.ts >= st).logical_and(g.ts < en)
            
            # Make sure this time step has activity
            if mask.sum() > 0:
                ei = g.edge_index[:, mask]
                ts = g.ts[mask] / (60*60*24)
                ew = g.edge_attr[mask]

                tgb.add_batch(
                    ei, ts, ew, 
                )
            else:
                print(f'{idx} from {st}-{en} has no edges')

            st = en 
            en += GRANULARITY

        torch.save(tgb.get(), f'{TG}indipendant/{idx}.pt')

    Parallel(n_jobs=16, prefer='processes', )(
        delayed(build_single)(i) for i in range(n_files)
    )

def build_data():
    tgb = BatchTGBase(NUM_ETYPES, NUM_NODES)

    all_files = glob.glob(RAW + '*.pt')
    n_files = len(all_files)

    batches_per_file = ceil(3600 / GRANULARITY)
    prog = tqdm(total=n_files*(batches_per_file+1))

    for i in range(n_files):
        prog.update()
        g = torch.load(RAW+str(i)+'.pt')
        g.edge_attr[:, -QUANT:] = torch.log10(g.edge_attr[:, -QUANT:]+10)

        st = g.ts[0].item()
        en = st + GRANULARITY

        z = None
        for batch in range(batches_per_file):
            prog.desc = f'{i}: ({batch}/{batches_per_file})'
            prog.update() 
            
            mask = (g.ts >= st).logical_and(g.ts < en)
            ei = g.edge_index[:, mask]
            ts = g.ts[mask] / (60*60*24)
            ew = g.edge_attr[mask]

            z = tgb.add_batch(
                ei, ts, ew, 
                return_value=True
            )

            st = en 
            en += GRANULARITY

        prog.desc = f'Loading file {i+1}'
        torch.save(z, f'{TG}/continuous/{i}.pt')


if __name__ == '__main__':
    build_data_indi()
    build_data()