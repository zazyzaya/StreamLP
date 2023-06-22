import glob 

import torch 

from models.optimized_tgb import TGBase
from models.mlp import BinaryClassifier

NUM_NODES = 29985
RAW = '/home/ead/iking5/process_lanl/torch_files/'
TG = '/home/ead/iking5/StreamLP/lanl-data/'

def build_data():
    tgb = TGBase()

    all_files = glob.glob(RAW + '*.pt')
    n_files = len(all_files)

    node_embs = []
    for i in range(n_files):
        g = torch.load(RAW+str(i)+'.pt')
        z = tgb.forward(g.edge_index, g.ts, g.edge_attr, )