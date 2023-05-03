import os 

import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
import torch  

from models.mlp import BinaryClassifier
from tgstream import train, eval, HYPERPARAMS

torch.set_num_threads(16)
HOME = os.path.dirname(__file__) + '/mixer-datasets/precalculated/tgbase/raw/'

def mlp(dataset):
    # Just look at each edge's feature data with no 
    # relational stuff at all. Every feature is totally
    # isolated
    _,x,_,y = torch.load(HOME+dataset + '.pt')
    y = y.float().unsqueeze(-1)

    # Chronological split as in the paper
    # 70 : 15 : 15
    tr = int(x.size(0)*0.70)
    va = int(x.size(0)*0.85)

    tr_x = x[:tr];   tr_y = y[:tr]
    va_x = x[tr:va]; va_y = y[tr:va]
    te_x = x[va:];   te_y = y[va:]

    # Bias to care more about rare 1-labeled rows
    weight0 = tr_y.sum()/tr_y.size(0)
    weight1 = 1-weight0
    weights = torch.tensor([weight1])

    # Normalize features 
    mm = MinMaxScaler()
    tr_x = torch.from_numpy(mm.fit_transform(tr_x)).float()
    va_x = torch.from_numpy(mm.transform(va_x)).float()
    te_x = torch.from_numpy(mm.transform(te_x)).float()

    aucs = []
    for _ in range(10):
        model = BinaryClassifier(x.size(1), class_weights=weights)
        best = train(HYPERPARAMS, model, tr_x, tr_y, va_x, va_y, te_x, te_y)
        model.load_state_dict(best['sd'])
        aucs.append(eval(model, te_x, te_y))
        print(aucs[-1])

    df = pd.DataFrame({'auc': aucs})

    print(df.mean())
    print(df.sem())

if __name__ == '__main__':
    mlp('reddit')