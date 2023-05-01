import torch 
from torch import nn 

class BinaryClassifier(nn.Module):
    def __init__(self, in_dim, hidden=[80,10], layers=3, drop=0.1, class_weights=None) -> None:
        '''
        Same default params as TGBase paper
        '''
        super().__init__()

        def lin_constructor(in_d, out_d):
            return nn.Sequential(
                nn.Linear(in_d, out_d),
                nn.Dropout(drop),
                nn.ReLU()
            )
        
        self.net = nn.Sequential(
            lin_constructor(in_dim, hidden[0]),
            *[
                lin_constructor(hidden[i], hidden[i+1])
                for i in range(layers-2)
            ],
            nn.Linear(hidden[-1], 1)
        )

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    def forward(self, x, labels):
        logits = self.net(x)

        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)

        return self.loss_fn(logits, labels)
    
    @torch.no_grad()
    def pred(self, x):
        logits = self.net(x)
        return torch.sigmoid(logits)