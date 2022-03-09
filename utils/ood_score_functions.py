import torch
from .utils import prediction_entropy

ood_score_functions = {
    "MSP":
        {"single": lambda l: 1 - l.softmax(dim=1).max(dim=1)[0].detach(),
         "ens": lambda ls: 1 - torch.stack([l.softmax(1) for l in ls], dim=0).mean(dim=0).max(dim=1)[0]},
    "H":
        {"single": lambda l: prediction_entropy(l.softmax(1)),
         "ens": lambda ls: prediction_entropy(torch.stack([l.softmax(1) for l in ls], dim=0).mean(dim=0))},
    "ML":
        {"single": lambda l: 1 - l.max(dim=1)[0].detach(),
         "ens": lambda ls: 1 - torch.stack(ls, dim=0).mean(dim=0).max(dim=1)[0]}
}

