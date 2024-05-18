import torch

def cat(l):
    return torch.cat(l, axis=1)
def shuffle(data):
    return data[torch.randperm(len(data))]
def standardize(data):
    return (data-data.mean(0))/(data.std(0)+1e-9)
def normalize(data):
    if len(data.shape)>1 and data.shape[1]>1:
        return torch.nn.functional.normalize(data)
    else:
        return data
def shuffle_keeping_class(data, labels):
    classes=torch.unique(labels)
    shuffled=torch.zeros_like(data)
    for c in classes:
        idx=torch.where(labels==c)[0]
        shuffled[idx]=data[idx][torch.randperm(len(idx))]
    return shuffled
