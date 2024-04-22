import torch
import numpy as np

def estimate_id(X, algorithm='twoNN', k=100, fraction=0.9, full_output=False):
    if algorithm=='twoNN':
        return twoNN(X, fraction)
    elif algorithm=='MLE':
        return MLE(X, k, full_output)

def MLE(X, k=100, full_output=False):
    X=X.float()
    X=torch.cdist(X,X)
    Y=torch.topk(X, k+1, dim=1, largest=False)[0][:,1:]
    Y=torch.log(torch.reciprocal(torch.div(Y, Y[:,-1].reshape(-1,1))))
    dim=torch.reciprocal(1/(k-1)*torch.sum(Y, dim=1))
    return dim if full_output else dim.mean()

def twoNN(X,fraction=0.9,distances=False):
    if not distances:
        X=torch.cdist(X,X)
    Y=torch.topk(X, 3, dim=1, largest=False)[0]
    # clean data
    k1 = Y[:,1]
    k2 = Y[:,2]
    #remove zeros and degeneracies (k1==k2)
    old_k1=k1
    k1 = k1[old_k1!=0]
    k2 = k2[old_k1!=0]
    old_k1=k1
    k1 = k1[old_k1!=k2] 
    k2 = k2[old_k1!=k2]
    # n.of points to consider for the linear regression
    npoints = int(np.floor(len(k1)*fraction))
    # define mu and Femp
    N = len(k1)
    mu,_ = torch.sort(torch.divide(k2, k1).flatten())
    Femp = (torch.arange(1,N+1,dtype=X.dtype))/N
    # take logs (leave out the last element because 1-Femp is zero there)
    x = torch.log(mu[:-1])[0:npoints]
    y = -torch.log(1 - Femp[:-1])[0:npoints]
    # regression, on gpu if available
    y=y.to(x.device)
    slope=torch.linalg.lstsq(x.unsqueeze(-1),y.unsqueeze(-1))
    return slope.solution.squeeze()
