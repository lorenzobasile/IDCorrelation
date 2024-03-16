import torch
import numpy as np
from scipy import stats
from time import time

def cat(l):
    return torch.cat(l, axis=1)
def shuffle(data):
    return data[torch.randperm(len(data))]

def estimate_id(X, algorithm='twoNN', k=100, fraction=0.9):
    if algorithm=='twoNN':
        return twoNN(X, fraction)
    elif algorithm=='MLE':
        return MLE(X, k)

def MLE(X, k=100):
    X=torch.cdist(X,X)
    Y=torch.topk(X, k+1, dim=1, largest=False)[0]
    Y=torch.log(torch.reciprocal(torch.div(Y, Y[:,-1].reshape(-1,1))[:,1:]))
    dim=torch.reciprocal(1/(k-1)*torch.sum(Y, dim=1))
    return dim.mean()

def twoNN(X,fraction=0.9):
    X=X.double()    
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
    Femp = (torch.arange(1,N+1,dtype=torch.float64))/N
    # take logs (leave out the last element because 1-Femp is zero there)
    x = torch.log(mu[:-1])[0:npoints]
    y = -torch.log(1 - Femp[:-1])[0:npoints]
    # regression, on gpu if available
    y=y.to(x.device)
    slope=torch.linalg.lstsq(x.unsqueeze(-1),y.unsqueeze(-1))
    return slope.solution.squeeze()

def id_correlation(dataset1, dataset2, N=100, algorithm='twoNN'):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    t0=time()
    id0=estimate_id(cat([dataset1, dataset2]).to(device), algorithm).item()
    shuffled_id=torch.zeros(N, dtype=torch.double)
    for i in range(N):
        shuffled_id[i]=estimate_id(cat([dataset1, shuffle(dataset2)]).to(device), algorithm).item()
    id_shuffled=shuffled_id.mean()
    std_shuffled=shuffled_id.std()
    Z=(id0-id_shuffled)/std_shuffled
    p=((shuffled_id<id0).sum()+1)/(N+1) #according to permutation test, not Z-test
    return {'Z': Z, 'p':p,  'original Id': id0, 'Mean shuffled Id': id_shuffled, 'Std shuffled Id': std_shuffled, 'time': time()-t0}

def ks_test(tensor1, tensor2):
    data_all = torch.cat([tensor1, tensor2], dim=0)
    cdf1 = torch.searchsorted(tensor1, data_all, side='right') / len(tensor1)
    cdf2 = torch.searchsorted(tensor2, data_all, side='right') / len(tensor2)
    cddiffs = torch.abs(cdf1 - cdf2)
    score = torch.max(cddiffs).item()
    en= len(tensor1) * len(tensor2) / (len(tensor1) + len(tensor2))
    return score, stats.distributions.kstwo.sf(score, np.round(en))

def ks_correlation(dataset1, dataset2, allpairs=False, N=10):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    t0=time()
    d0=torch.nn.functional.pdist(cat([dataset1, dataset2]).to(device)).sort()[0].cpu()
    shuffled=torch.zeros(N,len(d0))
    KS_orig=[]
    KS_shuf=[]
    for i in range(N):
        shuffled[i]=torch.nn.functional.pdist(cat([dataset1, shuffle(dataset2)]).to(device)).sort()[0].cpu()
        KS_orig.append(ks_test(d0, shuffled[i])[0])
    # this scales with N^2
    if allpairs:
        for i in range(N-1):
            for j in range(i+1,N):
                KS_shuf.append(ks_test(shuffled[i].to(device), shuffled[j].to(device))[0])
    else:
    #usually it's sufficient to take N shuffled pairs
        for _ in range(N):
            idx=np.random.choice(N, 2, replace=False)
            i=idx[0]
            j=idx[1]
            KS_shuf.append(ks_test(shuffled[i].to(device), shuffled[j].to(device))[0])
    KS_orig=np.array(KS_orig)
    KS_shuf=np.array(KS_shuf)
    res=stats.kstest(KS_orig, KS_shuf)
    score=res.statistic
    p=res.pvalue
    return {'score': score, 'p':p, 'time': time()-t0}