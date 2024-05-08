import torch
from tqdm import tqdm
from utils import metrics
from anatome.similarity import svcca_distance
from utils.utils import shuffle_keeping_class
import os

torch.manual_seed(0)

N=30000

models=sorted(os.listdir(f'./representations/imagenet'))
if not os.path.exists(f'results/coarse'):
    os.makedirs(f'results/coarse')

id_alg='twoNN'
device='cuda' if torch.cuda.is_available() else 'cpu'

idcorr=[]
pvalues=[]
dcor=[]
rbf_cka=[]
linear_cka=[]
cca=[]
labels=torch.load(f'./labels/imagenet.pt')
for i, model in enumerate(tqdm(models)): 
    rep1=torch.load(f'./representations/imagenet/{model}')
    rep2=shuffle_keeping_class(rep1, labels)
    if i==0:
        P=torch.randperm(len(rep1))[:N]
    rep1=rep1[P]
    rep2=rep2[P][torch.randperm(N)]
    corr=metrics.id_correlation(rep1, rep2, 200, id_alg, return_pvalue=True)
    idcorr.append(corr['corr'])
    pvalues.append(corr['p'])
    dcor.append(metrics.distance_correlation(rep1.to(device), rep2.to(device)))
    rbf_cka.append(metrics.rbf_cka(rep1.to(device), rep2.to(device)).cpu())
    linear_cka.append(metrics.linear_cka(rep1.to(device), rep2.to(device)).cpu())
    cca.append(1-svcca_distance(rep1.to(device), rep2.to(device), accept_rate=0.99, backend='svd').cpu())
    print(idcorr, pvalues, dcor, rbf_cka, linear_cka, cca)
idcorr=torch.tensor(idcorr)
pvalues=torch.tensor(pvalues)
dcor=torch.tensor(dcor)
rbf_cka=torch.tensor(rbf_cka)
linear_cka=torch.tensor(linear_cka)
cca=torch.tensor(cca)


'''
torch.save(idcorr, 'results/coarse/idcorr.pt')
torch.save(pvalues, 'results/coarse/pvalues.pt')
torch.save(dcor, 'results/coarse/dcor.pt')
torch.save(rbf_cka, 'results/coarse/rbf_cka.pt')
torch.save(linear_cka, 'results/coarse/linear_cka.pt')
torch.save(cca, 'results/coarse/svcca.pt')
'''
