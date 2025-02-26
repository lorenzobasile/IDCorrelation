import torch
from tqdm import tqdm
from utils import metrics
from anatome.similarity import svcca_distance
from utils.utils import shuffle_keeping_class
import os

N=30000

models=sorted(os.listdir(f'./representations/imagenet'))
if not os.path.exists(f'results/coarse'):
    os.makedirs(f'results/coarse')

id_alg='twoNN'
device='cuda' if torch.cuda.is_available() else 'cpu'

idcor=torch.zeros(5,len(models))
pvalues=torch.zeros(5,len(models))
dcor=torch.zeros(5,len(models))
rbf_cka=torch.zeros(5,len(models))
linear_cka=torch.zeros(5,len(models))
cca=torch.zeros(5,len(models))
labels=torch.load(f'./labels/imagenet.pt')
for i, model in enumerate(tqdm(models)): 
    torch.manual_seed(10)
    complete_rep=torch.load(f'./representations/imagenet/{model}')
    for j in range(5):
        
        shuffled_rep=shuffle_keeping_class(complete_rep, labels)
        if j==0:
            if i==0:
                P=torch.randperm(len(complete_rep))[:N]      
        rep1=complete_rep[P]
        rep2=shuffled_rep[P]#[torch.randperm(N)]
        corr=metrics.id_correlation(rep1, rep2, 100, 'twoNN', return_pvalue=True)
        idcor[j,i]=(corr['corr'])
        pvalues[j,i]=(corr['p'])
        dcor[j,i]=(metrics.distance_correlation(rep1.to(device), rep2.to(device)))
        rbf_cka[j,i]=(metrics.rbf_cka(rep1.to(device), rep2.to(device)).cpu())
        linear_cka[j,i]=(metrics.linear_cka(rep1.to(device), rep2.to(device)).cpu())
        cca[j,i]=(1-svcca_distance(rep1.to(device), rep2.to(device), accept_rate=0.99, backend='svd').cpu())

torch.save(idcor, 'results/coarse/idcor.pt')
torch.save(pvalues, 'results/coarse/pvalues.pt')
torch.save(dcor, 'results/coarse/dcor.pt')
torch.save(rbf_cka, 'results/coarse/rbf_cka.pt')
torch.save(linear_cka, 'results/coarse/linear_cka.pt')
torch.save(cca, 'results/coarse/svcca.pt')
