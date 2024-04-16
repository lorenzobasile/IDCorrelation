import torch
from tqdm import tqdm
from utils.intrinsic_dimension import id_correlation, estimate_id
import os
import argparse

torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="imagenet", help="dataset")

args = parser.parse_args()

N=30000

models=sorted(os.listdir(f'./representations/{args.dataset}'))
if not os.path.exists(f'results/{args.dataset}'):
    os.makedirs(f'results/{args.dataset}')

id_alg='twoNN'
device='cuda' if torch.cuda.is_available() else 'cpu'

ids=torch.zeros(len(models))
pvalues=torch.zeros(len(models), len(models))
zscores=torch.zeros(len(models), len(models))
noshuffle=torch.zeros(len(models), len(models))
shuffle_mean=torch.zeros(len(models), len(models))
shuffle_std=torch.zeros(len(models), len(models))
for i, model1 in enumerate(tqdm(models)): 
    rep1=torch.load(f'./representations/{args.dataset}/{model1}')
    rep1=torch.load(f'./representations/{args.dataset}/{model1}')
    if i==0:
        P=torch.randperm(len(rep1))[:N]
    rep1=rep1[P]
    #rep1=shuffle_keeping_class(rep1, labels)
    ids[i]=estimate_id(rep1.to(device), id_alg).item()
    for j, model2 in enumerate(models[i:]):
        rep2=torch.load(f'./representations/{args.dataset}/{model2}')[P]
        corr=id_correlation(rep1, rep2, 200, id_alg)
        zscores[i,j+i]=corr['Z']
        pvalues[i,j+i]=corr['p']
        noshuffle[i,j+i]=corr['original Id']
        shuffle_mean[i,j+i]=corr['Mean shuffled Id']
        shuffle_std[i,j+i]=corr['Std shuffled Id']
torch.save(ids, f'results/{args.dataset}/ids.pt')
torch.save(pvalues, f'results/{args.dataset}/pvalues.pt')
torch.save(zscores, f'results/{args.dataset}/zscores.pt')
torch.save(noshuffle, f'results/{args.dataset}/noshuffle.pt')
torch.save(shuffle_mean, f'results/{args.dataset}/shuffle_mean.pt')
torch.save(shuffle_std, f'results/{args.dataset}/shuffle_std.pt')


