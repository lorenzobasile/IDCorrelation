import torch
from tqdm import tqdm
from intrinsic_dimension import id_correlation, estimate_id
import os
import numpy as np
import argparse

torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="imagenet", help="dataset")
parser.add_argument('--id_alg', type=str, default="twoNN", help="ID estimation algorithm")

args = parser.parse_args()

models=sorted(os.listdir(f'./representations/{args.dataset}'))

if not os.path.exists(f'results/{args.dataset}/{args.id_alg}'):
    os.makedirs(f'results/{args.dataset}/{args.id_alg}')

ids=torch.zeros(len(models))
pvalues=torch.zeros(len(models), len(models))
zscores=torch.zeros(len(models), len(models))
noshuffle=torch.zeros(len(models), len(models))
shuffle_mean=torch.zeros(len(models), len(models))
shuffle_std=torch.zeros(len(models), len(models))
for i, model1 in enumerate(tqdm(models)):
    rep1=torch.nn.functional.normalize(torch.load(f'./representations/{args.dataset}/{model1}'))
    if i==0:
        P=torch.randperm(len(rep1))[:20000]
    rep1=rep1[P]
    ids[i]=estimate_id(rep1.to('cuda'), 'twoNN').item()
    for j, model2 in enumerate(models[i:]):
        rep2=torch.nn.functional.normalize(torch.load(f'./representations/{args.dataset}/{model2}'))[P]
        corr=id_correlation(rep1, rep2, 200, args.id_alg)
        zscores[i,j+i]=corr['Z']
        pvalues[i,j+i]=corr['p']
        noshuffle[i,j+i]=corr['original Id']
        shuffle_mean[i,j+i]=corr['Mean shuffled Id']
        shuffle_std[i,j+i]=corr['Std shuffled Id']
torch.save(ids, f'results/{args.dataset}/{args.id_alg}/ids.pt')
torch.save(pvalues, f'results/{args.dataset}/{args.id_alg}/pvalues.pt')
torch.save(zscores, f'results/{args.dataset}/{args.id_alg}/zscores.pt')
torch.save(noshuffle, f'results/{args.dataset}/{args.id_alg}/noshuffle.pt')
torch.save(shuffle_mean, f'results/{args.dataset}/{args.id_alg}/shuffle_mean.pt')
torch.save(shuffle_std, f'results/{args.dataset}/{args.id_alg}/shuffle_std.pt')


