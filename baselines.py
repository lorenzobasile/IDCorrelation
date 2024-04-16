import torch
from tqdm import tqdm
import os
import numpy as np
import argparse
from utils import functional
from anatome.similarity import svcca_distance
import torch.nn.functional as F

torch.manual_seed(0)
device='cuda' if torch.cuda.is_available() else 'cpu'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="imagenet", help="dataset")
args = parser.parse_args()

N=30000

models=sorted(os.listdir(f'./representations/{args.dataset}'))

if not os.path.exists(f'results/{args.dataset}'):
    os.makedirs(f'results/{args.dataset}')

linear_cka=torch.zeros(len(models), len(models))
rbf_cka=torch.zeros(len(models), len(models))
dcor=torch.zeros(len(models), len(models))
svcca=torch.zeros(len(models), len(models))
for i, model1 in enumerate(tqdm(models)):
    rep1=torch.load(f'./representations/{args.dataset}/{model1}')
    if i==0:
        P=torch.randperm(len(rep1))[:N]
    rep1=rep1[P]
    for j, model2 in enumerate(models[i:]):
        rep2=torch.load(f'./representations/{args.dataset}/{model2}')[P]
        svcca[i, j+i]=1-svcca_distance(rep1.to(device), rep2.to(device), accept_rate=0.99, backend='svd').cpu()
        dcor[i, j+i]=functional.Distance_Correlation(rep1.to(device), rep2.to(device)).cpu()
        linear_cka[i, j+i]=functional.linear_cka(rep1.to(device), rep2.to(device)).cpu()
        rbf_cka[i, j+i]=functional.rbf_cka(rep1.to(device), rep2.to(device)).cpu()
        print(svcca[i, j+i], dcor[i, j+i], linear_cka[i, j+i], rbf_cka[i, j+i])
        
        
torch.save(linear_cka, f'results/{args.dataset}/linear_cka.pt')
torch.save(rbf_cka, f'results/{args.dataset}/rbf_cka.pt')
torch.save(dcor, f'results/{args.dataset}/dcor.pt')
torch.save(svcca, f'results/{args.dataset}/svcca.pt')
