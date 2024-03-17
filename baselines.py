import torch
from tqdm import tqdm
from intrinsic_dimension import id_correlation, estimate_id
import os
import numpy as np
import argparse
from latentis.measure import functional
from anatome.similarity import svcca_distance
import torch.nn.functional as F

def Distance_Correlation(latent, control):

    latent = F.normalize(latent)
    control = F.normalize(control)

    matrix_a = torch.cdist(latent, latent)
    matrix_b = torch.cdist(control, control)
    matrix_A = matrix_a - torch.mean(matrix_a, dim = 0, keepdims= True) - torch.mean(matrix_a, dim = 1, keepdims= True) + torch.mean(matrix_a)
    matrix_B = matrix_b - torch.mean(matrix_b, dim = 0, keepdims= True) - torch.mean(matrix_b, dim = 1, keepdims= True) + torch.mean(matrix_b)

    Gamma_XY = torch.sum(matrix_A * matrix_B)/ (matrix_A.shape[0] * matrix_A.shape[1])
    Gamma_XX = torch.sum(matrix_A * matrix_A)/ (matrix_A.shape[0] * matrix_A.shape[1])
    Gamma_YY = torch.sum(matrix_B * matrix_B)/ (matrix_A.shape[0] * matrix_A.shape[1])


    correlation_r = Gamma_XY / torch.sqrt(Gamma_XX * Gamma_YY + 1e-9)
    return correlation_r

torch.manual_seed(0)
device='cuda' if torch.cuda.is_available() else 'cpu'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="imagenet", help="dataset")
args = parser.parse_args()

models=sorted(os.listdir(f'./representations/{args.dataset}'))

if not os.path.exists(f'results/{args.dataset}'):
    os.makedirs(f'results/{args.dataset}')

linear_cka=torch.zeros(len(models), len(models))
rbf_cka=torch.zeros(len(models), len(models))
dcor=torch.zeros(len(models), len(models))
svcca=torch.zeros(len(models), len(models))
for i, model1 in enumerate(tqdm(models)):
    rep1=torch.nn.functional.normalize(torch.load(f'./representations/{args.dataset}/{model1}'))
    if i==0:
        P=torch.randperm(len(rep1))[:20000]
    rep1=rep1[P]
    for j, model2 in enumerate(models[i:]):
        rep2=torch.nn.functional.normalize(torch.load(f'./representations/{args.dataset}/{model2}'))[P]
        svcca[i, j+i]=svcca_distance(rep1.to(device), rep2.to(device), accept_rate=0.99, backend='svd').cpu()
        dcor[i, j+i]=Distance_Correlation(rep1.to(device), rep2.to(device)).cpu()
        linear_cka[i, j+i]=functional.linear_cka(rep1.to(device), rep2.to(device)).cpu()
        rbf_cka[i, j+i]=functional.rbf_cka(rep1.to(device), rep2.to(device), sigma=0.4).cpu()
        
        
torch.save(linear_cka, f'results/{args.dataset}/linear_cka.pt')
torch.save(rbf_cka, f'results/{args.dataset}/rbf_cka.pt')
torch.save(dcor, f'results/{args.dataset}/dcor.pt')
torch.save(svcca, f'results/{args.dataset}/svcca.pt')

