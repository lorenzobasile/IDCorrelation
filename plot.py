import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="imagenet", help="dataset")
parser.add_argument('--metric', type=str, default='pvalues', help="metric to plot")
parser.add_argument('--id_alg', type=str, default='twoNN', help="ID estimation algorithm")
args = parser.parse_args()

path = f'results/{args.dataset}/{args.id_alg}/{args.metric}.pt'

def make_sym(matrix):
    sym=matrix+matrix.T
    sym[np.diag_indices_from(sym)] /= 2
    return sym
if args.dataset=='imagenet':
    model_list = np.array([
            'google_efficientnet-b0',
            'google_siglip-base-patch16-224',
            'google_vit-base-patch16-224',
            'google_vit-base-patch32-384',
            'google_vit-hybrid-base-bit-384',
            'google_vit-large-patch16-224',
            'microsoft_resnet-18',
            'openai_clip-v-vit-base-patch16',
            ])
    model_names = np.array([
            'EfficientNet',
            'SigLIP',
            'ViT-B-16-224',
            'ViT-B-32-384',
            'ViT-hybrid',
            'ViT-L-16-224',
            'ResNet',
            'CLIP-V',
            ])
else:
    model_list = np.array(['albert_albert-base-v2',
            'bert-base-cased',
            'bert-base-uncased',
            'google_efficientnet-b0',
            'google_vit-base-patch16-224',
            'google_vit-hybrid-base-bit-384',
            'microsoft_resnet-18',
            'openai_clip-t-vit-base-patch16',
            'openai_clip-v-vit-base-patch16',
            'roberta-base',
            ])
    model_names = np.array(['ALBERT',
            'BERT-C',
            'BERT-U',
            'EfficientNet',
            'ViT',
            'ViT-hybrid',
            'ResNet',
            'CLIP-T',
            'CLIP-V',
            'RoBERTa',
            ])
    isvision=np.ones(len(model_list))
    isvision[np.where(np.char.find(model_list, 'bert')>=0)[0]]=0
    isvision[np.where(np.char.find(model_list, 'clip-t')>=0)[0]]=0
    idx=np.argsort(isvision)
    model_list=model_list[idx]
    model_names=model_names[idx]
    isvision=isvision[idx]

results=torch.load(path)

fig, ax = plt.subplots(figsize=(14, 11))


sns.heatmap(make_sym(results.numpy()), ax=ax, annot=True, cmap='Blues', annot_kws={"fontsize": 12})

#plt.title('$I_d$ correlation p-value', fontsize=20)
plt.xticks(ticks=np.arange(0.5, len(model_names)+0.5), labels=model_names, rotation=45)
plt.yticks(ticks=np.arange(0.5, len(model_names)+0.5), labels=model_names, rotation=0)

plt.savefig(path[:-2]+"png", dpi=200, bbox_inches='tight')