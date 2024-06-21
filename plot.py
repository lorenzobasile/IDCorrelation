import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="imagenet", help="dataset")
parser.add_argument('--metric', type=str, default='pvalues', help="metric to plot")
args = parser.parse_args()

path = f'results/{args.dataset}/{args.metric}.pt'

def make_sym(matrix):
    sym=matrix+matrix.T
    sym[np.diag_indices_from(sym)] /= 2
    return sym

def offdiagonal(matrix):
    np.fill_diagonal(matrix, 0)
    return np.sum(matrix)/(matrix.shape[0]*(matrix.shape[0]-1))
if 'imagenet' in args.dataset:
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
            'CLIP',
            ])
else:
    model_list = np.array(['albert_albert-base-v2',
            'bert-base-cased',
            'bert-base-uncased',
            'google_efficientnet-b0',
            'google_electra-base-discriminator',
            'google_vit-base-patch16-224',
            'google_vit-hybrid-base-bit-384',
            'microsoft_resnet-18',
            'openai_clip-t-vit-base-patch16',
            'openai_clip-v-vit-base-patch16',
            ])
    model_names = np.array(['ALBERT',
            'BERT-C',
            'BERT-U',
            'EfficientNet',
            'Electra',
            'ViT',
            'ViT-hybrid',
            'ResNet',
            'CLIP-T',
            'CLIP-V',
            ])
    isvision=np.ones(len(model_list))
    isvision[np.where(np.char.find(model_list, 'bert')>=0)[0]]=0
    isvision[np.where(np.char.find(model_list, 'clip-t')>=0)[0]]=0
    isvision[np.where(np.char.find(model_list, 'electra')>=0)[0]]=0
    idx=np.argsort(isvision)
    model_list=model_list[idx]
    model_names=model_names[idx]
    isvision=isvision[idx]

metric_names={
        'idcor':'$I_d$Cor',
        'dcor':'dCor',
        'rbf_cka':'CKA (RBF)',
        'linear_cka':'CKA (linear)',
        'svcca':'SVCCA',
        'pvalues':'p-values',
        
}
results=torch.load(path)

#fig, ax = plt.subplots(figsize=(14, 11) if 'idcorr' in args.metric else (11, 11))
fig, ax = plt.subplots(figsize=(14, 11))

if 'imagenet' in args.dataset:
    ax=sns.heatmap(make_sym(results.numpy()), vmin=0, vmax=1, ax=ax, annot=True, cmap='Blues', annot_kws={"fontsize": 16}, cbar=True)#'idcorr' in args.metric)
else:
    ax=sns.heatmap(make_sym(results.numpy())[idx][:,idx], vmin=0, vmax=1, ax=ax, annot=True, cmap='Blues', annot_kws={"fontsize": 16}, cbar=True)#, cbar='idcorr' in args.metric)
print(f"Off diagonal mean for {args.metric}: {offdiagonal(make_sym(results.numpy()))}")
cbar = ax.collections[0].colorbar
# here set the labelsize by 20
cbar.ax.tick_params(labelsize=16)
plt.title(metric_names[args.metric], fontsize=40)
#change font size of ticks

plt.xticks(ticks=np.arange(0.5, len(model_names)+0.5), labels=model_names, rotation=45, fontsize=18)
plt.yticks(ticks=np.arange(0.5, len(model_names)+0.5), labels=model_names, rotation=0, fontsize=18)

if 'imagenet' not in args.dataset:
   # Add second layer of labels
    second_layer_labels_x = ['TEXT', 'IMAGE']
    second_layer_labels_y = ['TEXT', 'IMAGE']

    plt.axhline(len(model_names)/2, color='k', linewidth=1.5)
    plt.axvline(len(model_names)/2, color='k', linewidth=1.5)

    for i, label in enumerate(second_layer_labels_x):
        plt.text((len(model_names)/4)*(2*i+1), len(model_names)+1.5, label, ha='center', va='center', fontsize=18)

    for i, label in enumerate(second_layer_labels_y):
        plt.text(-1.7, (len(model_names)/4)*(2*i+1), label, ha='center', va='center', rotation=90, fontsize=18)

plt.savefig(path[:-2]+"svg", dpi=200, bbox_inches='tight', format='svg')
