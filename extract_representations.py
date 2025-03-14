import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from utils.models import get_model
from utils.data import get_data
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="microsoft/resnet-18", help="model name")
parser.add_argument('--dataset', type=str, default="imagenet", help="dataset")
parser.add_argument('--batch_size', type=int, default=256, help="batch size")
args = parser.parse_args()

if not os.path.exists(f'./representations/{args.dataset}'):
    os.makedirs(f'./representations/{args.dataset}')

dataset, collate=get_data(args.dataset)

processor, model, clf, conv, modality = get_model(args.model)

dataloader=DataLoader(dataset, collate_fn=partial(collate, modality), batch_size=args.batch_size, shuffle=False)

device='cuda' if torch.cuda.is_available() else 'cpu'

model=model.to(device)

model.eval()
with torch.no_grad():
    representations=[]
    labels=[]
    for i, x, in tqdm(enumerate(dataloader)):
        if modality=='text':
            x=processor(x, padding=True, truncation=True, return_tensors="pt")
        else:
            if 'imagenet' in args.dataset:
                x,y=x
                labels.append(y)
            N=len(x)
            x=processor(x, return_tensors="pt")
        if 'blip' in args.model:
            out = model.vision_model(**x.to(device), output_hidden_states=True)
            reps = torch.nn.functional.normalize(model.vision_proj(out[0][:,0,:]))
            representations.append(reps.detach().cpu())
            continue
        out = model(**x.to(device), output_hidden_states=True)
        if conv or 'siglip' in args.model:
            reps = out['pooler_output'].reshape(N, -1)
        elif 'clip-v' in args.model:
            reps = out['image_embeds']
        elif 'clip-t' in args.model:
            reps = out['text_embeds']
        else:
            reps = out['hidden_states'][-1][:,0]
        representations.append(reps.detach().cpu())
        
representations = torch.cat(representations)
if 'imagenet' in args.dataset:
    labels = torch.cat(labels)
    torch.save(labels, f'./labels/{args.dataset}.pt')
torch.save(representations, f'./representations/{args.dataset}/{args.model.replace("/", "_")}.pt')

