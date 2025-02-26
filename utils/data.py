import pandas as pd
from datasets import load_dataset, Dataset
from PIL import Image
import torch



def get_data(dataset_name, flickr_id=0):
    if dataset_name=='coco':
        dataset = load_dataset('json', data_files='./data/annotations/captions_val2014.json', split='all')['annotations'][0]
        dataset=pd.DataFrame.from_dict(dataset).drop_duplicates(subset='image_id')
        dataset = Dataset.from_pandas(dataset)
        def collate(modality, samples):
            if modality=='image':
                images = [Image.open(f'./data/val2014/COCO_val2014_'+str(sample["image_id"]).zfill(12)+'.jpg').convert("RGB") for sample in samples]
                return images
            else:
                texts = [sample['caption'] for sample in samples] 
                return texts
        return dataset, collate
    elif dataset_name=='flickr':
        dataset=load_dataset('nlphuji/flickr30k', split='test', streaming=True, trust_remote_code=True)
        def collate(modality, samples):
            if modality=='image':
                images = [sample['image'].convert("RGB") for sample in samples]
                return images
            else:
                texts = [sample['caption'][flickr_id] for sample in samples] 
                return texts
        return dataset, collate
    elif dataset_name=='imagenet':
        dataset=load_dataset('imagenet-1k', split='validation', streaming=True, trust_remote_code=True)

        def collate(modality, samples):
            images = [sample['image'].convert("RGB") for sample in samples]
            labels = torch.tensor([sample['label'] for sample in samples])
            return images, labels
        return dataset, collate

    
    elif dataset_name=='N24News':
        dataset = load_dataset('json', data_files='./data/N24News/news/nytimes.json', split='all')
        print(len(dataset))

        def collate(modality, samples):
            if modality=='image':
                images = [Image.open(f'./data/N24News/imgs/{sample["image_id"]}.jpg').convert("RGB") for sample in samples]
                return images
            else:
                texts = [sample['caption'] for sample in samples] 
                return texts
        return dataset, collate

