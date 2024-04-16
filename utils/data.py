import pandas as pd
from datasets import load_dataset, Dataset
from PIL import Image
import torch


def get_data(dataset_name):
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
    elif dataset_name=='imagenet':
        dataset=load_dataset('imagenet-1k', split='validation', streaming=True, trust_remote_code=True)

        def collate(modality, samples):
            images = [sample['image'].convert("RGB") for sample in samples]
            labels = torch.tensor([sample['label'] for sample in samples])
            return images, labels
        return dataset, collate
    elif dataset_name=='N24News':
        def clean_article(item):
            def clean_text(text):
                text = text.strip()
                text = text.replace("“", '"').replace("”", '"')
                text = text.replace("‘", "'").replace("’", "'")
                return text
            abstract = item["abstract"]
            caption = item["caption"]
            article = clean_text(item["article"])
            
            tips = "As a subscriber, you have 10 gift articles to give each month. Anyone can read what you share."
            article_cleaned = (
                article.replace(abstract, "").replace(caption, "").replace(tips, "")
            )
            while True:
                if article_cleaned.startswith("\n"):
                    article_cleaned = article_cleaned[1:]
                else:
                    break
            
            return article_cleaned

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