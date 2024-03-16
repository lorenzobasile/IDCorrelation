from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTokenizer, CLIPTextModelWithProjection
from transformers import ViTImageProcessor, ViTModel
from transformers import AutoImageProcessor, ResNetModel
from transformers import ViTHybridImageProcessor, ViTHybridModel
from transformers import EfficientNetImageProcessor, EfficientNetModel
from transformers import SiglipImageProcessor, SiglipVisionModel
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
from transformers import AlbertTokenizer, AlbertModel



def get_model(model_name):
    if 'clip-v' in model_name:
        clf=False
        conv=False
        return CLIPImageProcessor.from_pretrained(model_name.replace('clip-v', 'clip')), CLIPVisionModelWithProjection.from_pretrained(model_name.replace('clip-v', 'clip')), clf, conv, 'image'
    elif 'clip-t' in model_name:
        clf=False
        conv=False
        return CLIPTokenizer.from_pretrained(model_name.replace('clip-t', 'clip')), CLIPTextModelWithProjection.from_pretrained(model_name.replace('clip-t', 'clip')), clf, conv, 'text'
    elif 'siglip' in model_name:
        clf=False
        conv=False
        return SiglipImageProcessor.from_pretrained(model_name), SiglipVisionModel.from_pretrained(model_name), clf, conv, 'image'
    elif 'resnet' in model_name:
        clf=True
        conv=True
        return AutoImageProcessor.from_pretrained(model_name), ResNetModel.from_pretrained(model_name), clf, conv, 'image'
    elif 'hybrid' in model_name:
        clf=True
        conv=False
        return ViTHybridImageProcessor.from_pretrained(model_name), ViTHybridModel.from_pretrained(model_name), clf, conv, 'image'
    elif 'vit' in model_name:
        clf=True
        conv=False
        return ViTImageProcessor.from_pretrained(model_name), ViTModel.from_pretrained(model_name), clf, conv, 'image'
    elif 'roberta' in model_name:
        clf=False
        conv=False
        return RobertaTokenizer.from_pretrained(model_name), RobertaModel.from_pretrained(model_name), clf, conv, 'text'
    elif 'albert' in model_name:
        clf=False
        conv=False
        return AlbertTokenizer.from_pretrained(model_name), AlbertModel.from_pretrained(model_name), clf, conv, 'text'
    elif 'bert' in model_name:
        clf=False
        conv=False
        return BertTokenizer.from_pretrained(model_name), BertModel.from_pretrained(model_name), clf, conv, 'text'
    else:
        clf=True
        conv=True
        return EfficientNetImageProcessor.from_pretrained(model_name), EfficientNetModel.from_pretrained(model_name), clf, conv, 'image'
