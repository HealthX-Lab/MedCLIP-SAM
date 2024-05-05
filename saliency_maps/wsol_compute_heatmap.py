from wsolevaluation.data_loaders import WSOLImageLabelDataset
from torchvision.transforms import transforms
# from torch.autograd import Variable
# import torch.nn.functional as F
import torchvision
import torch
# import cv2
import os
import json
# from PIL import Image
import numpy as np
# from skimage.filters import threshold_otsu
from tqdm import tqdm
import argh
from tools.cam import CAMWrapper
from model_loader.clip_loader import load_clip
from tools.cam import load_cam


class SaveHeatMap:
    def __init__(self, dataset='imagenet', model='resnet50', out_folder=None, method='gradient', is_clip=False,  for_visualization=False, visualization_folder='visualization_samples', prompt_eng: bool=False):
        self.out_folder = f'data/heatmaps/{dataset}_{model}' if out_folder is None else out_folder
        self.model_name = model
        self.method     = method
        self.dataset    = dataset
        self.is_clip    = is_clip
        self.for_visualization = for_visualization
        self.visualization_folder = (
            f'{visualization_folder}/{dataset}_{model}/{method}'
            if for_visualization
            else None
        )
        if for_visualization:
            os.makedirs(self.visualization_folder, exist_ok=True)
        if prompt_eng:
            self.load_prompt_dict()
    
    def load_dataset(self, clip_preprocess=None, resize=(384,384), shuffle=False):
        
        if self.dataset != 'imagenet':
            raise NotImplementedError
        self.data_path = 'wsolevaluation/dataset/ILSVRC'
        meta_path = 'wsolevaluation/metadata/ILSVRC/val'
        pytorch_preprocessFn = transforms.Compose([transforms.Resize((224, 224)),
                                                # transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])

        if self.is_clip:
            if resize is not None:
                clip_preprocess.transforms.insert(0, transforms.Resize(resize))
            trans = clip_preprocess
            with open('imagenet_labels.json') as f:
                self.imagenet_label_mapper = json.load(f) 
        else:
            trans = pytorch_preprocessFn
        self.dataset = WSOLImageLabelDataset(data_root=self.data_path, metadata_root=meta_path, transform=trans, proxy=False, shuffle=shuffle)

    def load_model(self, model_name=None, target_layer=None):
        preprocess = None
        if model_name is None:
            model_name = self.model_name 
        if model_name in ['resnet50', 'resnet18']:
            self.model = torchvision.models.__dict__[model_name](pretrained=True)
            target_layer = self.model.layer4[-1] # for resnet50, should be 'layer4'
        elif model_name in ['RN50x16', 'RN50x4', 'RN50', 'RN101', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'hali']:
            self.model, preprocess, target_layer, cam_trans, clip = load_clip(model_name)
            self.clip = clip
        else:
            raise NotImplementedError("Model not yet implemented")    
        self.model.eval()
        self.model.cuda()
        return preprocess, target_layer, cam_trans
    
    def subfolder_path(self, img_path):
        name_list = img_path.split('/')
        sub_path = self.out_folder
        for sub_folder in name_list[:-1]:
            sub_path = os.path.join(sub_path, sub_folder)
        return sub_path, name_list[-1]
    
    def load_method(self, preprocess, target_layer=None, channels=None, drop=True, topk=300, dataset_size=(224, 224), cam_trans=None, use_channel_dict=False, channel_search_path=None, is_transformer=False):
        tokenizer = self.clip.tokenize if self.is_clip else None
        self.get_heatmap = load_cam(self.model, 
                                    self.method, 
                                    [target_layer], 
                                    preprocess, 
                                    cam_trans,
                                    drop=drop, 
                                    topk=topk, 
                                    is_clip=self.is_clip, 
                                    tokenizer=tokenizer, 
                                    use_channel_dict=use_channel_dict,
                                    channel_search_path=channel_search_path,
                                    is_transformer=is_transformer)
        
    def save_heatmap(self, dataset_size=(224, 224), prompt_eng=False):
        
        dataset = self.dataset
        for idx, imgdata in tqdm(enumerate(dataset), total=len(dataset)):
            img_tensor = imgdata[0]
            label      = imgdata[1]
            img_path   = imgdata[2]
            # image      = Image.open(f'{self.data_path}/{img_path}')
            # img_size = np.asarray(Image.open(f'{self.data_path}/{img_path}')).shape[:-1]
            sub_folder, file_name = self.subfolder_path(img_path)
            out_file_name = f'{sub_folder}/{file_name}.npy'
            os.makedirs(sub_folder, exist_ok=True)
            if os.path.exists(out_file_name) and not self.for_visualization:
                continue
            
            if self.is_clip:
                cls_name   = self.imagenet_label_mapper[str(label)][1].replace('_', ' ')
                # if use prompt_eng get the prompt from dictionary
                prompt = self.get_prompt(cls_name) if prompt_eng else cls_name
                
                # text_token = self.clip.tokenize(prompt).cuda()
                heatmap = self.get_heatmap((img_tensor, prompt), 0, dataset_size)
            else:
                heatmap = self.get_heatmap(img_tensor, label, dataset_size)
            
            if np.isnan(heatmap).sum():
                heatmap = np.nan_to_num(heatmap, 0) 
            
            np.save(out_file_name, heatmap)

    def load_prompt_dict(self, prompt_file='data/engineered_prompts/RN50x16_imagenet_n15.json'):
        import pandas as pd
        prompts = pd.read_json(prompt_file, orient='index')
        self.prompt_dict = prompts.to_dict()[0]   

    def get_prompt(self, cls_name):      
        return self.prompt_dict[cls_name]
    
    
def main(model='resnet50', 
         method='gradcam', 
         dataset='imagenet', 
         is_clip:bool =False, 
         shuffle: bool = False, 
         gpu: int=0, 
         topk: int = 300, 
         channels:int = None, 
         prompt_eng: bool=False, 
         out_folder='data/heatmaps', 
         drop: bool = False, 
         layer: str = None, 
         custom: str=None,
         for_visualization: bool= False,
         use_channel_dict: bool=False,
         channel_search_path: str=None,
         is_transformer: bool = False):            
    torch.cuda.set_device(gpu)
    # if channels is not None:
    #     channels = channels
    out_folder = f'{out_folder}/{model}_{method}_{dataset}' if custom is None else f'{out_folder}/{model}_{method}_{custom}_{dataset}'
    heatmap_calculator = SaveHeatMap(model=model, dataset=dataset, method=method, out_folder=out_folder, is_clip=is_clip, for_visualization=for_visualization, prompt_eng=prompt_eng)
    preprocess, target_layer, cam_trans = heatmap_calculator.load_model(target_layer = layer)
    heatmap_calculator.load_dataset(clip_preprocess=preprocess, shuffle=shuffle)
    heatmap_calculator.load_method(preprocess=None, 
                                   target_layer=target_layer, 
                                   channels=channels, 
                                   drop=drop, 
                                   topk=topk, 
                                   cam_trans=cam_trans,
                                   use_channel_dict=use_channel_dict,
                                   channel_search_path=channel_search_path,
                                   is_transformer=is_transformer)
    heatmap_calculator.save_heatmap(prompt_eng=prompt_eng)         

parser = argh.ArghParser()
parser.add_commands([main,
                    ])

if __name__ == '__main__':
    argh.dispatch(parser)
    # main(method='gradcam', model='RN50x16', is_clip=True, drop=False)