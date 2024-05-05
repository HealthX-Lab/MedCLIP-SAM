from tkinter import image_names
import pandas as pd
import numpy as np
from tools.iou_tool import compute_recall, xywh2xyxy, resize_box
from tools.misc import get_dataset_class_name
from tools.heatmap2bb import graycam2bb, heatmap2binary
from tools.prompt_engineering import get_engineered_prompts
from pycocotools import mask

from pytorch_grad_cam.utils.image import hila_heatmap_transform
from evaluation.misc import get_boxes_from_frame, Counter
from PIL import Image
from tqdm import tqdm
import os
import PIL
import json

class InfoGroundEval(object):
    def __init__(self,
                 image_folder: str,
                 image_src: str,
                 meta_data: pd.DataFrame,
                 cam: object,
                 prompt_eng: bool = False,
                 num_prompts: int = 10,
                 is_clip: bool = False,
                 save_heatmap: bool = False,
                 heatmap_path: str = None,
                 save_results: bool = False,
                 result_file_name: str = None,
                 hila_transform: bool = False,
                 ):
        self.cam = cam
        self.prompt_eng = prompt_eng
        self.save_heatmap = save_heatmap
        self.save_heatmap_path = heatmap_path
        self.save_results = save_results
        self.result_file_name = result_file_name
        self.is_clip = is_clip
        self.meta_data = meta_data
        self.image_src = image_src
        self.image_folder = image_folder
        self.num_prompts = num_prompts
        self.dataset_classes, self.name2id, self.id2name = get_dataset_class_name(self.image_src)
        self.counter = Counter(self, self.dataset_classes, num_prompts) if prompt_eng else Counter(self.dataset_classes)
        self.hila_transform = hila_transform
        if self.save_results:
            self.check_result_file()
        if self.image_src == 'toaster_imagenet':
            self.path_mapper = pd.read_json('meta_data/toaster_imagenet_path_mapper.json').to_dict()[0]
    
    def check_result_file(self, ):
        self.results = []
        self.result_file = f'results/{self.image_src}/{self.result_file_name}.hdf5'
        os.makedirs(f'results/{self.image_src}', exist_ok=True)
        if not os.path.exists(f'results/{self.image_src}/{self.result_file_name}'):
            file = pd.DataFrame(columns=['class_id', 'image_id', 'recall', 'max_iou'])
            file.to_hdf(self.result_file, 'stats', format='table')
            

    def get_file_name(self, image_id: int, img_frame: pd.DataFrame):
        if self.image_src == 'coco':
            file_name =  f"{image_id:012d}.jpg"
        elif self.image_src == 'parts_imagenet':
            file_name = img_frame.file_name.unique()[0]
        elif self.image_src == 'lvis':
            file_name = self.meta_data.get_group(image_id).image_path.values[0]
        elif self.image_src == 'flickr':
            file_name = f"{image_id}.jpg"
        elif self.image_src == 'toaster_imagenet':
            file_name = self.path_mapper[image_id]
        else:
            raise ValueError('Unsupported image source.')
         
        return file_name
            
    
    def get_prompt(self, keywords: list[str]):
        if not self.prompt_eng:
            return " ".join(keywords)
        if len(keywords) > 1:
            raise ValueError('Only support one keyword for prompt engineering.')
        return get_engineered_prompts(keywords[0], self.num_prompts, synonym_frist=False)
    
    
    def get_cam(self, image: PIL.Image, class_id: int):
        # get prompt
        keyword = self.id2name[class_id]
        input_text = self.get_prompt([keyword])
        # get cam
        if self.is_clip:
            inputs = (image, input_text)
            label = 0
        else:
            inputs = image
            label = class_id
        image_size = image.size
        if self.hila_transform:
            model_input_size = self.cam.preprocess.transforms[0].size
            grayscale_cam = self.cam(inputs, label, None)
            grayscale_cam = hila_heatmap_transform(grayscale_cam, model_input_size, image_size)
        else:
            grayscale_cam = self.cam(inputs, label, image_size)
        return grayscale_cam, input_text


    def get_parts_imagenet_cam(self, image: PIL.Image, class_id: int, sid:str, imagenet_sid_labels: json):
        # get prompt
        
        keyword = self.id2name[class_id]
        input_text = self.get_prompt([keyword])
        imagenet_name_text = self.get_prompt([imagenet_sid_labels[sid].replace('_', ' ') + ' ' + keyword.split(' ')[1] ])
        # get cam
        if self.is_clip:
            inputs = (image, imagenet_name_text)
            label = 0
        else:
            inputs = image
            label = class_id
        image_size = image.size
        if self.hila_transform:
            model_input_size = self.cam.preprocess.transforms[0].size
            grayscale_cam = self.cam(inputs, label, None)
            grayscale_cam = hila_heatmap_transform(grayscale_cam, model_input_size, image_size)
        else:
            grayscale_cam = self.cam(inputs, label, image_size)
        return grayscale_cam, input_text
    
    def evaluate(self, resume: bool = False, rescale: bool =False,
                 ):
        img_skip = 0
        for image_id, img_frame in tqdm(self.meta_data, desc='Evaluate', total=len(self.meta_data)):
            file_name = self.get_file_name(image_id, img_frame)
            sid = file_name.split('_')[0]
            with open('imagenet_sid_labels.json') as json_file:
                imagenet_sid_labels = json.load(json_file)

            image_path = f"{self.image_folder}/{file_name}"
            if not os.path.exists(image_path):
                img_skip += 1
                continue
            image = Image.open(image_path)
            class_group = img_frame.groupby('class_id')
            for class_id, instances in class_group:
                # get cam
                grayscale_cam, input_text = self.get_parts_imagenet_cam(image, class_id, sid, imagenet_sid_labels)
                # cam to bb
                pred_box = graycam2bb(grayscale_cam, 1)
                # get gt boxes
                gt_boxes = get_boxes_from_frame(instances, trans=xywh2xyxy)
                # rescale gt boxes
                if rescale:
                    target_size = image.size
                    box_size = (int(instances.iloc[0].image_width), int(instances.iloc[0].image_height))
                    gt_boxes = [resize_box(box, box_size, target_size) for box in gt_boxes]
                # compute recall
                choice_recall, _, _, max_iou = compute_recall([pred_box], gt_boxes)
                match = np.ones(1) if any(choice_recall) else np.zeros(1)
                self.counter.update(input_text, match)
                if self.save_heatmap:
                    raise NotImplementedError
                if self.save_results:
                    self.results.append({'class_id': class_id, 'image_id': image_id, 'recall': match.item(), 'max_iou': max_iou})

        self.counter.summary(detail=False)
        if img_skip > 0: #* for easy debugging.
            print(f"Skipped {img_skip} images due to image not found.")
        if self.save_results:
            # frame = pd.DataFrame.from_dict(self.results)
            # frame.to_hdf(self.result_file, 'stats')
            with pd.HDFStore(self.result_file, 'a') as store:
                store.append('stats', pd.DataFrame.from_dict(self.results), format='table', data_columns=True)
    
    def save_coco_json(self, type: str='segment', json_file_name: str = 'test.json', out_path: str = 'results/coco_json/'):
        coco_json = []
        img_skip = 0
        os.makedirs(out_path, exist_ok=True)
        for image_id, img_frame in tqdm(self.meta_data, desc=f'Saving {type}', total=len(self.meta_data)):
            file_name = self.get_file_name(image_id, img_frame)
            image_path = f"{self.image_folder}/{file_name}"

            image = Image.open(image_path)
            class_group = img_frame.groupby('class_id')
            for class_id, instances in class_group:
                # get cam
                grayscale_cam, input_text = self.get_cam(image, class_id)
                binary_map = heatmap2binary(grayscale_cam)
                rle = mask.encode(np.asfortranarray(binary_map))
        
                coco_json.append(dict(zip(['image_id', 'category_id', 'segmentation'], [image_id, class_id, rle['counts'].decode('utf-8')])))
                
        json.dump(coco_json, open(f'{out_path}/{json_file_name}.json', 'w'))

