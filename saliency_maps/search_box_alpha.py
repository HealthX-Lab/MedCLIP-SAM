import argh
import torch
import os
import pprint
import cv2
import numpy as np
from PIL import Image
import pandas as pd

from tools.iou_tool import xywh2xyxy, compute_recall
from tqdm import tqdm

from tools.heatmap2bb import graycam2bb, Heatmap2BB
from evaluation.coco_eval import InfoGroundEval
from data_loader.metadata_convert import COCOLoader
from model_loader.model_loader import load_model
from tools.cam import load_cam
from evaluation.misc import get_boxes_from_frame, Counter
import json
class searchAlpha(InfoGroundEval):
                
    def search_aplha(self, grid_size: float, result_file_name:str, threshold: float = 1.0):
        grid_length = int(1/grid_size) + 1
        self.counter = Counter(list(self.id2name.keys()), grid_length)
        img_skip = 0
        for image_id, img_frame in tqdm(self.meta_data, desc='Evaluate', total=len(self.meta_data)):
            file_name = self.get_file_name(image_id, img_frame)
            image_path = f"{self.image_folder}/{file_name}"
            image = Image.open(image_path)
            class_group = img_frame.groupby('class_id')
            for class_id, instances in class_group:
                heatmap_file_path = f"{self.save_heatmap_path}/{file_name}_{class_id:03d}.npy"
                # get prompt
                keyword = self.id2name[class_id]
                input_text = self.get_prompt([keyword])
                # get cam
                if os.path.exists(heatmap_file_path):
                    grayscale_cam = np.load(heatmap_file_path)
                else:
                    continue #* Use existing heatmap only
                    # grayscale_cam = self.get_cam(image, class_id, input_text)
                # get candidate boxes
                # candidate_boxes = Heatmap2BB._get_pred_boxes_experimental(grayscale_cam, grid_size, threshold=threshold)
                candidate_boxes = Heatmap2BB.get_pred_boxes(grayscale_cam, grid_size, threshold=threshold)
                gt_boxes = get_boxes_from_frame(instances, trans=xywh2xyxy)
                
                # compute recall
                recall_list, _, _, _ = compute_recall(candidate_boxes, gt_boxes, grid_length)
                match = recall_list.numpy()
                self.counter.update(class_id, match)

        self.result = pd.DataFrame(self.counter.recalls, columns=np.arange(0, 1+grid_size, grid_size))
        self.counter.summary(detail=False, label_wise=True)
        self.result.to_hdf(result_file_name, 'stats')        

def search_alpha(model_name:str = 'RN50x16',
                cam_version:str = 'gradcam',
                image_src:str = 'coco',
                image_folder:str = '/home/peijie/dataset/COCO2017/train2017',
                meta_file:str = 'meta_data/coco_train_instances_stats.hdf5',
                out_folder:str = 'data/aplha_search',
                subset_size:int = 100,
                grid_size: float = 0.02,
                threshold: float = 1.0,
                ):
    os.makedirs(out_folder, exist_ok=True)
    # load model
    model, preprocess, target_layer, cam_trans, tokenizer = load_model(model_name, True, False)
    # load data
    coco_hdf_loader = COCOLoader(data_src=image_src, meta_file=meta_file, subset_size=subset_size, subset_type='class_balanced')
    meta_data = coco_hdf_loader()
    # load cam
    cam = load_cam(model=model,
                    cam_version=cam_version, 
                    preprocess=preprocess, 
                    target_layers=[target_layer],
                    cam_trans=cam_trans,
                    is_clip=True,
                    tokenizer=tokenizer, 
                    drop=False,
                    topk=300)
    # evaluate on different alpha values
    model_name_ = model_name.replace('/', '_')
    heatmap_path = f'data/heatmaps/coco/{model_name_}_{cam_version}_train'
    searcher = searchAlpha(image_folder, image_src, meta_data, cam, heatmap_path=heatmap_path, save_heatmap=True, is_clip=True)
    result_file_name = f'{out_folder}/{model_name_}_{cam_version}_{image_src}_a{grid_size}.hdf5'
    searcher.search_aplha(grid_size, result_file_name, threshold)
        
        
parser = argh.ArghParser()
parser.add_commands([search_alpha,
                     
                    
                    ])

if __name__ == '__main__':
    argh.dispatch(parser)
    # search_alpha()