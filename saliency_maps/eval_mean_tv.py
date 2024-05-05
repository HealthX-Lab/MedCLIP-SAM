import argh
import torch
import os
import pprint
import cv2
import numpy as np
from PIL import Image
import pandas as pd

from tools.iou_tool import xywh2xyxy, compute_iou
from tqdm import tqdm

from tools.heatmap2bb import graycam2bb
from evaluation.coco_eval import InfoGroundEval
from data_loader.metadata_convert import COCOLoader
from model_loader.model_loader import load_model
from tools.cam import load_cam
from evaluation.misc import get_boxes_from_frame
from tools.total_variance import TotalVariance
class EvaluateTotalVariance(InfoGroundEval):
                
    def compute_tv(self, cam_version: str):
        get_tv = TotalVariance()
        self.record = []
        for idx, row in tqdm(self.meta_data.iterrows(), desc='Computing Total Variance', total=len(self.meta_data)):
            instance_frame = row.to_frame()
            image_id = row.image_id
            class_id = row.class_id
            cls_name = self.id2name[class_id]
            file_name = self.get_file_name(image_id, instance_frame)
            image_path = f"{self.image_folder}/{file_name}"
            image = Image.open(image_path)

            # get cam
            out_file_name = file_name.split("/")[-1] if "/" in file_name else file_name
            heatmap_file_path = f"{self.save_heatmap_path}/{out_file_name}_{class_id:03d}.npy"
            if os.path.exists(heatmap_file_path):
                cam = np.load(heatmap_file_path)
            else:
                cam = self.get_cam(image, class_id, cls_name)
            # evaluate
            cam = np.nan_to_num(cam)
            self.record.append(get_tv(torch.tensor(np.maximum(cam, 0))))
        tvs = torch.cat(self.record)
        mean_tv = tvs.mean()
        std_tv = tvs.std()
        print(f'Method: {cam_version}\nMean Total Variance: {mean_tv}\nStandard Deviation: {std_tv}')
            

        
def compute_tv(model_name:str = 'RN50x16',
                cam_version:str = 'gscorecam',
                image_src:str = 'coco',
                image_folder:str = '/home/peijie/dataset/COCO2017/val2017',
                meta_file:str = 'meta_data/coco_val_instances_stats.hdf5',
                subset_size:int = 10000,
                subset_type: str = 'random',
                custom_name: str = None,
                heatmap_path: str = None,
                ):
        # define output path
    if heatmap_path is None:
        model_name_ = model_name.replace('/', '_')
        heatmap_path = f'data/heatmaps/{image_src}/{model_name_}_{cam_version}' if custom_name is None else f'data/heatmaps/{image_src}/{model_name_}_{cam_version}_{custom_name}'

    # load model
    model, preprocess, target_layer, cam_trans, tokenizer = load_model(model_name, True, False)
    # load data
    coco_hdf_loader = COCOLoader(data_src=image_src, meta_file=meta_file, subset_size=subset_size, subset_type=subset_type, image_base_seperation=False)
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
    evaluator = EvaluateTotalVariance(image_folder, image_src, meta_data, cam, heatmap_path=heatmap_path, is_clip=True)

    evaluator.compute_tv(cam_version)

parser = argh.ArghParser()
parser.add_commands([compute_tv,
                    ])

if __name__ == '__main__':
    argh.dispatch(parser)
    # compute_tv(cam_version='testcam')