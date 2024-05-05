import argh
import contextlib
import torch
import os
import pprint
import cv2
import numpy as np
from PIL import Image
import pandas as pd

from tools.iou_tool import xywh2xyxy, compute_iou
from tqdm import tqdm
from sklearn import metrics

from tools.heatmap2bb import graycam2bb
from evaluation.coco_eval import InfoGroundEval
from data_loader.metadata_convert import COCOLoader
from model_loader.model_loader import load_model
from tools.cam import load_cam
from evaluation.misc import get_boxes_from_frame, Counter
from tools.utils import getFileList
import multiprocessing
from multiprocessing import Pool
# from torch.multiprocessing import Pool, set_start_method
# with contextlib.suppress(RuntimeError):
    #  set_start_method('spawn', force=True)
     
import json
import random

class SaveIoUs(object):
    def __init__(self, activation_folder: str, output_path: str, meta_frame: pd.DataFrame):
        self.output_path = output_path
        self.activation_folder = activation_folder
        self.file_list = getFileList(self.activation_folder, if_path=False, suffix='.npy')
        self.image_size_dict = json.load(open(f'{activation_folder}/image_size_dict.json'))
        self.meta = meta_frame

    def resize_map(self, activation_map, image_size):
        cam = activation_map[0]
        cam = np.maximum(cam, 0)
        result = []
        # fix bug in cv2 that it does not support type 23. (float16)
        if cam.dtype == 'float16':
            cam = cam.astype(np.float32)
        for img in cam:
            img = cv2.resize(img, image_size)
            img = img - np.min(img)
            img = img/ (np.max(img) + 1e-8) 
            result.append(img)
        result = np.float32(result)
        return result

    def get_ious(self, file_list):
        # ious = {}
        
        for file_name in tqdm(file_list):
            image_id, class_id = int(file_name.split('_')[0]), int(file_name.split('_')[1].split('.')[0])
            out_put_path = f'{self.output_path}/{image_id}_{class_id}.npy'
            if os.path.exists(out_put_path):
                continue
            activations = np.load(f'{self.activation_folder}/{file_name}')
            activation_map = self.resize_map(activations, self.image_size_dict[str(image_id)])
            instances = self.meta[(self.meta.image_id == image_id) & (self.meta.class_id == class_id)]
            gt_boxes = get_boxes_from_frame(instances, trans=xywh2xyxy)
            iou = self._get_iou(activation_map, gt_boxes)
            np.save(out_put_path, iou) 
            # ious[f'{image_id}_{class_id}'] = iou
        # return ious
        
    def _get_iou(self, activation_map, gt_boxes):
        ious = []
        for feature_map in activation_map:
            iou_list = []
            for bin in np.arange(0, 1, 0.05):
                box = graycam2bb(feature_map, thresh_val=bin)
                iou_list.append(max((compute_iou(box, gt) for gt in gt_boxes)))
            ious.append(iou_list)
        return np.array(ious)
        
    

class WeightEvaluation(object):
    def __init__(self, weight_path, iou_path):
        self.weight_path = weight_path
        self.iou_path = iou_path
        self.file_list = getFileList(self.iou_path, if_path=False, suffix='.npy')
        
    def eval(self, out_path: str, metric: str = 'auc'):
        results = []
        mean_metric = []
        for file in tqdm(self.file_list):
            image_id, class_id = int(file.split('_')[0]), int(file.split('_')[1].split('.')[0])
            weight = np.load(f'{self.weight_path}/{file}')
            weight = np.nan_to_num(weight)
            #* discard the nagetive values and normalize the weight such that the sum of the weight is 1
            if weight.sum() != 1:
                weight = np.maximum(weight, 0)
                weight /= weight.sum()

            iou = np.load(f'{self.iou_path}/{file}')
            metric_value = self.get_metric(iou, metric)
            score = np.einsum("nc, c->n", weight, metric_value)
            # score = np.mean(weight * iou)
            max_score = np.max(iou)
            mean_metric.append(metric_value)
            results.append({'image_id': image_id, 'class_id': class_id, 'score': score, 'max_score': max_score, 'metric': metric_value})

        result_df = pd.DataFrame.from_dict(results)

        mean_score = result_df['score'].mean()[0]
        mean_best = result_df['max_score'].mean()
        print(f'mean score: {mean_score:.4f}, mean best: {mean_best:.4f}, mean metric: {np.mean(mean_metric):.4f}')
        result_df.to_hdf(f'{out_path}_{metric}_results.h5', key='results')
        
                
    def get_metric(self, iou: str or np.ndarray, method='auc'):
        if isinstance(iou, str):
            iou = np.load(iou)
        if method == 'auc':
            xrange = np.arange(0, 1, 0.05)
            metric = [metrics.auc(xrange, iou[i]) for i in range(len(iou))]
        # sort auc in descending order, and get the orders
        # auc_sorted = np.argsort(auc)[::-1] # 3072
        elif method == 'iou':
            metric = iou.max(axis=1)
        return metric
    

class RecordWeight(InfoGroundEval):
                
    def record_weight(self, weight_path: str, actication_path: str, save_activation: bool = False):
        image_size = {}
        for image_id, img_frame in tqdm(self.meta_data, desc='Recording weights', total=len(self.meta_data)):
            file_name = self.get_file_name(image_id, img_frame)
            image_path = f"{self.image_folder}/{file_name}"
            image = Image.open(image_path)
            class_group = img_frame.groupby('class_id')
            image_size[image_id] = [image.size[0], image.size[1]]
            for class_id, instances in class_group:
                out_file_name = f"{image_id}_{class_id}.npy"
                
                keyword = self.id2name[class_id]
                
                # * get 3072 feature maps
                weight_output_file_path = f'{weight_path}/{out_file_name}'
                if os.path.exists(weight_output_file_path):
                   continue

                _ = self.get_cam(image, class_id, keyword)
                weights = self.cam.cam.weights # (1, 3072)
                np.save(weight_output_file_path, weights)
                if save_activation:
                    activations = self.cam.cam.activations[0] # (1, 3072, 12, 12)            
                    np.save(f'{actication_path}/{out_file_name}', activations)
        if save_activation:
            json.dump(image_size, open(f'{actication_path}/image_size_dict.json', 'w'))
        
def record_weights(model_name:str = 'RN50x16',
                    cam_version:str = 'gradcam',
                    image_src:str = 'coco',
                    image_folder:str = '/home/peijie/dataset/COCO2017/train2017',
                    meta_file:str = 'meta_data/coco_train_instances_stats.hdf5',
                    split:str = 'train',
                    subset_size:int = 50,
                    job_partitions: str = '1-0',
                    save_activation: bool = False,
                    topk: int = 300,
                    custom_name: str = None,
                    ):
    weight_path = f'data/weights/{image_src}_{split}/{model_name}_{cam_version}' if custom_name is None else f'data/weights/{image_src}_{split}/{model_name}_{cam_version}_{custom_name}'
    os.makedirs(weight_path, exist_ok=True)
    actication_path = f'data/activations/{image_src}_{split}/{model_name}'
    os.makedirs(actication_path, exist_ok=True)
    # load model
    model, preprocess, target_layer, cam_trans, tokenizer = load_model(model_name, True, False)
    # load data
    coco_hdf_loader = COCOLoader(data_src=image_src, meta_file=meta_file, subset_size=subset_size, subset_type='class_balanced', image_base_seperation=True, partitions=job_partitions)
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
                    topk=topk)
    evaluator = RecordWeight(image_folder, image_src, meta_data, cam, is_clip=True)

    evaluator.record_weight(weight_path, actication_path, save_activation)

def save_ious(activation_folder: str = 'data/activations/coco_train/RN50x16',
              output_path: str = 'data/activations/coco_train/RN50x16_ious',
              meta_frame: str = 'meta_data/coco_train_instances_stats.hdf5',
              ):
    os.makedirs(output_path, exist_ok=True)
    meta_data = pd.read_hdf(meta_frame)
    iou_tool = SaveIoUs(activation_folder, output_path, meta_data)
    n = 64
    p = Pool(n)
    subjob_length = len(iou_tool.file_list)//n
    args = [(iou_tool.file_list[i*subjob_length: (i+1)*subjob_length], ) for i in range(n+1)]
    p.starmap(iou_tool.get_ious, args)
    p.close()
    p.join()
    

def eval_weight(cam_version:str = 'gradcam',
                model_name:str = 'RN50x16',
                weight_path: str = 'data/weights/coco_train',
                iou_path: str = 'data/activations/coco_train/RN50x16_ious',
                metric: str = 'auc'):
    weight_path = f'{weight_path}/{model_name}_{cam_version}'
    evaluator = WeightEvaluation(weight_path, iou_path)
    evaluator.eval(weight_path, metric)


parser = argh.ArghParser()
parser.add_commands([record_weights,
                    save_ious,
                    eval_weight,
                    ])

if __name__ == '__main__':
    argh.dispatch(parser)
    # eval_weight()
    # record_weights(cam_version='groupcam')