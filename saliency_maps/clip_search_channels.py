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
import json
import random
class searchImageNet(InfoGroundEval):
                
    def record_ious_per_class(self, class_id: int, out_path: str):
        class_wise_data = self.meta_data[self.meta_data.class_id == class_id]
        image_groups = class_wise_data.groupby('image_id')
        self.record = np.zeros((3072, 20)) # choose to record and save class wise results.
        for image_id, img_frame in tqdm(image_groups, desc=f'Evaluating class {class_id:03d}', total=len(image_groups)):
            # image_id = row.image_id
            # img_frame = row.to_frame().T
            cls_name = self.id2name[class_id]

            file_name = self.get_file_name(image_id, img_frame)
            image_path = f"{self.image_folder}/{file_name}"
            image = Image.open(image_path)
            # * get 3072 feature maps
            img_tensor = self.cam.preprocess(image).unsqueeze(0).cuda()
            cls_token = self.cam.tokenizer(cls_name).cuda()
            activation_map = self.cam.cam.getRawActivation(input_tensor=(img_tensor, cls_token), img_size = image.size)
            
            # evaluate
            gt_boxes = get_boxes_from_frame(img_frame)
            for i, feature_map in enumerate(activation_map):
                iou_list = []
                for bin in np.arange(0, 1, 0.05):
                    box = graycam2bb(feature_map, thresh_val=bin)
                    iou_list.append(max((compute_iou(box, gt) for gt in gt_boxes)))
                self.record[i] = np.array(iou_list)
        np.save(f'{out_path}/{class_id}_ious.npy', self.record)
    
    def save_channel_dict(self, iou_path:str, out_path:str):
        from sklearn import metrics
        channel_dict = {}
        xrange = np.arange(0, 1, 0.05)
        iou_files = os.listdir(iou_path)
        for iou_file in enumerate(iou_files):
            class_id = int(iou_file.split('_')[1].split('.')[0])
            class_ious = np.load(iou_file)
            class_name = self.id2name[class_id]
            auc = [metrics.auc(xrange, iou) for iou in class_ious]
            # sort auc in descending order, and get the orders
            auc_sorted = np.argsort(auc)[::-1] # 3072
            channel_dict[class_name] = auc_sorted.tolist()
        json.dump(channel_dict, open(f'{out_path}/imagenet_channel_dict', 'w'))
        
def search_imagenet_channels(model_name:str = 'RN50x16',
                            cam_version:str = 'scorecam',
                            image_src:str = 'imagenet',
                            image_folder:str = '/home/peijie/dataset/ILSVRC2012/train',
                            meta_file:str = 'meta_data/imagenet_train_instances.hdf5',
                            subset_size:int = 100,
                            output_file:str = 'data/channel_search_ious'
                            ):
    os.makedirs(output_file, exist_ok=True)
    # load model
    model, preprocess, target_layer, cam_trans, tokenizer = load_model(model_name, True, False)
    # load data
    coco_hdf_loader = COCOLoader(data_src=image_src, meta_file=meta_file, subset_size=subset_size, subset_type='class_balanced', image_base_seperation=False)
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
    searcher = searchImageNet(image_folder, image_src, meta_data, cam)
    class_list = meta_data.class_id.unique().tolist()
    random.shuffle(class_list)
    for class_id in class_list:
        if not os.path.exists(f'{output_file}/{class_id}_ious.npy'):
            searcher.record_ious_per_class(class_id, output_file)
    if len(os.lisdir(output_file)) == 1000:
        searcher.save_channel_dict(output_file, 'data/channel_dict')
        
        
parser = argh.ArghParser()
parser.add_commands([search_imagenet_channels,
                    
                    ])

if __name__ == '__main__':
    argh.dispatch(parser)
    search_imagenet_channels()