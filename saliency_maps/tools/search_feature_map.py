# import argh
# import torch
# import os
# import pprint
# import cv2
# import numpy as np
# from PIL import Image
# from dataset import COCO2017
# import pandas as pd
# from torch.utils.data.dataloader import DataLoader
# from tools.iou_tool import xywh2xyxy, compute_iou
# from tqdm import tqdm

# from visualize_clip import get_img_iter
# from loader.clip_loader import load_clip
# from tools.misc import get_dataset_class_name
# from tools.cam import CAMforCLIP
# from tools.heatmap2bb import graycam2bb

# def searchImageNet():
#     from wsolevaluation.data_loaders import WSOLImageLabelDataset
#     data_root = ""
#     meta_path = 'wsolevaluation/metadata/ILSVRC/val'            
#     dataset = WSOLImageLabelDataset(data_root=data_root, metadata_root=meta_path, transform=trans, proxy=False, choose_class=True)


# def searchMaps(img_folder = 'coco', 
#             coco_cat   = 'airplane',
#             clip_version='RN50x16', 
#             resize=False,
#             cam_version='scorecam',
#             gpu=0,
#             save_folder='data/featuremap_search',
#             drop=False,
#             coco_split='train',
#             prompt=None,
#             sample_size=500,
#                     ):
#     from tqdm import tqdm
#     torch.cuda.set_device(int(gpu))
#     pp = pprint.PrettyPrinter(indent=4)
#     os.makedirs(f'{save_folder}', exist_ok=True)
#     temp_folder = f'{save_folder}/temp'
#     os.makedirs(temp_folder, exist_ok=True)
#     print('Loading models...')
#     device = "cuda" if torch.cuda.is_available() else "cpu"


#     if clip_version == 'hila':
#         from CLIP_hila.clip import clip
#         clip_model, preprocess = clip.load('ViT-B/32', device=device, jit=False)
#     else:
#         import clip
#         clip_model, preprocess = clip.load(clip_version, device=device)


#     cam = CAMforCLIP(clip_model, drop=drop, cam_version=cam_version, mute=True)
#     print('Model loaded.')


#     # load dataset
#     dataset = COCO2017(split=coco_split, class_list=[coco_cat], noAnn=True, returnPath=True, indict=True, transform_box=xywh2xyxy)
#     myloader = DataLoader(dataset, batch_size=1, shuffle=True)    
    
#     if sample_size is None:
#         job_length = len(dataset)
#     else:
#         job_length = int(sample_size) if int(sample_size) <= len(dataset) else len(dataset)
#     # get predicted boxes
#     ious = []
#     for idx, imgdata in tqdm(enumerate(myloader), total=job_length):
#         if sample_size is not None:
#             if idx >= int(sample_size):
#                 break
#         #unpack data
#         img_path = imgdata['image'][0]
#         img_id   = imgdata['img_id'][0].item()
#         box_info = imgdata['box_info']

#         # since clip only make one choice, loop through all annotations and combine annotations with the same names as one
#         ann_frame = pd.DataFrame(columns=['gt_box', 'cls_name', 'cat_id'])
#         for annotation in box_info:
#             # cls_name    = annotation['class_name'][0]['name'][0]
#             cls_name    = annotation['class_name'][0]
#             gt_box      = annotation['bbox'].tolist()[0]
#             category_id = annotation['category_id'].item()
#             ann_frame = ann_frame.append({'gt_box':gt_box, 'cls_name':cls_name, 'cat_id':category_id}, ignore_index=True)

#         raw_image = Image.open(img_path)
#         raw_size = raw_image.size
#         input_img = preprocess(raw_image).unsqueeze(0)

#         if prompt is None:
#             text        = coco_cat
#             if (text != coco_cat) and (coco_cat != 'all'):
#                 continue
#         else:
#             text = prompt

#         gt_boxes    = ann_frame.gt_box.to_list()
#         # category_id = ann_frame[ann_frame.cls_name == coco_cat].cat_id.iloc[0]

#         # compute clip logit
#         text_token= clip.tokenize(text)
#         # derive cam       
#         # * get 3072 feature maps
#         activation_map = cam.cam.getRawActivation((input_img, text_token), img_size = raw_size)
        
#         iou_array = []
#         for feature_map in activation_map:
#             iou_list = []
#             for bin in np.arange(0.05, 1, 0.05):
#                 box = graycam2bb(feature_map, thresh_val=bin)
#                 iou_list.append(max([compute_iou(box, gt) for gt in gt_boxes])) #? Choose largest IoU as prediction (one bin)
#             iou_array.append(iou_list) # one map
#         # iou_array = np.array(iou_array)
#         ious.append(iou_array) # one image
    
#     ious = np.array(ious)
#     ious = ious.mean(axis=0)
#     np.save(f'{save_folder}/{coco_cat}_ious.npy', ious)
    
    
#     # if save:
#     #     with open(f"{save_dir}/COCO_{coco_split}_{coco_cat}_{cam_version}_{clip_version}_th{bin_thres}.json", "w") as outfile: 
#     #         json.dump(pred_list, outfile)    

# def search_coco(start_index=0, end_index=10, coco_classes_file='COCO_CLASSES.json', gpu=0, sample_size=500):
#     import json
#     gpu = int(gpu)
#     with open(coco_classes_file) as file:
#         coco_classes = json.load(file)
#     target_classes = coco_classes[int(start_index): int(end_index)]
#     for target in target_classes:
#         searchMaps(coco_cat=target, gpu=gpu, sample_size=sample_size)
    
from sklearn import metrics
from tools.utils import getFileList
import json
import numpy as np
from tqdm import tqdm
import torch

def find_best_channels(out_path, src_path, topk=125):

    files = getFileList(src_path, suffix='.npy', if_path=False)
    xrange = np.arange(0.05, 1, 0.05)
    channels = {}
    for npy_file in tqdm(files):
        class_name = npy_file.split('_')[0]
        ious = np.load(f'{src_path}/{npy_file}')
        auc = [metrics.auc(xrange, iou) for iou in ious]
        auc = torch.tensor(auc)
        top_values, top_index = auc.topk(topk)
        channels[class_name] = top_index.tolist()
    with open(out_path, 'w') as f:
        json.dump(channels, f, indent=4)
        
        
        
# parser = argh.ArghParser()
# parser.add_commands([searchMaps,
#                     search_coco,
#                     find_best_channels,
                    
#                     ])

# if __name__ == '__main__':
#     argh.dispatch(parser)
#     # save_map_index(out_path='data/featuremap_search/airplane_index.csv', src_ious='data/featuremap_search/airplane_ious.npy', topk=100)
#     # searchMaps(gpu=7)
#     # search_coco()