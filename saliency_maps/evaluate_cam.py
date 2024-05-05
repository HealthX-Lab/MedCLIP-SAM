from evaluation.coco_eval import InfoGroundEval, ThresholdSearch
from data_loader.metadata_convert import COCOLoader
from model_loader.clip_loader import load_clip
from tools.cam import load_cam

import pandas as pd
import os
import argh

def search_threshold_from_heatmap(cam_version: str = 'gradcam',
                     grid_length: int = 20,
                     heatmap_path: str = 'data/heatmaps/coco/RN50x16_gradcam',
                     meta_file: str = 'data/heatmaps/coco/RN50x16_gradcam.hdf5',
                     ):
    meta_data = pd.read_hdf(meta_file, 'stats')
    os.makedirs('data/search_threshold', exist_ok=True)
    result_file = f'data/search_threshold/{cam_version}.hdf5'
    
    searcher = ThresholdSearch(heatmap_path=heatmap_path,
                               cam_version=cam_version,
                               meta_data=meta_data,
                               result_file_name=result_file,
                               grid_length=grid_length,
                               )
    searcher.search()

def info_ground_eval(model_name:str = 'RN50x16',
                     cam_version:str = 'gradcam',
                     image_src:str = 'coco',
                     image_folder:str = '/home/pzc0018@auburn.edu/dataset/COCO2017/val2017',
                     meta_file:str = 'meta_data/coco_val_instances_stats.hdf5',
                     prompt_eng: bool = False,
                     is_clip:bool = False,
                     drop_channels: bool = False,
                     topk_channels: int = 300,
                     use_channel_dict: bool = False,
                     channel_search_path: str = None,
                     custom_name: str = None,
                     job_partitions: str = '1-0', #* particite the job to multiple, i.e., '3-0' means 3 jobs, run for the first job
                     save_results: bool = False,
                     custom_clip:bool = False, #* experimental feature, keep it false for normal use
                     hila_transform: bool = False,
                     save_heatmap: bool = False,
                     subset_size: int = -1, # use all data
                     subset_type: str = 'class_balanced', # choose from {'class_balanced', 'random'}
                     threshold: float = 1.0,
                     alpha: float = 1.0, # parameter for heatmap to bounding box
                     heatmap_path: str = None,
                     ):
        # load model
        model, preprocess, target_layer, cam_trans, clip = load_clip(model_name, custom=custom_clip)
        # load data
        coco_hdf_loader = COCOLoader(data_src=image_src, meta_file=meta_file, partitions=job_partitions, subset_size=subset_size, subset_type=subset_type)
        meta_data = coco_hdf_loader()
        # load cam
        is_transformer = "ViT" in model_name
        cam = load_cam(model=model,
                       cam_version=cam_version, 
                       preprocess=preprocess, 
                       target_layers=[target_layer],
                       cam_trans=cam_trans,
                       drop=drop_channels,
                       topk=topk_channels,
                       is_clip=is_clip,
                       tokenizer=clip.tokenize,
                       use_channel_dict=use_channel_dict,
                       channel_search_path=channel_search_path,
                       is_transformer=is_transformer,)
        # define output path
        model_name_ = model_name.replace('/', '_')
        if heatmap_path is None:
            heatmap_path = f'data/heatmaps/{image_src}/{model_name_}_{cam_version}' if custom_name is None else f'data/heatmaps/{image_src}/{model_name_}_{cam_version}_{custom_name}'
        os.makedirs(heatmap_path, exist_ok=True)
        # result_file_name = f'{model_name}_{cam_version}_j{job_partitions}_{custom_name}' if custom_name is not None else f'{model_name}_{cam_version}_j{job_partitions}'
        result_file_name = f'{model_name_}_{cam_version}_{custom_name}' if custom_name is not None else f'{model_name_}_{cam_version}'
        if save_heatmap:
            subset_df = pd.concat([meta_data.get_group(key) for key in meta_data.groups])
            # subset_df.to_hdf(f'data/heatmaps/{image_src}/{result_file_name}.hdf5', 'stats')
        # load evaluator
        evaluator = InfoGroundEval(image_folder=image_folder, 
                                   image_src=image_src, 
                                   meta_data=meta_data,
                                   cam=cam,
                                   prompt_eng=prompt_eng,
                                   is_clip=is_clip,
                                   save_heatmap=save_heatmap,
                                   heatmap_path=heatmap_path,
                                   save_results=save_results,
                                   result_file_name=result_file_name,
                                   hila_transform=hila_transform)

        # run evaluation
        evaluator.evaluate(threshold=threshold, alpha=alpha)

parser = argh.ArghParser()
parser.add_commands([info_ground_eval,
                     search_threshold_from_heatmap,
                    ])

if __name__ == '__main__':
    argh.dispatch(parser)