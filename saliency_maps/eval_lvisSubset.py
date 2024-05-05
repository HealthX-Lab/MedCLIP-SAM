from evaluation.coco_eval import InfoGroundEval
from data_loader.metadata_convert import COCOLoader
from model_loader.clip_loader import load_clip
from tools.cam import load_cam
from data_loader.get_hdf import get_subset
import pandas as pd
import os
import argh

def info_ground_eval(model_name:str = 'RN50x16',
                     cam_version:str = 'gradcam',
                     image_src:str = 'lvis',
                     image_folder:str = '/home/peijie/dataset/COCO2017',
                     meta_file:str = 'meta_data/lvis_plus_train_instances_stats.hdf5',
                     num_of_classes:int = 1,
                     num_of_same_classes:int = 1,
                     subset_size:int = 200,
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
                     save_heatmap: bool = False
                     ):
        # load data
        meta = pd.read_hdf(meta_file, 'stats')
        meta_subset = get_subset(meta, num_of_classes=num_of_classes, num_of_same_classes=num_of_same_classes, num_of_images=subset_size)
        coco_hdf_loader = COCOLoader(data_src=image_src, meta_file=meta_subset, partitions=job_partitions)
        meta_data = coco_hdf_loader()
        # load model
        model, preprocess, target_layer, cam_trans, clip = load_clip(model_name, custom=custom_clip)
        # load cam
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
                       channel_search_path=channel_search_path)
        # load evaluator
        model_name_ = model_name.replace('/', '_')
        heatmap_path = f'data/heatmaps/{image_src}/{model_name_}_{cam_version}_c{num_of_classes:02d}' if custom_name is None else f'data/heatmaps/{image_src}/{model_name_}_{cam_version}__c{num_of_classes:02d}_{custom_name}'
        os.makedirs(heatmap_path, exist_ok=True)
        # result_file_name = f'{model_name_}_{cam_version}_j{job_partitions}_{custom_name}' if custom_name is not None else f'{model_name_}_{cam_version}_j{job_partitions}'
        result_file_name = f'{model_name_}_{cam_version}_c{num_of_classes:02d}_{custom_name}' if custom_name is not None else f'{model_name_}_{cam_version}_c{num_of_classes:02d}'
        evaluator = InfoGroundEval(image_folder=image_folder, 
                                   image_src=image_src, 
                                   meta_data=meta_data,
                                   cam=cam,
                                   prompt_eng=prompt_eng,
                                   is_clip=is_clip,
                                   save_heatmap=save_heatmap,
                                   heatmap_path=heatmap_path,
                                   save_results=save_results,
                                   result_file_name=result_file_name)
        
        # run evaluation
        evaluator.evaluate()

parser = argh.ArghParser()
parser.add_commands([info_ground_eval,
                    ])

if __name__ == '__main__':
    argh.dispatch(parser)