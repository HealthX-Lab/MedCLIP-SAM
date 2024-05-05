from evaluation.coco_eval import InfoGroundEval
from data_loader.metadata_convert import COCOLoader
from model_loader.clip_loader import load_clip
from tools.cam import load_cam

import argh

def info_ground_eval(model_name:str = 'RN50x16',
                     cam_version:str = 'scorecam',
                     image_src:str = 'parts_imagenet',
                     image_folder:str = '/home/ubuntu/gscorecam/datasets/PartsImageNet/test',
                     meta_file:str = '/home/ubuntu/gscorecam/meta_data/partsImageNet_parts_test.hdf5',
                     prompt_eng: bool = False,
                     is_clip:bool = False,
                     drop_channels: bool = False,
                     topk_channels: int = 300,
                     use_channel_dict: bool = False,
                     channel_search_path: str = None,
                     custom_name: str = None,
                     job_partitions: str = '20-0', #* particite the job to multiple, i.e., '3-0' means 3 jobs, run for the first job
                     save_results: bool = True,
                     custom_clip:bool = False, #* experimental feature, keep it false for normal use
                     ):
        # load model
        model, preprocess, target_layer, cam_trans, clip = load_clip(model_name, custom=custom_clip)
        # load data
        coco_hdf_loader = COCOLoader(data_src=image_src, meta_file=meta_file, partitions=job_partitions)
        meta_data = coco_hdf_loader()
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
        heatmap_path = f'data/heatmaps/{image_src}_{model_name.replace("/","_")}_{cam_version}' if custom_name is None else f'data/heatmaps/{image_src}_{model_name.replace("/","_")}_{cam_version}_{custom_name}'
        # result_file_name = f'{model_name}_{cam_version}_j{job_partitions}_{custom_name}' if custom_name is not None else f'{model_name}_{cam_version}_j{job_partitions}'
        result_file_name = f'{model_name.replace("/","_")}_{cam_version}_{custom_name}' if custom_name is not None else f'{model_name.replace("/","_")}_{cam_version}'
        evaluator = InfoGroundEval(image_folder=image_folder, 
                                   image_src=image_src, 
                                   meta_data=meta_data,
                                   cam=cam,
                                   prompt_eng=prompt_eng,
                                   is_clip=is_clip,
                                   save_heatmap=False,
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