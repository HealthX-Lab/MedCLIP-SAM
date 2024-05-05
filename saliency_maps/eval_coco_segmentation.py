from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval 
from data_loader.metadata_convert import COCOLoader
from model_loader.clip_loader import load_clip
from tools.cam import load_cam
from evaluation.coco_eval import InfoGroundEval
import argh


class EvalCOCO:
    def __init__(self, coco_path='/home/peijie/dataset/COCO2017'):
        self.coco_path = coco_path
    
    def load_cocoGt(self, split):
        ann_file = f'{self.coco_path}/annotations/instances_{split}2017.json'
        self.cocoGt = COCO(ann_file)

    def load_cocoDt(self, result_file):
        self.cocoDt = self.cocoGt.loadRes(result_file)
    
    def eval(self, ann_type='bbox'):
        if ann_type not in ['segm','bbox','keypoints']:
            raise ValueError('invalid ann_type: {}. Choose from {segm, bbox, keypoints}'.format(ann_type))
        self.cocoEval = COCOeval(self.cocoGt, self.cocoDt, ann_type)
        self.cocoEval.evaluate()
        self.cocoEval.accumulate()
        self.cocoEval.summarize()
    
def eval_segment(json_file: str = 'results/coco_json/test.json',
                 coco_path: str = '/home/peijie/dataset/COCO2017',
                 split: str = 'val',
                 ):
    evaluator = EvalCOCO(coco_path)
    evaluator.load_cocoGt(split)
    evaluator.load_cocoDt(json_file)
    evaluator.eval('segm')

def save_segment( 
                 model_name:str = 'RN50x16',
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
                ):
    
    # load model
    model, preprocess, target_layer, cam_trans, clip = load_clip(model_name, custom=custom_clip)
    # load data
    coco_hdf_loader = COCOLoader(data_src=image_src, meta_file=meta_file, partitions=job_partitions)
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
    # save result json
    model_name_ = model_name.replace('/', '_')
    json_file_name = f'{model_name_}_{cam_version}_{custom_name}' if custom_name is not None else f'{model_name_}_{cam_version}'
    evaluator = InfoGroundEval(image_folder=image_folder, 
                            image_src=image_src, 
                            meta_data=meta_data,
                            cam=cam,
                            prompt_eng=prompt_eng,
                            is_clip=is_clip,
                            save_heatmap=False,
                            heatmap_path='',
                            save_results=save_results,
                            result_file_name='',
                            hila_transform=hila_transform)
    evaluator.save_coco_json('segment', json_file_name)
    
    
parser = argh.ArghParser()
parser.add_commands([save_segment,
                     eval_segment,
                    ])

if __name__ == '__main__':
    argh.dispatch(parser)