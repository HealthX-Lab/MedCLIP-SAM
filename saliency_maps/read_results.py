from dataclasses import dataclass
import pandas as pd
import os
import torch
import argparse
from PIL import Image, ImageDraw, ImageFont

from model_loader.clip_loader import load_clip
from tools.cam import load_cam
from evaluation.misc import get_boxes_from_frame
from tools.iou_tool import xywh2xyxy, compute_recall
from tools.drawing import Drawer, draw_text
from tools.heatmap2bb import graycam2bb
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import cm, tight_layout


# os.environ["CUDA_VISIBLE_DEVICES"]='0'

CAM_VERSIONS = ['gscorecam', 'gradcam', 'scorecam', 'hilacam']
GSCORECAM_INTERPOLATION = ['gscorecam-300', 'gscorecam-900','gscorecam-1500','gscorecam-2400', 'scorecam', 'rise'] # ['gscorecam-300', 'gscorecam-900','gscorecam-1500','gscorecam-2400']
CAM_LABELS_DICT = {
    'gscorecam': 'gScoreCAM',
    'gradcam': 'GradCAM',
    'scorecam': 'ScoreCAM',
    'hilacam': 'HilaCAM',
    'rise': 'RISE',
    'gscorecam-300': 'gScoreCAM-300',
    'gscorecam-900': 'gScoreCAM-900',
    'gscorecam-1500': 'gScoreCAM-1500',
    'gscorecam-2400': 'gScoreCAM-2400',
}


META_DIR = 'meta_data'

# configuration for gpu2
'''
CONFIGS = {
    'coco': {
        'image_dir': '/home/qi/gscorecam/datasets/MS_COCO/val2017',
        'merge_data_path':'/home/qi/gscorecam/meta_data/coco_4_methods_max_iou.hdf5',
        'meta_path': 'meta_data/coco_val_instances_stats.hdf5',
        'cat_path': 'meta_data/coco2lvis.csv',
        'results_data_dir' : '/home/qi/gscorecam/coco_saved_data_for_analysis'
    },

    'part_imagenet': {
        'image_dir': '/home/qi/gscorecam/datasets/PartsImageNet/test',
        'merge_data_path': '/home/qi/gscorecam/meta_data/part_imagenet_4_methods_max_iou.hdf5',
        'meta_path': 'meta_data/partsImageNet_parts_test.hdf5',
        'cat_path': 'meta_data/partsImageNet_categories.csv',
        'imagenet_cat_path': 'meta_data/imagenet_sid_labels.json',
        'results_data_dir' : '/home/qi/gscorecam/part_imagenet_saved_data_for_analysis'
    }

}
'''

# configuration for dlgpu/server1
CONFIGS = {
    'coco': {
        'image_dir': '/home/ubuntu/gscorecam/datasets/MS_COCO/val2017',
        'merge_data_path':'/home/ubuntu/gscorecam/meta_data/coco_4_methods_max_iou.hdf5',
        'meta_path': 'meta_data/coco_val_instances_stats.hdf5',
        'cat_path': 'meta_data/coco2lvis.csv',
        'results_data_dir' : '/home/ubuntu/gscorecam/coco_saved_data_for_analysis'
    },

    'part_imagenet': {
        'image_dir': '/home/ubuntu/gscorecam/datasets/PartsImageNet/test',
        'merge_data_path': '/home/ubuntu/gscorecam/meta_data/part_imagenet_4_methods_max_iou.hdf5',
        'meta_path': 'meta_data/partsImageNet_parts_test.hdf5',
        'cat_path': 'meta_data/partsImageNet_categories.csv',
        'imagenet_cat_path': 'meta_data/imagenet_sid_labels.json',
        'results_data_dir' : '/home/ubuntu/gscorecam/part_imagenet_saved_data_for_analysis'
    }

}

class CAMResultAnalysis:

    def __init__(self, dataset: str='coco', cam_list: list= CAM_VERSIONS) -> None:
        self.dataset = dataset
        self.cam_list = cam_list
        self.configs = CONFIGS[self.dataset]
        self.meta_data = self.load_meta()
        self.our_method = 'gscorecam'
        self.merge_data_path = f"{META_DIR}/{self.dataset}_{len(self.cam_list)}_methods_max_iou.hdf5" 
        if not self.check_exist(self.merge_data_path):
            print(f"Load the separate cam results file from folder {self.configs['results_data_dir']}")
            self.merge_data_files(self.configs['results_data_dir'])
        self.class_id_name_dict = self.load_class_id_name()
        self.imagenet_sid_name_dict = pd.read_json(self.configs['imagenet_cat_path'], typ='series')
        

    def check_exist(self, file_path: str):
        return bool(os.path.exists(file_path))

    def load_class_id_name(self):
        class_id_name_dict = {}
        if self.dataset == 'part_imagenet':
            pd_cat_data = pd.read_csv(self.configs['cat_path']) 
            for idx, cat_row in pd_cat_data.iterrows():
                class_id_name_dict[int(cat_row.cat_id)] = cat_row.cat_name

        elif  self.dataset == 'coco':
            pd_cat_data = pd.read_csv(self.configs['cat_path']) 
            for idx, cat_row in pd_cat_data.iterrows():
                class_id_name_dict[int(cat_row.coco_class_id)] = cat_row.coco_class_name

        else:
            raise ValueError("Please give the correct path to meta file")

        return class_id_name_dict

    def load_meta(self):

        return pd.read_hdf(self.configs['meta_path'])

    def merge_data_files(self, data_dir):
        df_cam_results = {}
        for cam in self.cam_list:
            file_path = os.path.join(data_dir, f"{self.dataset}_RN50x16_{cam}.hdf5") if 'hila' not in cam else os.path.join(data_dir, f"{self.dataset}_ViT-B_32_{cam}.hdf5")
            df_cam_results[cam] = pd.read_hdf(file_path)

        save_results_path = f"{META_DIR}/{self.dataset}_{len(self.cam_list)}_methods_max_iou.hdf5"
        
        all_results = []

        for idx, row in df_cam_results[self.cam_list[0]].drop_duplicates().iterrows():
            row_dict = {}
            file_name = f'{int(row.image_id):012d}.jpg' if self.dataset == 'coco' else self.meta_data.loc[(self.meta_data.class_id == row.class_id) & (self.meta_data.image_id == row.image_id)].file_name.values[0]
            print(f'{idx:05d}: file name: {file_name}, class id: {row.class_id}, image id: {row.image_id}, max iou: {row.max_iou:.03f}.')
            row_dict['class_id'] = row.class_id
            row_dict['image_id'] = row.image_id
            row_dict['file_name'] = file_name
            row_dict[ f'{self.cam_list[0]}_max_iou'] = row.max_iou
    
            for othercam in self.cam_list[1:]:
                othercam_row = df_cam_results[othercam].loc[(df_cam_results[othercam].class_id == row.class_id) & (df_cam_results[othercam].image_id == row.image_id)]
                row_dict[ f'{othercam}_max_iou'] = 0.0 if othercam_row.empty else othercam_row.max_iou.values[0]
                        
            all_results.append(row_dict)

        file = pd.DataFrame.from_dict(all_results)
        file.to_hdf(save_results_path, 'stats', format='table')

    def select_topk_gscorecam(self, topk: int=100,
                    is_ascend: bool=False, 
                    method: str='diff', # 'diff' or 'abs'
                    target_method: str='scorecam',):# 'grad' or 'score' or 'hilacam'
        
        if target_method not in CAM_VERSIONS:
            raise ValueError(f'Please choose a CAM method in {self.cam_list}.')

        df_all_cam_data = pd.read_hdf(self.merge_data_path)
        target_column = f'{target_method}_max_iou'
        if method == 'abs':
            # select top k in one column
            df_selection = df_all_cam_data.sort_values(by=target_column, ascending=is_ascend).head(topk)

        elif method == 'diff':
            our_max_iou = df_all_cam_data[f'{self.our_method.lower()}_max_iou']
            df_all_cam_data[f'{target_method}_diff'] = our_max_iou - df_all_cam_data[target_column]
            # select top k in one column
            df_selection = df_all_cam_data.sort_values(by=f'{target_method}_diff', ascending=is_ascend).head(topk)
        else:
            raise ValueError('method should be diff or abs.')

        return df_selection, is_ascend

    def visulize_selection(self, 
                    df_selection: pd.DataFrame, 
                    target_cam:str = 'gscorecam',
                    show_box: bool = True,
                    show_raw_img: bool = True,
                    is_ascend: bool = False,
                    custom_name: str = None,
                    show_text: bool = False,
                    use_imagenet_class_name: bool = False,
                    ):
        diff_target_method = [col_name for col_name in df_selection.columns.values.tolist() if "diff" in col_name]
        check_vis_method = len(diff_target_method) > 0
        
        label = 0
        if check_vis_method:
            if is_ascend:
                save_folder = f'{self.dataset}_{target_cam}_{diff_target_method[0]}_worst_{len(df_selection)}_selection_visualization'
            else:
                save_folder = f'{self.dataset}_{target_cam}_{diff_target_method[0]}_top_{len(df_selection)}_selection_visualization'
        else:
            if is_ascend:
                save_folder = f'{self.dataset}_{target_cam}_worst_{len(df_selection)}_selection_visualization'

            else:
                save_folder = f'{self.dataset}_{target_cam}_top_{len(df_selection)}_selection_visualization'

        if custom_name:
            save_folder = f'{custom_name}_' + save_folder

        if not os.path.exists(save_folder):
            os.makedirs(save_folder) 

        cam_models = self.load_cam_models()
        # cam_models = self.load_cam_models_for_interpolation()
        # self.cam_list = GSCORECAM_INTERPOLATION

        for idx, row in df_selection.iterrows():
            if use_imagenet_class_name:
                input_text = self.self.imagenet_sid_name_dict[str(row.file_name.split('_')[0])]
                input_text = input_text.replace('_', ' ')
                part_name = self.class_id_name_dict[int(row.class_id)].split(' ')[-1]
                input_text += f' {part_name}'
                
            else:
                input_text = self.class_id_name_dict[int(row.class_id)]
                input_text = " ".join(input_text.split())
                
            print(f'{input_text}')
            gt_row = self.meta_data.loc[(self.meta_data.class_id == row.class_id) & (self.meta_data.image_id == row.image_id)]

            image_path = f"{self.configs['image_dir']}/{row.file_name}"
            raw_image = Image.open(image_path)
            input_size = raw_image.size

            inputs = (raw_image, input_text)
            cams = [cam(inputs, label, input_size) for cam in cam_models]
            bboxes = [graycam2bb(cam, 1) for cam in cams]

            gt_boxes = get_boxes_from_frame(gt_row, trans=xywh2xyxy)

            cam_imgs = [Drawer.overlay_cam_on_image(raw_image, cam, use_rgb=True) for cam in cams]

            iou_list = []
            used_gt_boxes = []
            for pred_box in bboxes:
                choice_recall, _, gt_box, max_iou = compute_recall([pred_box], gt_boxes)
                iou_list.append(max_iou)
                used_gt_boxes.append(gt_box)


            if show_box:
                box_images = []
                # iou_list = [row.gscorecam_max_iou, row.gradcam_max_iou, row.scorecam_max_iou, row.hilacam_max_iou]
                for cam_img, box, iou, method, used_gt_box in zip(cam_imgs, bboxes, iou_list, self.cam_list, used_gt_boxes):
                    box_img = Drawer.draw_boxes(cam_img, [box],color='red', loc='below', width = 3)
                    if show_text:
                        box_img = draw_text(box_img, [f'{CAM_LABELS_DICT[method]} IoU: {iou:.03f}'],text_size=16, inside=False)
                    if len(gt_boxes) != 0:
                        box_img = Drawer.draw_boxes(box_img, gt_boxes, color='green', width = 3)
                    box_images.append(box_img)
                final_cam_images = box_images
            else:
                final_cam_images = cam_imgs
            if show_text:
                raw_image = draw_text(raw_image, [f'Prompt: {input_text}'], text_size=16, inside=False)
            if show_raw_img:
                final_cam_images.insert(0, raw_image)

            cat_image = Drawer.concat(final_cam_images)

            if check_vis_method:
                save_name = f"diff_{row[diff_target_method[0]]:.03f}_{int(row.class_id):02d}_{int(row.image_id)}"
                for cam in self.cam_list:
                    save_name += f'_{cam}_{iou_list[self.cam_list.index(cam)]:.03f}'
            else:
                save_name = f'{target_cam}_{iou_list[self.cam_list.index(target_cam)]:.03f}_{int(row.class_id):02}_{int(row.image_id)}'
                for cam in self.cam_list:
                    if cam != target_cam:
                        save_name += f'_{cam}_{iou_list[self.cam_list.index(cam)]:.03f}'
            save_name += f'_{row.file_name.split(".")[0]}.jpg'
            cat_image.save(f'{save_folder}/{save_name}')

    def load_cam_models(self, 
                        is_clip: bool = True, 
                        drop_channels: bool = False, 
                        topk_channels: int = 300, 
                        use_channel_dict: bool = False, 
                        channel_search_path: str = None, 
                        custom_clip: bool = False, #* experimental feature, keep it false for normal use
                        ):
        # load models
        cam_models = []
        for cam_version in self.cam_list: 
            if cam_version == 'hilacam':
                model, preprocess, target_layer, cam_trans, clip = load_clip('ViT-B/32', custom=custom_clip)
            else: 
                model, preprocess, target_layer, cam_trans, clip = load_clip('RN50x16', custom=custom_clip)
            print(f'{cam_version} is loading...')
            cam_models.append(load_cam(model=model,
                                cam_version=cam_version, 
                                preprocess=preprocess, 
                                target_layers=[target_layer],
                                cam_trans=cam_trans,
                                drop=drop_channels,
                                topk=topk_channels,
                                is_clip=is_clip,
                                tokenizer=clip.tokenize,
                                use_channel_dict=use_channel_dict,
                                channel_search_path=channel_search_path))
                                
        return cam_models


    def load_cam_models_for_interpolation(self, 
                        is_clip: bool = True, 
                        drop_channels: bool = False, 
                        topk_channels: int = 300, 
                        use_channel_dict: bool = False, 
                        channel_search_path: str = None, 
                        custom_clip: bool = False, #* experimental feature, keep it false for normal use
                        ):
        # load models
        cam_models = []
        for cam_version in self.cam_list: 
            if 'gscorecam' in cam_version:
                cam_name = cam_version.split('-')[0]
                topk_channels = int(cam_version.split('-')[-1])
            
            else:
                cam_name = cam_version

            model, preprocess, target_layer, cam_trans, clip = load_clip('RN50x16', custom=custom_clip)
            print(f'{cam_version} is loading...')
            cam_models.append(load_cam(model=model,
                                cam_version=cam_name, 
                                preprocess=preprocess, 
                                target_layers=[target_layer],
                                cam_trans=cam_trans,
                                drop=drop_channels,
                                topk=topk_channels,
                                is_clip=is_clip,
                                tokenizer=clip.tokenize,
                                use_channel_dict=use_channel_dict,
                                channel_search_path=channel_search_path))
                                
        return cam_models

    def add_caption_to_images(self, image_folder: str='diff_coco_gscorecam_scorecam_diff_top_50_selection_visualization', target_method: str='scorecam', use_imagenet_class_name=True):
        
        unit_size = 256
        font = ImageFont.truetype(font='font/arial.ttf', size=20)

        images = [file for file in os.listdir(image_folder) if file.endswith(('jpeg', 'png', 'jpg'))]

        for image in images:
            data_collect = image.split('_')
            cam_versions = [x for x in data_collect if 'cam' in x]
            

            if use_imagenet_class_name:
                input_text = self.imagenet_sid_name_dict[str(image.split('_')[-2])]
                input_text = input_text.replace('_', ' ')
                part_name = self.class_id_name_dict[int(image.split('_')[2])].split(' ')[-1]
                input_text += f' {part_name}'

            else:
                input_text = self.class_id_name_dict[int(image.split('_')[3])]
                input_text = " ".join(input_text.split())

            img = Image.open(os.path.join(image_folder, image))

            fix_width = unit_size*(len(cam_versions)+1) if 'gscorecam' in cam_versions else unit_size*len(cam_versions)
            img_size = img.size
            img_ratio = img_size[0]/img_size[1]
            adapt_height = int(fix_width/img_ratio)
            resized_img = img.resize((fix_width, adapt_height), Image.NEAREST)

            if 'diff' in data_collect:
                # draw text in the image
                # draw = ImageDraw.Draw(resized_img)
                new_img = Image.new('RGB', (unit_size*3, adapt_height + font.getsize('ABCabc')[1] + 10), (255, 255, 255))
                new_img.paste(resized_img.crop((0, 0,unit_size*2, adapt_height)), (0, 0))
                draw = ImageDraw.Draw(new_img)

                draw.text((5, adapt_height), f'Prompt: {input_text}', fill='black', font=font) 
                for idx, cam in enumerate(cam_versions):
                    if cam == 'gscorecam':
                        value_idx = data_collect.index(cam) + 1
                        start_pixel = (idx+1)*(unit_size) + 5
                        draw.text((start_pixel, adapt_height), f'{CAM_LABELS_DICT[cam]}: {float(data_collect[value_idx]):}', fill='black', font=font)

                    elif cam == target_method:

                        value_idx = data_collect.index(cam) + 1

                        new_img.paste(resized_img.crop((unit_size*(idx+1), 0, unit_size*(idx+2), adapt_height)), (unit_size*2, 0))
                        start_pixel = 2*(unit_size) + 5
                        draw.text((start_pixel, adapt_height), f'{CAM_LABELS_DICT[cam]}: {float(data_collect[value_idx]):}', fill='black', font=font)

            elif 'progress' in data_collect:

                cam_versions.append('rise') if 'scorecam' in cam_versions else None
                new_img = Image.new('RGB', (fix_width, adapt_height + font.getsize('ABCabc')[1] + 10), (255, 255, 255))
                new_img.paste(resized_img, (0 ,0))
                draw = ImageDraw.Draw(new_img)
                if 'gscorecam-300' in cam_versions:
                    draw.text((5, adapt_height), f'Prompt: {input_text}', fill='black', font=font)

                for idx, cam in enumerate(cam_versions):
                    if 'gscorecam' in cam:
                        start_pixel = (idx+1)*(unit_size) + 5
                    else:
                        start_pixel = idx*(unit_size) + 5
                    value_idx = data_collect.index(cam) + 1
                    draw.text((start_pixel, adapt_height), f'{CAM_LABELS_DICT[cam]}: {float(data_collect[value_idx]):}', fill='black', font=font)

            else:
                new_img = Image.new('RGB', (fix_width, adapt_height + font.getsize('ABCabc')[1] + 10), (255, 255, 255))
                new_img.paste(resized_img, (0, 0))
                draw = ImageDraw.Draw(new_img)

                draw.text((5, adapt_height), f'Prompt: {input_text}', fill='black', font=font) 
                for idx, cam in enumerate(cam_versions):
                    value_idx = data_collect.index(cam) + 1
                    start_pixel = (idx+1)*(unit_size) + 5
                    draw.text((start_pixel, adapt_height), f'{CAM_LABELS_DICT[cam]}: {float(data_collect[value_idx]):}', fill='black', font=font)

            new_img.save(os.path.join(image_folder, f'labled_{image[:-4]}.pdf'))


class PlotCAMRersults:
    def __init__(self, dataset: str='coco', cam_list: list=CAM_VERSIONS) -> None:
        self.dataset = dataset
        self.cam_list = cam_list
        self.configs = CONFIGS[self.dataset]
        self.color_map = cm.tab10.colors[:4]
        try:
            self.merge_cam_data = pd.read_hdf(self.configs['merge_data_path']) 
            self.class_id_name = pd.read_csv(self.configs['cat_path'])

        except: 
            raise ValueError("Please provide the necessary data files.")

        self.image_dir = self.configs['image_dir']
        self.save_folder = f'{self.dataset}_saved_plots'


    def iou_by_class_analysis(self):            
        dict_class_iou_mean = {}
        dict_class_iou_max = {}
        dict_class_iou_min = {}
        class_list = []

        for class_id, df_class in self.merge_cam_data.groupby('class_id'):
            class_list.append(class_id)
            for cam in self.cam_list:
                if cam in dict_class_iou_mean:
                    dict_class_iou_mean[cam].append(df_class[f'{cam}_max_iou'].mean())
                    dict_class_iou_max[cam].append(df_class[f'{cam}_max_iou'].max())
                    dict_class_iou_min[cam].append(df_class[f'{cam}_max_iou'].min())

                else:
                    dict_class_iou_mean[cam] = [df_class[f'{cam}_max_iou'].mean()]
                    dict_class_iou_max[cam] = [df_class[f'{cam}_max_iou'].max()]
                    dict_class_iou_min[cam] = [df_class[f'{cam}_max_iou'].min()]
        print(dict_class_iou_mean)
        dict_class_iou = {'average': dict_class_iou_mean, 'max': dict_class_iou_max, 'min':dict_class_iou_min}
        for analysis_type in ['average', 'max', 'min']:
            self.bar_plot_comparison(dict_class_iou[analysis_type], class_list, analysis_type=analysis_type)


    def bar_plot_comparison(self, dict_iou, class_list, analysis_type):
        fig, ax = plt.subplots(figsize=(20,16))
        class_labels = [self.class_id_name[class_id] for class_id in class_list]
        np_class = np.asarray(class_list) * 5
        width = 1.0

        ax.bar(np_class - 3*width/2, dict_iou['gscorecam'], width=width, label='gscorecam')
        ax.bar(np_class - width/2, dict_iou['gradcam'], width=width,  label='gradcam')
        ax.bar(np_class + width/2, dict_iou['scorecam'], width=width,  label='scorecam')
        ax.bar(np_class + 3*width/2, dict_iou['hilacam'], width=width,  label='hilacam')
        plt.xticks(np_class, class_labels,rotation = 45, ha='right')

        plt.xlabel(f"{self.dataset} catergory names")
        plt.ylabel(f"The {analysis_type} over each catergory")
        plt.title(f"The {analysis_type} for different CAM methods")
        plt.legend()
        # plt.show()
        plt.savefig(f'plot_{self.dataset}_{analysis_type}.jpg')


    def iou_by_area_analysis(self, num_sections: int = 3, even_samples: bool=True):
        
        # plt.style.use('ggplot')
        dict_iou_by_area = {}
        all_object_area_ratio = []
        meta_data = pd.read_hdf(self.configs['meta_path'])
        df_all_cam_data = self.merge_cam_data.reset_index()

        for idx, df_row in df_all_cam_data.iterrows():
            df_instant = meta_data[(meta_data.class_id == df_row.class_id) &  (meta_data.image_id == df_row.image_id)]

            if self.dataset == 'coco':
                object_area_ratio = df_instant.object_ratio.values.mean()
            elif self.dataset == 'part_imagenet':
                area = df_instant.object_size.values.mean()
                image_path = f'{self.image_dir}/{df_instant.file_name.values[0]}'
                image_h, image_w = Image.open(image_path).size
                image_area = image_h*image_w
                object_area_ratio = area/image_area

            all_object_area_ratio.append(object_area_ratio)
            for cam in self.cam_list:
                if cam not in dict_iou_by_area.keys():
                    dict_iou_by_area[cam] = [df_row[f'{cam}_max_iou']]

                else:
                    dict_iou_by_area[cam].append(df_row[f'{cam}_max_iou'])

        np_area_ratio = np.array(all_object_area_ratio)
        sorted_area_ratio = np.sort(np_area_ratio)
        
        if even_samples:
            sub_area_ratio = np.array_split(sorted_area_ratio, num_sections)
        else:
            # interval_setting = list(np.delete(np.linspace(0, 1.0 , num=num_sections+1, endpoint=True), 0))
            interval_setting = list(np.delete(np.linspace(0, 1.0 , num=num_sections, endpoint=False), 0))
            sub_area_ratio = np.array(np.split(sorted_area_ratio, np.bincount(np.digitize(sorted_area_ratio, interval_setting)).cumsum())[:-1])

        dict_iou_by_area_plots = {}
        area_interval_list = []

        for idx, sub_area_list in enumerate(sub_area_ratio):
            print(idx, sub_area_list)
            area_interval_list.append(max(sub_area_list))

        def interval_idx(area: float): 
            for interval_idx, max_area in enumerate(area_interval_list):
                if area <= max_area:
                    return interval_idx
            
            raise ValueError('Please area value is not in the list.')

        for cam in self.cam_list:

            dict_iou_by_area_plots[cam] = [[] for _ in range(len(area_interval_list))]
            for area, iou in zip(all_object_area_ratio, dict_iou_by_area[cam]):  
                iou_idx = interval_idx(area)
                # print(f'area size: {iou_idx}')
                dict_iou_by_area_plots[cam][iou_idx].append(1.0 if iou > 1.0 else iou) 

        plt.grid(True)
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize = (3*len(sub_area_ratio)+3, 8))
        # fig, ax = plt.subplots()
        fig.subplots_adjust(right=0.8)
        # fig.tight_layout(rect=[0,0,0.8,1])


        mark_fig = {'gscorecam': dict(marker='o'), 
                    'gradcam': dict(marker='s'),
                    'scorecam': dict(marker='X'),
                    'hilacam': dict(marker='^')}

        box_fig = {'gscorecam': dict(facecolor=self.color_map[0]), 
                    'gradcam': dict(facecolor=self.color_map[1]),
                    'scorecam': dict(facecolor=self.color_map[2]),
                    'hilacam': dict(facecolor=self.color_map[3])}

        pos_adjustment = {'gscorecam': -0.6, 
                    'gradcam': -0.2,
                    'scorecam': 0.2,
                    'hilacam': 0.6}

        box_width = 0.4
        plot_labels = [f'0 - {max_iou:0.3f} \n ({len(sub_area_ratio[idx])})' if idx == 0 else f'{area_interval_list[idx-1]:0.3f} - {min(max_iou, 1.0):0.3f} \n ({len(sub_area_ratio[idx])})' for idx,  max_iou in enumerate(area_interval_list)]
        axes_boxplot = []

        for cam in self.cam_list:
            x_pos = np.array(range(len(area_interval_list)))*2+1+pos_adjustment[cam]
            axes_boxplot.append(ax.boxplot(dict_iou_by_area_plots[cam], flierprops=mark_fig[cam], boxprops=box_fig[cam], widths=box_width, positions=x_pos, patch_artist=True))
            
            for pos, iou_group in zip(x_pos, dict_iou_by_area_plots[cam]):
                plt.text(pos, -0.03, f'{(np.array(iou_group) > 0.5).sum()/len(iou_group):.03f}', horizontalalignment='center', size='medium', color=box_fig[cam]['facecolor'], weight='semibold')
        # ax.autoscale(enable=True, axis='x')
        ax.legend([bp['boxes'][0] for bp in axes_boxplot], [CAM_LABELS_DICT[cam] for cam in self.cam_list], loc='upper left', bbox_to_anchor=(1.01, 1))
        # plt.xticks(np_area_ratio,rotation = 45, ha='right')
        plt.xticks(np.array(range(len(area_interval_list)))*2+1, plot_labels)
        plt.xlabel(f"The range of the object ratio")
        plt.ylabel(f"The IoU of the heatmap")
        # plt.title(f"The IoU for different CAM methods")
        if even_samples:
            plt.savefig(f'plot_{self.dataset}_iou_over_same_samples_{num_sections}_interval.jpg')
        else:
            plt.savefig(f'plot_{self.dataset}_iou_over_same_dist_{num_sections}_interval.jpg')
        
        # plt.show()


    def accuracy_by_class_analysis(self,all_cam_data_path: str='/home/ubuntu/gscorecam/meta_data/parts_imagenet_max_iou_all_methods.hdf5'):
        df_all_cam_data = pd.read_hdf(all_cam_data_path)
        df_cat_name = pd.read_csv('meta_data/partsImageNet_categories.csv')
        df_all_cam_data = df_all_cam_data.rename({'gscore_max_iou': 'gscorecam_max_iou', 'grad_max_iou': 'gradcam_max_iou', 'score_max_iou': 'scorecam_max_iou'}, axis=1)

        dict_class_accuracy = {}

        class_list = []

        for class_id, df_class in df_all_cam_data.groupby('class_id'):
            class_list.append(class_id)
            for cam in self.cam_list:
                if cam in dict_class_accuracy:
                    dict_class_accuracy[cam].append(len(df_class[f'{cam}_max_iou'][df_class[f'{cam}_max_iou'] > 0.5])/len(df_class[f'{cam}_max_iou']))

                else:
                    dict_class_accuracy[cam] = [len(df_class[f'{cam}_max_iou'][df_class[f'{cam}_max_iou'] > 0.5])/len(df_class[f'{cam}_max_iou'])]
        self.bar_plot_comparison(dict_class_accuracy, class_list, analysis_type='accuracy')

    


def filter_pd_selection(df: pd.DataFrame, file_name_list):
    return df[df.file_name.isin(file_name_list)]

def plot_progress_gscorecam(df: pd.DataFrame):

    df_selection, is_ascend = df.select_topk_gscorecam(topk=50)
    for method_list in [GSCORECAM_INTERPOLATION[:-2], GSCORECAM_INTERPOLATION[-2:]]:
        print(method_list)
        df.cam_list = method_list
        if 'rise' in method_list:
            df.visulize_selection(df_selection, show_raw_img=False, is_ascend=is_ascend, custom_name='progress')
        else: 
            df.visulize_selection(df_selection, is_ascend=is_ascend, custom_name='progress')
        torch.cuda.empty_cache()



            





if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # dataset setting
    parser.add_argument('--vis_method', type=str,
                        default='diff', help="To compute the difference of iou or absolute value (diff or abs)")
    parser.add_argument('--is_ascend', action='store_true', help="whether rank the iou in ascending order. (True or False)")
    parser.add_argument('--dataset', type=str, default='part_imagenet', help="Choose the dataset for the results anaylysis.")
    parser.add_argument('--target_method', type=str, default='scorecam', help="folder to save the visualization results.")
    args = parser.parse_args()

    # vis_data = CAMResultAnalysis(dataset = 'coco', cam_list = CAM_VERSIONS)
    vis_data = CAMResultAnalysis(dataset = args.dataset)
    # df_selection, is_ascend = vis_data.select_topk_gscorecam(topk=50, target_method=args.target_method)
    # vis_data.visulize_selection(df_selection, is_ascend=is_ascend, custom_name='imagenet_class_diff', use_imagenet_class_name=True)


    # plot_progress_gscorecam(vis_data)
    # file_name_list = ['000000580410.jpg']
    # df_selection = filter_pd_selection(df_selection, file_name_list)

    
    # plot_cam = PlotCAMRersults(dataset = 'coco')
    # plot_cam.iou_by_area_analysis(even_samples=False, num_sections=5)


    vis_data.add_caption_to_images(image_folder='imagenet_class_diff_part_imagenet_gscorecam_scorecam_diff_top_50_selection_visualization', target_method='scorecam',use_imagenet_class_name=True)
