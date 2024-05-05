import torch
import os
import pprint
import argparse
import random
import pandas as pd
from PIL import Image
from typing import List, Tuple, Dict, Union
from open_clip import create_model_from_pretrained, get_tokenizer

from tools.cam import CAMWrapper, CLIP_topk_channels
from tools.drawing import concat_images, draw_text, Drawer
from tools.heatmap2bb import graycam2bb
from tools.iou_tool import xywh2xyxy
from tools.misc import get_dataset_class_name
from model_loader.clip_loader import load_clip, load_clip_from_checkpoint
import torch.nn as nn

class BiomedCLIP(nn.Module):
    def __init__(
        self,
        model
    ):
        super().__init__()
        self.model = model

    def forward(self, images,texts):
        # Getting Image and Text Features

        image_embeddings, text_embeddings, logit_scale = self.model(images, texts)

        # Calculating the Loss
        image_logits = (logit_scale * image_embeddings @ text_embeddings.t())
        text_logits = (logit_scale * text_embeddings @ image_embeddings.t())

        return image_logits,text_logits



def find_best_matching_string(input_text, dataset_classes):
    if dataset_classes is None:
        return None
    if input_text in dataset_classes:
        return input_text
    return next((cls_name for cls_name in dataset_classes if cls_name in input_text), None)

              
def CAM(args):
    torch.cuda.set_device(int(args.gpu))
    iterator = CLIPIntereactiveCAM(args.model_name, args.cam_version, args.topk, args.image_src, args.image_folder, args.split, args.shuffle, args.save_folder, args.bin_thres, args.show_box, 
                 resize_image=args.resize_image, custom_clip=args.custom_clip, use_channel_dict=args.use_channel_dict, drop=args.drop, resize=args.resize, checkpoint=args.checkpoint, is_clip=args.is_clip, batch_size=args.batch_size, num_cams=args.num_cams)
    iterator.iterate_images()

  
class CLIPIntereactiveCAM:
    def __init__(self, clip_version, cam_version, topk, image_src, image_folder, split='val', shuffle=True, save_folder='test', bin_thres=1, show_box=False, is_clip=False,
                 resize_image=False, custom_clip=False, use_channel_dict=False, channel_search_path=None, drop=False, resize=False, checkpoint=None, coco_path=None, iterate=True, batch_size=128, num_cams=1, resize_box=False):
        self.clip_version = clip_version
        self.cam_version = cam_version
        self.topk = topk
        self.custom_clip = custom_clip
        self.use_channel_dict = use_channel_dict
        self.drop = drop
        self.resize = resize
        self.image_src = image_src
        self.image_folder = image_folder
        self.resize_image = resize_image
        self.save_folder = save_folder
        self.bin_thres = bin_thres
        self.show_box = show_box
        self.coco_path= coco_path
        self.checkpoint = checkpoint
        self.split = split
        self.iterate = iterate
        self.is_clip = is_clip
        self.batch_size = batch_size
        self.num_cams = num_cams
        self.resize_box = resize_box
        
        self._load_cam()
        self.dataset_classes, self.name2id, self.id2name= get_dataset_class_name(self.image_src)
        print("Loading dataset...")
        self._load_dataset(split, shuffle)
        os.makedirs('cam_temp', exist_ok=True)
        
        if self.use_channel_dict:
            print("Loading channel dictionary...")
            self.channel_dict = CLIP_topk_channels(channel_search_path, cat_name='all', topk=self.topk)

    def _load_cam(self):
        if(args.model_domain == "clip"):
            clip_model, self.preprocess, target_layer, cam_trans, clip = load_clip(self.clip_version, resize=self.resize, custom=self.custom_clip)
            tokenizer = clip.tokenize
            if self.checkpoint is not None:
                clip_model = load_clip_from_checkpoint(self.checkpoint, clip_model)
                for i,p in clip_model.named_parameters():
                    print(i,p.size())
        
        elif(args.model_domain == "biomedclip"):
            _, _, _, cam_trans, clip = load_clip(self.clip_version, resize=self.resize, custom=self.custom_clip)
            biomedclip_model, self.preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            clip_model = BiomedCLIP(biomedclip_model)
            target_layer = clip_model.model.visual.trunk.blocks[11].norm2
            
        cam_versions = self.cam_version.split(',')
        self.cam = []

        for cam_version in cam_versions:   
            self.cam.append(CAMWrapper(clip_model, target_layers=[target_layer], tokenizer=tokenizer, cam_version=cam_version, preprocess=self.preprocess, topk=self.topk,
                            drop=self.drop, channels={}, cam_trans=cam_trans, is_clip=self.is_clip, batch_size=self.batch_size,model_domain=args.model_domain, num_cams=self.num_cams))

    # load dateset into class methods
    def _load_dataset(self, split: str, shuffle: bool) -> None:
        if self.image_src in ['coco', 'imagenet', 'lvis_filtered_6_classes', 'lvis_filtered_12_classes', 'lvis_filtered_1_class']:
            self.meta_data = pd.read_hdf(f'meta_data/{self.image_src}_{split}_instances_stats.hdf5', 'stats')
            target_class = input('Specify a class or loop though all images?\n')
            if target_class.isnumeric():
                target_class = int(target_class)
            elif target_class in self.dataset_classes:
                target_class = self.name2id[target_class]
            else:
                print('Target class not found, using all classes')
            
            if isinstance(target_class, int):
                self.meta_data = self.meta_data[self.meta_data['class_id'] == target_class]

            image_list = self.meta_data['image_id'].values.tolist()
            
        elif self.image_src == 'parts_imagenet':
            self.meta_data = pd.read_hdf('meta_data/partsImageNet_parts_test.hdf5', 'stats')
            target_class = input('Specify a class or loop though all images?\n')
            if target_class.isnumeric():
                target_class = int(target_class)
            elif target_class in self.dataset_classes:
                target_class = self.name2id[target_class]
            else:
                print('Target class not found, using all classes')
            if isinstance(target_class, int):
                self.meta_data = self.meta_data[self.meta_data['class_id'] == target_class]
            image_list = self.meta_data['image_id'].unique().tolist()
        else:
            image_list = os.listdir(self.image_folder)    
        
        if shuffle:
            random.shuffle(image_list)
        self.image_list = image_list
        self.img_iter = iter(image_list)
        
    def imageID2Path(self, image_id: int) -> str:
        if self.image_src == 'coco':
            return f'{self.image_folder}/{image_id:012d}.jpg'
        elif self.image_src == 'imagenet':
            return f'{self.image_folder}/{image_id:08d}.jpg'
        elif self.image_src.startswith('lvis'):
            image_path = self.meta_data[self.meta_data.image_id == image_id].image_path.values[0]
            return f'{self.image_folder}/{image_path}'
        else:
            return f'{self.image_folder}/{image_id}'
        
    def iterate_images(self):
        gt_boxes = None
        if(args.one_by_one == False):
            input_text = input("Enter text prompt for all images:\n")
        else:
            input_text = None
        while True:
            if self.iterate:
                try:
                    image_id = next(self.img_iter)
                except StopIteration:
                    break
            else:
                image_id = input('Which image(id) are we looking for?')
                try:
                    image_id in self.image_list
                except Exception:
                    print('Image not found, please check the ID is correct or dataset.')
                    continue

            image_path = self.imageID2Path(image_id)
            raw_image = Image.open(image_path)
            if raw_image.mode != 'RGB':
                raw_image = raw_image.convert('RGB')
            raw_image_size = raw_image.size
            if self.resize_image: # resize raw image
                crop_size = self.preprocess.transforms[1].size
                input_image = raw_image.resize((crop_size, crop_size))
            else:
                input_image = raw_image

            self._intereact_prompt(input_image, raw_image_size, image_id,input_text)

    def display_image(self, input_image, cams, gt_bboxes, image_id, raw_image_size):
        cam_imgs = [Drawer.overlay_cam_on_image(input_image, cam, use_rgb=True) for cam in cams]
        bboxes = [graycam2bb(cam, thresh_val=self.bin_thres) for cam in cams]
        if self.show_box:
            box_images = []
            for cam_img, box in zip(cam_imgs, bboxes):
                if self.resize_box:
                    from tools.iou_tool import resize_box
                    gt_box = resize_box(gt_box, box_size=self.box_size, target_size=input_image.size)
                box_img = Drawer.draw_boxes(cam_img, [box], color='red')
                if len(gt_bboxes) != 0:
                    box_img = Drawer.draw_boxes(box_img, gt_bboxes, color='green')
                box_images.append(box_img)
            final_cam_images = box_images
        else:
            final_cam_images = cam_imgs

        cat_image = Drawer.concat(final_cam_images)
        cat_image.save(f'{self.save_folder}/{image_id}')
        return cat_image
        
    def print_cam_info(self, input_image: Image, input_text: str) -> Tuple[str, float]:
        output_dict = {'Input text': input_text}
        if(args.model_domain == "clip"):
            clip_logits = self.cam[0].getLogits(input_image, input_text)
            score = clip_logits[1].detach().cpu().item()
            output_dict['CLIP Score'] = score
            temp_text = input_text.replace(' ', '_')
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(output_dict)
            return temp_text, score
        elif(args.model_domain == "biomedclip"):
            temp_text = input_text.replace(' ', '_')
            pp = pprint.PrettyPrinter(indent=4)
            return temp_text, 0.0

    # get gt boxes from hdf5 file
    def get_gt_boxes(self, input_text: str, image_id: int):
        intances = self.meta_data[self.meta_data['image_id'] == image_id]
        mapped_text = find_best_matching_string(input_text, self.dataset_classes)
        try:
            target_class_id = self.name2id[mapped_text]
            gt_boxes = intances[intances['class_id'] == target_class_id][['x', 'y', 'w', 'h']].to_numpy()
            return [xywh2xyxy(box) for box in gt_boxes]
        except Exception:
            print("No gt boxes found for this image.")
            return []
        
                
    def _intereact_prompt(self, input_image, raw_image_size, image_id,input_text=None):
        input_size = input_image.size
        if(args.one_by_one):
            print('Press n for next image. Press x to terminate')
            while True:
                input_text = input("Type something in the image:\n")
                if input_text == 'n':
                    break
                elif input_text == 'x':
                    quit()
                elif input_text == 'save_img':
                    cat_image.save(f'{self.save_folder}/{self.cam_version}/{image_id}_{temp_text}.jpg')
                elif input_text == 'save_raw':
                    cam_imgs = [Drawer.overlay_cam_on_image(input_image, cam, use_rgb=True) for cam in grayscale_cams]
                    name = input('What is the name of the image?\n')
                    cam_imgs[0].save(f'results/test_img/{name}_{temp_text}.jpg')
                else:
                    gt_bboxes = self.get_gt_boxes(input_text, image_id) if self.show_box else []
                    # get cam
                    grayscale_cams = self._get_cams(input_image, input_text, input_size)

                    # display interact with image
                    cat_image = self.display_image(input_image, grayscale_cams, gt_bboxes, image_id, raw_image_size)

                    # print out text info.
                    temp_text, score = self.print_cam_info(input_image, input_text)
        else:
            gt_bboxes = self.get_gt_boxes(input_text, image_id) if self.show_box else []
            # get cam
            grayscale_cams = self._get_cams(input_image, input_text, input_size)

            # display interact with image
            cat_image = self.display_image(input_image, grayscale_cams, gt_bboxes,image_id, raw_image_size)

            # print out text info.
            temp_text, score = self.print_cam_info(input_image, input_text)

    def _get_cams(self, input_image, input_text, input_size):
        return [cam((input_image, input_text), 0, input_size,args.model_domain) for cam in self.cam]
                
        
        
def get_parser():
    parser = argparse.ArgumentParser()
    # dataset setting
    parser.add_argument('--image-src', type=str,
                        default='custom',
                        help="The dataset name of images")
    parser.add_argument('--image-folder', type=str, default='/home/pzc0018@auburn.edu/dataset/COCO2017/val2017',
                        help="The root folder for images")
    parser.add_argument('--shuffle', action='store_false',
                        help="Whether to shuffle the data")
    parser.add_argument('--resize-image', action='store_true',
                        help="Whether to resize the image before any operation.")
    parser.add_argument('--split', type=str, default='val',
                    help="One of [val, test]. They correspond to ")
    
    # cam setting
    parser.add_argument('--cam-version', type=str, default='gscorecam',
                    help="Specify the cam version")
    parser.add_argument('--drop', action='store_true',
                        help="Whether to drop the channels in scorecam")
    parser.add_argument('--topk', type=int, default=300,
                        help="Number of channels used by scorecam(with drop) or gscorecam")
    parser.add_argument('--bin-thres', type=float, default=1,
                        help="Binary threshold for cam heatmaps, default 1 means otsu method is used")
    parser.add_argument('--resize', default='adapt',
                        help="Resize method, one of [adapt, crop, pad, none]")
    parser.add_argument('--use-channel-dict', default=False, action='store_true',
                        help='Whether to use channel dict (Best channels for scorecam).')
    parser.add_argument('--batch-size', default=128, type=int,
                        help="Batch size for scorecam based methods.")
    parser.add_argument('--num-cams', default=1, type=int,
                        help="Number of cams to be used.")
    parser.add_argument('--model-domain',type=str, default='clip',help="Whether CLIP or BiomedCLIP is used")

    parser.add_argument('--one-by-one',action='store_true',
                        help="Whether to prompt the user for each image or not.")
    
    # model setting 
    parser.add_argument('--model-name', type=str, default='ViT-B/16',
                        help="The model name.")
    parser.add_argument('--is-clip', action='store_false',
                        help="Whether to use clip model.")
    parser.add_argument('--custom-clip', action='store_true',
                        help="Whether to use custom clip ViT model with custom methods.")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="The checkpoint to load.")
    
    # misc
    parser.add_argument('--gpu', type=int, default=0,
                        help="The gpu id to use")
    parser.add_argument('--save-folder', type=str, default='visualization_samples/clip_cam',
                        help="Output path of visualization method. (use 'save_img' to save image during visualization.)")
    parser.add_argument('--show-box', action='store_true', default=False,
                        help="Whether to show gt boxes in the image.")
    parser.add_argument('--iterate', action='store_false',
                        help='Whether to iterate through the dataset')

    return parser.parse_args()
if __name__ == '__main__':
    args = get_parser()
    CAM(args)