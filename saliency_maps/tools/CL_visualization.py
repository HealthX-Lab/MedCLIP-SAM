import os
import random

def get_dataset_class_name(dataset_name):
    import json
    imagenet_label_mapper = {}
    if dataset_name == 'coco':
        with open ('COCO_CLASSES.json', 'r') as f:
            dataset_classes = json.load(f)
    elif dataset_name == 'imagenet':
        with open('imagenet_labels.json') as f:
            imagenet_label_mapper = json.load(f) 
            imagenet_label_mapper = {key:value[1] for key, value in imagenet_label_mapper.items()}
            dataset_classes = list(imagenet_label_mapper.values())
    else:
        dataset_classes = None
    return dataset_classes, imagenet_label_mapper


class datasetManager:
    def __init(self, dataset_name, CL):
        self.CL = CL
        self.dataset_name = dataset_name
        self.dataset_classes, self.imagenet_label_mapper = get_dataset_class_name(dataset_name)
        self.iterable = True
    
    def get_dataset(self, split='val'):
        if self.dataset_name == 'coco':
            self.dataset = self.cocoManager(split)    
        elif self.dataset_name == 'nao':
            self.dataset = self.naoManager(split)
        elif self.dataset_name == 'imagenet':
            self.dataset = self.ImageNetManager()
        elif self.dataset_name == 'refcoco':
            self.dataset = self.refcocoManager(split)
        else:
            self.dataset = self.imageFolderManager(self.dataset_name)
    
    def cocoManager(self, split, coco_method='1', target_class=None):
        from dataset import COCOHelper
        coco_method = input('Choose a visualization method:\n1:Go to specific image(from image id).\n2:Loop within a category.\n') if self.CL else coco_method
        if coco_method == '1':
            self.iterable = False
            return COCOHelper(split=split, returnPath=True)
        target_class = input('Please input the target class:\n') if self.CL else target_class
        return COCOHelper(split=split, returnPath=True, class_list=target_class)

    def naoManager(self, split, nao_method='1', target_class=None):
        from dataset import NAO
        nao_method = input("Loop within a specific class (1) or full dataset (2)?\n") if self.CL else nao_method
        if nao_method != '1':
            return NAO(split=split, returnPath=True)

        target_class = input('Please input the target class:\n') if self.CL else target_class
        return NAO(split=split, target_class=target_class, returnPath=True)
    
    def ImageNetManager(self, method='1', target_class=None):
        from wsolevaluation.data_loaders import WSOLImageLabelDataset
        method = input('Loop within a class?\n') if self.CL else method
        if method == '1':
            target_class = input('which class?\n')
        if target_class.isdigit():
            target_class = int(target_class)      
        elif target_class not in self.dataset_classes:
            target_class = None
        return WSOLImageLabelDataset(data_root='wsolevaluation/dataset/ILSVRC', metadata_root='wsolevaluation/metadata/ILSVRC/val' , transform=None, proxy=False, shuffle=True, choose_class=target_class)
        from wsolevaluation.data_loaders import configure_metadata, get_bounding_boxes
        metadata = configure_metadata('wsolevaluation/metadata/ILSVRC/val')
        gt_boxes_dict = get_bounding_boxes(metadata)
        
    def refcocoManager(self,split):
        from refer.refer import REFER
        from dataset import refCOCOBBox
        refer = REFER(data_root='refer/data', dataset='refcocog', splitBy='google')
        return refCOCOBBox(refer, split=split, batch_size=1, shuffle=True)
        
    def imageFolderManager(self, img_folder, shuffle=True):
        img_list = os.listdir(img_folder)
        if shuffle:
            random.shuffle(img_list)
        
    def get_img_iter(self):
        for item in self.dataset:
            if self.dataset_name == 'coco':
                if self.iterable:
                    img_path = item['image']
                    img_id   = item['img_id']
                    gt_boxes = [item['bbox'] for item in item['box_info']]
        
class Visualizer:
    def __init__(self, CL=True):
        self.CL = CL
        
    def get_img_iter(dataset):
        for item in dataset:
        yield img_path, gt_box_dict, target_cls_name