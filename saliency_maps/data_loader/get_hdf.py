import pandas as pd
import json
from torchvision.datasets.imagenet import ImageNet
from tqdm import tqdm
import os

def get_imagenet_meta(image_folder: str = '/home/peijie/dataset/ILSVRC2012', split='train', include_box=False) -> pd.DataFrame:
    import xml.etree.ElementTree as ET
    id2name = json.load(open('meta_data/imagenet_id2name.json'))
    dataset = ImageNet(image_folder, split)
    meta_data = []
    skip = 0
    for (image, class_id) in tqdm(dataset.imgs):
        class_name = id2name[str(class_id)]
        file_name = "/".join(image.split('/')[-2:])
        image_id = image.split('/')[-1].split('.')[0]
        # get box
        if split == 'train':
            wnid = image_id.split('_')[0]
            box_path = f'{image_folder}/{split}_box/{wnid}/{image_id}.xml'
            if not os.path.exists(box_path):
                skip += 1
                continue
            tree = ET.parse(box_path)
            root = tree.getroot()
            image_width = root[3][0].text
            image_height = root[3][1].text
            for child in root:
                if child.tag == 'object':
                    x_min, y_min, x_max, y_max = child[4][0].text, child[4][1].text, child[4][2].text, child[4][3].text
                    x, y, w, h = int(x_min), int(y_min), int(x_max) - int(x_min), int(y_max) - int(y_min)
                    meta_data.append({'image_id': image_id, 'class_id': class_id, 'class_name': class_name, 'x': x, 'y': y, 'w': w, 'h': h, 'image_width': image_width, 'image_height': image_height, 'file_name': file_name})
    print(f'{skip} images skipped')
    return pd.DataFrame.from_dict(meta_data)


def get_imagenet_bb(bb_folder: str = '/home/peijie/dataset/ILSVRC2012/val_bbox'):
    import xml.etree.ElementTree as ET
    file_list = os.listdir(bb_folder)
    label_mapper = pd.read_json('imagenet_labels.json', orient='index')
    label_mapper.columns = ['wid', 'class_name']
    instances = []
    for file_name in tqdm(file_list):
        tree = ET.parse(f'{bb_folder}/{file_name}')
        root = tree.getroot()
        image_id = int(file_name.split('.')[0].split('_')[-1])
        image_width = root[3][0].text
        image_height = root[3][1].text
        for child in root:
            if child.tag == 'object':
                class_wid = child[0].text
                class_id = label_mapper[label_mapper.wid == class_wid].index.item()
                class_name = label_mapper.iloc[class_id].class_name.replace('_', " ")
                x_min, y_min, x_max, y_max = child[4][0].text, child[4][1].text, child[4][2].text, child[4][3].text
                x, y, w, h = int(x_min), int(y_min), int(x_max) - int(x_min), int(y_max) - int(y_min)
                instances.append({'image_id': image_id, 'class_id': class_id, 'class_name': class_name, 'x': x, 'y': y, 'w': w, 'h': h, 'image_width': image_width, 'image_height': image_height})
    return pd.DataFrame.from_dict(instances)
    

def get_imagene_toaster_meta(image_folder: str = '/home/peijie/dataset/adversarial_patches', box_path: str = ''):
    from tools.utils import get_all_files
    file_paths, _ = get_all_files(image_folder, suffix='jpeg')
    
        
# get subset of coco like dataFrame
def get_subset(meta: pd.DataFrame, num_of_classes: int, num_of_same_classes: int, num_of_images: int = 500):

    num_of_objects = num_of_classes * num_of_same_classes
    avaliable_meta = meta[(meta.num_classes == num_of_classes) & (meta.num_same_class <= num_of_same_classes) & (meta.num_objects <= num_of_objects)]
    if len(avaliable_meta.image_id.unique()) > num_of_images:
        return meta[meta.image_id.isin(avaliable_meta.image_id.drop_duplicates().sample(num_of_images, random_state=123456))] #* for reproducibility
    return avaliable_meta