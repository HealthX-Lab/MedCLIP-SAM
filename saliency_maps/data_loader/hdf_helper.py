from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision
import pandas as pd
from PIL import Image

class HDF5Dataset(Dataset):
    def __init__(self, hdf_path: str, image_folder: str, dataset_name: str, hdf_key: str = 'stats', preprocess: None or torchvision.transforms.Compose = None): 
        self.hdf_path = hdf_path
        self.image_folder = image_folder
        self.dataset_name = dataset_name
        self.meta_data = pd.read_hdf(hdf_path, hdf_key)
        

def id2file(image_id: int, dataset_name: str, image_folder: str) -> str: 
    if dataset_name in {'coco', 'lvis'}:
        return f'{image_folder}/{image_id:012d}.jpg'
    elif dataset_name == 'imagenet':
        wnetid = image_id.split("_")[0]
        return f'{image_folder}/{wnetid}/{image_id}.JPEG'
    else:
        raise NotImplementedError

class List2Dataset(Dataset):
    def __init__(self, image_list: list, image_folder: str, dataset_name: str, preprocess: None or torchvision.transforms.Compose = None):
        self.image_list = image_list
        self.image_folder = image_folder
        self.dataset_name = dataset_name
        self.preprocess = preprocess
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image_id = self.image_list[idx]
        image_path = id2file(image_id, self.dataset_name, self.image_folder)
        image = Image.open(image_path).convert('RGB')
        if self.preprocess is not None:
            image = self.preprocess(image)
        return image_id, image


