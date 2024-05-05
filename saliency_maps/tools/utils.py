import os
import shutil
from pathlib import Path

def getFileList(dir, if_path=True, suffix=None):
    folder_list = os.listdir(dir)
    files = [f for f in folder_list if os.path.isfile(os.path.join(dir, f))]
    if suffix is not None:
       files = [f for f in files if Path(f).suffix == suffix] 
    if if_path:
        return [os.path.join(dir, file) for file in files]
    return files

def save_cocoimg_from_id(traget_folder='data/scorecam_strange_images_withcam', source_folder='/home/peijie/dataset/COCO2017/train2017', save_folder='data/scorecam_strange_images'):
    file_list = getFileList(traget_folder, if_path=False)
    os.makedirs(save_folder, exist_ok=True)
    for file_name in file_list:
        img_id = file_name.split('_')[-1].split('.')[0]
        img_name = f'{int(img_id):012d}.jpg'
        shutil.copyfile(f'{source_folder}/{img_name}', f'{save_folder}/{img_name}')


def get_all_files(dir, returnPath: bool=False, suffix: str=None, fileOnly: bool=True):
    file_list = []
    sub_folders = []
    for (dir_path, dir_names, file_names) in os.walk(dir):
        if returnPath:
            file_list.extend([os.path.join(dir_path, file) for file in file_names if suffix is None or file.endswith(suffix)])
        else:
            file_list.extend([file for file in file_names if suffix is None or file.endswith(suffix)])

        sub_folders.extend([os.path.join(dir_path, folder) for folder in dir_names])
    return file_list, sub_folders