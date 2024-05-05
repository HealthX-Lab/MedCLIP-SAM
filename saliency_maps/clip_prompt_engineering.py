from tools.prompt_engineering import get_engineered_prompts
from model_loader.clip_loader import load_clip
from data_loader.hdf_helper import List2Dataset
from torch.utils.data import DataLoader
from tools.misc import get_dataset_class_name
from evaluation.misc import Counter
from tqdm import tqdm
from PIL import Image
import pandas as pd
import json
import argh
import torch
import os

def prompt_engineering(model_name:str ='RN50x16', 
                       dataset: str ='coco', 
                       meta_data: str ='meta_data/coco_train_instances_stats.hdf5', 
                       image_folder: str ='/home/pzc0018@auburn.edu/dataset/COCO2017/train2017',
                       candidate_prompts: int =15,
                       sample_size: int =1000,
                       output_path: str ='data/engineered_prompts',):
    model, preprocess, _, _, clip = load_clip(model_name)
    model.eval()
    model.cuda()

    dataset_classes, name2id, id2name = get_dataset_class_name(dataset)
    prompt_dicts = {class_name: get_engineered_prompts(class_name, candidate_prompts, synonym_frist=False, openai_templates=11) for class_name in dataset_classes}
    
    
    meta_data = pd.read_hdf(meta_data, 'stats')
    grouped_data = meta_data.groupby('class_id')
    counter = Counter(dataset_classes, candidate_prompts)
    ordered_class_names = []
    # prompt engineering
    for class_id, class_images in tqdm(grouped_data, desc='Prompt Engineering', total=len(dataset_classes)):
        class_name = id2name[class_id]
        ordered_class_names.append(class_name)
        prompt_candidates = prompt_dicts[class_name]
        image_id_list = class_images.sample(sample_size).image_id.tolist() if len(class_images) >= sample_size else class_images.image_id.tolist()
        # get the score for each candidates
        list_dataset = List2Dataset(image_id_list, image_folder, dataset, preprocess)
        list_loader = DataLoader(list_dataset, batch_size=64, shuffle=False, num_workers=8)
        prompt_tokens = clip.tokenize(prompt_candidates).cuda()
        for image_ids, images in list_loader:
            with torch.no_grad():
                logit_per_image, _ = model(images.cuda(), prompt_tokens)
            scores = logit_per_image.cpu().detach().numpy().astype(float).mean(axis=0)
            counter.update(class_name, scores)
    highest_score_idx = counter.recalls.argmax(axis=1)
    
    best_prompts = {class_name: prompt_dicts[class_name][highest_score_idx[i]] for i, class_name in enumerate(ordered_class_names)}
    os.makedirs(output_path, exist_ok=True)
    json.dump(best_prompts, open(f'{output_path}/{model_name}_{dataset}_n{candidate_prompts}.json', 'w'))

parser = argh.ArghParser()
parser.add_commands([prompt_engineering,
                    ])

if __name__ == '__main__':
    argh.dispatch(parser)