import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO
import os
import pandas as pd
import json


def save_partsImageNet_annotation(annotation_dir: str="/home/qi/PartsImageNet/", split: str="test") -> pd.DataFrame:
    
    partsImageNet_annotation_file_path = os.path.join(annotation_dir, f"{split}.json")
    coco_annotation = COCO(annotation_file=partsImageNet_annotation_file_path)
    
    with open('/home/qi/gscorecam/qi_codes/imagenet_label_to_wordnet_synset.txt', 'r') as f:
        data = f.read()
    
    js = json.loads(data)

    parts_list = []
    for id, extent in coco_annotation.anns.items():
        # print(extent)
        object_id = extent['id']
        class_id = extent['category_id']
        object_size = extent['area']
        image_id = extent['image_id']
        x, y, w, h = extent['bbox']
        img_file_name = coco_annotation.imgs[image_id]['file_name']

        parts_list.append({
                'object_id': object_id, 
                'class_id': class_id,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'object_size':object_size, 
                'image_id': image_id, 
                'file_name': img_file_name,
     })

    parts_df = pd.DataFrame(parts_list)
    parts_df.to_hdf(f"partsImageNet_parts_{split}.hdf5", key="stats", mode="w")

        
def save_partsImageNet_category(annotation_dir: str="/home/qi/PartsImageNet/", split: str="test") -> pd.DataFrame:
    partsImageNet_annotation_file_path = os.path.join(annotation_dir, f"{split}.json")
    coco_annotation = COCO(annotation_file=partsImageNet_annotation_file_path)

    cat_id = []
    cat_name = []
    super_cat_name = []
    for id, cat in coco_annotation.cats.items():
        cat_id.append(cat['id'])
        cat_name.append(cat['name'])
        super_cat_name.append(cat['supercategory'])

    cat_dict = {'cat_id': cat_id, 'cat_name': cat_name, 'super_cat_name': super_cat_name}
    df = pd.DataFrame(cat_dict)
    df.to_csv("partsImageNet_categories.csv", index=True)




def save_samples():

    coco_annotation_file_path = "/home/qi/PartsImageNet/test.json"

    coco_annotation = COCO(annotation_file=coco_annotation_file_path)

    # Category IDs.
    cat_ids = coco_annotation.getCatIds()
    print(f"Number of Unique Categories: {len(cat_ids)}")
    print("Category IDs:")
    print(cat_ids)  # The IDs are not necessarily consecutive.

    # All categories.
    cats = coco_annotation.loadCats(cat_ids)
    cat_names = [cat["name"] for cat in cats]
    print("Categories Names:")
    print(cat_names)

    for cat_id in cat_ids:
        # Get the ID of all the images containing the object of the category.
        img_ids = coco_annotation.getImgIds(catIds=[cat_id])
        print(f"Number of Images Containing {cat_names[cat_id]}: {len(img_ids)}")

        img_name_list = [line.strip() for line in open("/home/qi/PartsImageNet/img_list.txt", "r").readlines()]

        for img_id in img_ids:
            # Pick one image.
            # img_id = img_ids[2]
            img_info = coco_annotation.loadImgs([img_id])[0]
            img_file_name = img_info["file_name"]
            class_name = img_file_name.split("_")[0]
            img_path = os.path.join(f"/home/qi/PartsImageNet/test/{class_name}", img_file_name)

            if img_file_name in img_name_list:
                print(
                    f"Image ID: {img_id}, File Name: {img_file_name}, Image name: {img_file_name}"
                )            # Get all the annotations for the specified image.
                ann_ids = coco_annotation.getAnnIds(imgIds=[img_id], iscrowd=None)
                anns = coco_annotation.loadAnns(ann_ids)
                print(f"Annotations for Image ID {img_id}:")
                print(anns)
                forelet_anns = []
                for ann in anns:
                    if ann['id'] in [10750, 10751]:
                        forelet_anns.append(ann)
                    # Use URL to load image.
                    im = Image.open(img_path)

                    # Save image and its labeled version.
                    plt.axis("off")
                    plt.imshow(np.asarray(im))
                    plt.savefig(f"{img_file_name.split('.')[0]}.jpg", bbox_inches="tight", pad_inches=0)
                    # Plot segmentation and bounding box.
                
                    coco_annotation.showAnns([ann], draw_bbox=False)
                    plt.savefig(f"{ann['id']}_{img_file_name.split('.')[0]}_{ann['category_id']}_annotated.jpg", bbox_inches="tight", pad_inches=0)
                    plt.close()
                
                if len(forelet_anns) >1:
                    im = Image.open(img_path)

                    # Save image and its labeled version.
                    plt.axis("off")
                    plt.imshow(np.asarray(im))
                    plt.savefig(f"{img_file_name.split('.')[0]}.jpg", bbox_inches="tight", pad_inches=0)
                    # Plot segmentation and bounding box.
                
                    coco_annotation.showAnns(forelet_anns, draw_bbox=False)
                    plt.savefig(f"forelegs_{img_file_name.split('.')[0]}_annotated.jpg", bbox_inches="tight", pad_inches=0)

def load_imagenet_labels(json_path: str= '/home/qi/gscorecam/qi_codes/imagenet_labels.json'):

    df_imagenet_labels = pd.read_json(json_path)
    imagenet_labels_dict = {}
    for idx in range(1000):
        imagenet_labels_dict[df_imagenet_labels[idx][0]] = df_imagenet_labels[idx][1]

    # save dict into json
    with open('imagenet_nid_labels.json', 'w') as f:
        json.dump(imagenet_labels_dict, f)

    print('test')



if __name__ == "__main__":

    # save_partsImageNet_annotation(split="test")
    load_imagenet_labels()