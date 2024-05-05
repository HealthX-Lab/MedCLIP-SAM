import pandas as pd
import ast
import json

def get_dataset_class_name(dataset_name):
    import json
    imagenet_label_mapper = {}
    if dataset_name == 'coco':
        mapper = pd.read_csv('meta_data/coco2lvis.csv', index_col=0)
        id2name = dict(zip(mapper.coco_class_id.tolist(), mapper.coco_class_name.tolist()))
        name2id = dict(zip(mapper.coco_class_name.tolist(), mapper.coco_class_id.tolist()))
        dataset_classes = mapper.coco_class_name.tolist()
    elif dataset_name.startswith('lvis'):
        mapper = pd.read_json('meta_data/LVIS_clsID2clsName.json', orient='index')
        id2name = mapper.to_dict()[0]
        dataset_classes = list(id2name.values())
    elif dataset_name in ['imagenet', 'toaster_imagenet']:
        mapper = pd.read_json('meta_data/imagenet_id2name.json', orient='index')
        id2name = mapper.to_dict()[0]
        dataset_classes = mapper[0].tolist()

    elif dataset_name == 'parts_imagenet':
        with open('meta_data/partsImageNet_categories.csv') as f:
            mapper = pd.read_csv(f, index_col=0)
            id2name = dict(zip(mapper.cat_id.tolist(), mapper.cat_name.tolist()))
            dataset_classes = mapper.cat_name.tolist()
    else:
        print(f"Cannot find dataset class name from source {dataset_name}.")
        dataset_classes = None
        name2id = {}
        id2name = {}
    if dataset_classes is not None:
        name2id = dict(zip(id2name.values(), id2name.keys()))
    return dataset_classes, name2id, id2name

def get_coco_class_names():
    return ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

def get_catID2clsName(dataset='coco'):
    import json
    if dataset == 'coco':
        with open('COCO_catID2cls_name.json', 'r') as f:
            catID2cls_name = json.load(f)
    return catID2cls_name

def filter_logs_from_dict(log_file, in_dataframe=True):
    """[keep dictionary in txt file]

    Args:
        log_file ([str]): [file of text log]

    Returns:
        [pd dataframe]: [DataFrame that contains the log dax    ta]
    """
    data_list = []
    dict_string = ''
    dict_add = False
    with open(log_file, 'r') as f:
        for one_line in f.read().splitlines():
            if ('{' in one_line) and ('}' in one_line):
                data_list.append(ast.literal_eval(one_line))
                continue
            if '{' in one_line:
                dict_add = True
            if dict_add:     
                dict_string += one_line
            if '}' in one_line:
                dict_add = False
                data_list.append(ast.literal_eval(dict_string))
                dict_string = ''
    return (pd.DataFrame.from_dict(data_list) if in_dataframe else data_list)
    
def get_prompt_dict_from_dataframe(dataframe, out_path, key='class', value='prompt'):
    if isinstance(dataframe, str):
        data = pd.read_csv(dataframe, index_col=0)
    else:
        data = dataframe
    keys = data[key].tolist()
    values = data[value].tolist()
    prompt_dict = dict(zip(keys, values))
    with open(out_path, 'w') as f:
        json.dump(prompt_dict, f)
    
