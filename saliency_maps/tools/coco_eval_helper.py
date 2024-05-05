import json

CatID2classname = {"1": "person", "2": "bicycle", "3": "car", "4": "motorcycle", "5": "airplane", "6": "bus", "7": "train", "8": "truck", "9": "boat", "10": "traffic light", "11": "fire hydrant", "13": "stop sign", "14": "parking meter", "15": "bench", "16": "bird", "17": "cat", "18": "dog", "19": "horse", "20": "sheep", "21": "cow", "22": "elephant", "23": "bear", "24": "zebra", "25": "giraffe", "27": "backpack", "28": "umbrella", "31": "handbag", "32": "tie", "33": "suitcase", "34": "frisbee", "35": "skis", "36": "snowboard", "37": "sports ball", "38": "kite", "39": "baseball bat", "40": "baseball glove", "41": "skateboard", "42": "surfboard", "43": "tennis racket", "44": "bottle", "46": "wine glass", "47": "cup", "48": "fork", "49": "knife", "50": "spoon", "51": "bowl", "52": "banana", "53": "apple", "54": "sandwich", "55": "orange", "56": "broccoli", "57": "carrot", "58": "hot dog", "59": "pizza", "60": "donut", "61": "cake", "62": "chair", "63": "couch", "64": "potted plant", "65": "bed", "67": "dining table", "70": "toilet", "72": "tv", "73": "laptop", "74": "mouse", "75": "remote", "76": "keyboard", "77": "cell phone", "78": "microwave", "79": "oven", "80": "toaster", "81": "sink", "82": "refrigerator", "84": "book", "85": "clock", "86": "vase", "87": "scissors", "88": "teddy bear", "89": "hair drier", "90": "toothbrush"}
classID2CatID = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}

def cocoName2ID():
    return {name: int(id) for id, name in CatID2classname.items()}    
    
class JsonCreator:
    def __init__(self, out_path, file_name, format='all'):
        self.out_path = out_path
        self.file_name = file_name
        self.file_list = []
        self.best_list = []
        self.format = format
    
    def add_item(self, item, best=None):
        if best is not None:
            self.best_list.append(best)
        self.file_list.append(item)
    
    @staticmethod
    def xyxy2xywh(box):
        return [box[0], box[1], box[2]-box[0], box[3]-box[1]]
    
    @staticmethod
    def form_result_item(image_name: str, image_id: int, target_name: str, target_id: int, gt_boxes: list, **kwargs) -> dict:
        return {
            'image_name': image_name,
            'image_id': image_id,
            'target_name': target_name,
            'target_id': target_id,
            'gt_boxes': gt_boxes,
            **kwargs
        }    
    
    @staticmethod
    def form_coco_bbox_ann_dict(image_id: int, category_id: int, bbox: list, score: float, format_box: bool = True)-> dict:
        if format_box:
            bbox = JsonCreator.xyxy2xywh(bbox)
        return {
            'image_id': image_id,
            'category_id': category_id,
            'bbox': bbox,
            'score': score,
        }   
    
    @staticmethod
    def form_lvis_bbox_ann_dict(ann_id: int, image_id: int, category_id: int, bbox: list, format_box: bool = True)-> dict:
        if format_box:
            bbox = JsonCreator.xyxy2xywh(bbox)
        return {
            'id': ann_id,
            'image_id': image_id,
            'category_id': category_id,
            'bbox': bbox,
 
        }

    @staticmethod
    def form_coco_segm_ann_dict(image_id: int, category_id: int, segmentation: dict, score: float)-> dict:
        """[form_coco_segm_ann_dict]

        Args:
            image_id (int): [image if of coco dataset]
            category_id (int): [category id of coco dataset]
            segmentation (dict): {"size": [width, height], "counts": COCO binary segmentation}
            score (float): [score]

        Returns:
            dict: [coco format dict]
        """
        return {
            'image_id': image_id,
            'category_id': category_id,
            'segmentation': segmentation,
            'score': score,
        }
    
    def save_json(self):
        with open(f'{self.out_path}/{self.file_name}.json', 'w') as f:
            json.dump(self.file_list, f)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
def cocoEval(pred_file='', coco_cat=[], split='val'):
    annType = ['segm','bbox','keypoints']
    annType = annType[1]
    coco_path='/home/peijie/dataset/COCO2017'
    annFile = f'{coco_path}/annotations/instances_{split}2017.json'
    if split=='test':
        annFile = f'{coco_path}/annotations/image_info_test2017.json'
    img_folder = f'{coco_path}/{split}2017'
    cocoGt = COCO(annFile)

    cocoDt = cocoGt.loadRes(pred_file)
    imgIds=sorted(cocoGt.getImgIds([]))
    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()