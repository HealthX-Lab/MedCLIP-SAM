from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
from tqdm import tqdm
import torch
import argh
import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import copy
sys.path.append('/home/peijie/phrase_grounding/clip_retrieval')
from refer.refer import REFER
from dataset import Flickr30k, refCOCOBBox
from dataloader_oneStage import ReferDataset
import tensorflow_hub as hub


import tensorflow as tf
def load_img(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=3)
  return img

def load_OI_detector(model_name='mobilenet'):
    if model_name == 'IRV2':
        module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1" #@param ["https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1", "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"]
    elif model_name == 'mobilenet':
        module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
    else:
        raise Exception("Model not found.")
    detector = hub.load(module_handle).signatures['default']
    return detector

def get_OI_proposals(detector, img_path):
    img = load_img(img_path)
    converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    result = detector(converted_img)
    result = {key:value.numpy() for key,value in result.items()}
    proposals = result['detection_boxes']
    # ipdb.set_trace()
    return proposals

# def get_proposal_box(detector_name, detector, trans, img_path):
#     if detector_name == 'yolo':
#         detector_result = detector(img_path)
#         bbox_frame = detector_result.pandas().xyxy[0]
#         if len(bbox_frame) == 0: #skip
#             continue
#         proposal_box = get_yolo_proposals(bbox_frame)
#     elif detector_name in ['IRV2', 'mobilenet']:
#         proposal_box = proposal_list[idx]
#         if not isinstance(proposal_box, list):
#             proposal_box = proposal_box.tolist()
#     elif detector_name == 'fasterrcnn':
#         img_tensor = img_trans(img_path).unsqueeze_(0)   

def get_OI_detector_BBs(detector_name='IRV2', dataset='flickr30k', split='test', splitBy='unc', gpu='4'):
    print('Saving all proposals...')
    if dataset == 'flickr30k':
        dataloader = DataLoader(Flickr30k(split=split), batch_size=1, num_workers=10, shuffle=False)
    elif dataset in ['refclef', 'refcoco', 'refcoco+', 'refcocog']:
        refer = REFER(data_root='refer/data', dataset=dataset, splitBy=splitBy)
        dataloader = DataLoader(refCOCOBBox(refer, split=split), batch_size=1, num_workers=10, shuffle=False)
    elif dataset == 'referit':
        dataloader = ReferDataset(dataset=dataset, split='test')
    detector = load_OI_detector(model_name=detector_name)
    proposal_list = []
    with tf.device(f'/gpu:{gpu}'):
        for idx, (img_path, true_bb, description) in enumerate(tqdm(dataloader)):
            if dataset != 'referit':
                img_path = img_path[0]
            # google IRV2 box are in [ymin, xmin, ymax, xmax] format
            IRV2_box = get_OI_proposals(detector, img_path)
            proposal_box = []
            img_org = Image.open(img_path)
            im_width, im_height = img_org.size
            for box in IRV2_box:
                ymin, xmin, ymax, xmax =  box[0]*im_height, box[1]*im_width, box[2]*im_height, box[3]*im_width
                raw_bb = [xmin, ymin, xmax, ymax]
                int_bb = list(map(float, raw_bb))
                proposal_box.append(int_bb)
            proposal_list.append(proposal_box)
        torch.save(proposal_list, f'datasets/{dataset}_{detector_name}_{split}_BB.pt')

parser = argh.ArghParser()
parser.add_commands([get_OI_detector_BBs,
                    ])
if __name__ == '__main__':
    argh.dispatch(parser)
    # print("anything")