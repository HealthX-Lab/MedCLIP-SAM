from PIL import ImageDraw, Image
from PIL.ImageFilter import GaussianBlur
import copy
import torch
import numpy as np
import torchvision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn


import textwrap
from skimage.transform import resize
import cv2
# toPIL = transforms.ToPILImage()

def extend_boxes(boxes, extend_size, img_size):
    extended_boxes = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        xmin = max(xmin - extend_size, 0)
        ymin = max(ymin - extend_size, 0)
        xmax = min(xmax + extend_size, img_size[0])
        ymax = min(ymax + extend_size, img_size[1])
        # xmax = xmax + extend_size if (xmax + extend_size) <= img_size[0] else img_size[0]
        # ymax = ymax + extend_size if (ymax + extend_size) <= img_size[1] else img_size[1]
        extended_boxes.append([xmin, ymin, xmax, ymax])
    return extended_boxes

def zero_out_box(image, box, means=(0.48145466, 0.4578275, 0.40821073)):
    img = copy.deepcopy(image)
    # img_width, img_height = image.size
    boxheight = box[3] - box[1]
    boxwidth = box[2] - box[0]
    zero_array1 = torch.ones((boxheight, boxwidth))*means[0]
    zero_array2 = torch.ones((boxheight, boxwidth))*means[1]
    zero_array3 = torch.ones((boxheight, boxwidth))*means[2]

    zero_array = torch.stack((zero_array1, zero_array2, zero_array3))
    zero_box_img = transforms.ToPILImage()(zero_array)
    img.paste(zero_box_img, box)
    return img

def zero_out_img_tensor(img_tensor, box):
    xmin, ymin, xmax, ymax = box
    img_tensor[:, int(ymin):int(ymax), int(xmin):int(xmax)] = 0
    return img_tensor

def extend_bb(bb, extend_size, img_size):
    x1, y1, x2, y2 = bb
    xmin = x1 - extend_size if (x1 - extend_size) >=0 else 0
    ymin = y1 - extend_size if (y1 - extend_size) >=0 else 0
    xmax = x2 - extend_size if (x2 - extend_size) <= img_size[0] else img_size[0]
    ymax = y2 - extend_size if (y2 - extend_size) <= img_size[1] else img_size[1]
    return [xmin, ymin, xmax, ymax]


def get_yolo_proposals(bb_frame):
    proposals = []
    for bb_idx, row in bb_frame.iterrows():
        proposals.append((row.xmin, row.ymin, row.xmax, row.ymax))
    return proposals

def mask_out_box(image, box, method='gaussian', conf_para=0.25, means=(0.48145466, 0.4578275, 0.40821073)):
    # im = image
    # mask = Image.new('L', im.size, color = 255)
    # draw=ImageDraw.Draw(mask)
    # transparent_area = box
    # draw.rectangle(transparent_area, fill = 0)
    # im.putalpha(mask)
    img = copy.deepcopy(image)
    mask_area = img.crop(box)
    if method == 'Gaussian':
        toPIL = transforms.ToPILImage()
        mask_area_array = np.asarray(mask_area)
        w, h = mask_area.size
        noise = (conf_para * np.random.randn(h, w, 3) * 255).astype('uint8')
        noisy_box_array =  mask_area_array + noise
        masked_box = toPIL(noisy_box_array)
    if method == 'Blur':
        masked_box = mask_area.filter(GaussianBlur(conf_para))
    if method == 'ZeroOut': #NOTE this cannot make the pixel to zero due to the quantization process. Use zero_out_img_tensor for accurate zeroing out.
        boxheight = round(box[3] - box[1])
        boxwidth = round(box[2] - box[0])
        zero_array1 = torch.ones((boxheight, boxwidth))*means[0]
        zero_array2 = torch.ones((boxheight, boxwidth))*means[1]
        zero_array3 = torch.ones((boxheight, boxwidth))*means[2]

        zero_array = torch.stack((zero_array1, zero_array2, zero_array3))
        masked_box = transforms.ToPILImage()(zero_array)
    img.paste(masked_box, (round(box[0]), round(box[1])))
    return img

def get_center(box):
    return torch.tensor([(box[0]+box[2])/2, (box[1]+box[3])/2])

def find_knn_box(target_box, proposals, k=5, by='center'):
    target_center = get_center(target_box)
    proposal_centers = []
    for box in proposals:
        proposals.append(get_center(box))
    
    dist = torch.cdist(target_center, proposal_centers, p=2) # compute eucilidean distance
    return dist.topk(k)

def maskout_bb(img_dir, boxes, conf_para=0.25, preprocess=False, method='blur'):
    img_org = Image.open(img_dir)
    masked_images = []
    for box_ in boxes:
        if method == 'zerotensor':
            if preprocess: 
                img_tensor = preprocess(img_org)  
                masked_img = zero_out_img_tensor(img_tensor, box_)
                masked_images.append(masked_img) 
            else:
                img_tensor = transforms.ToTensor()(img_org)
                masked_img = zero_out_img_tensor(img_tensor, box_)
                masked_images.append(transforms.ToPILImage()(masked_img))
        else:    
            masked_img = mask_out_box(img_org, box_, conf_para=conf_para, method=method)
            if preprocess:
                processed_masked_img = preprocess(masked_img)
                masked_images.append(processed_masked_img)
            else:
                masked_images.append(masked_img)
    return torch.stack(masked_images) if preprocess else masked_images # if preprocess return tensor, otherwise return PIL images
            

def maskout_phrase(sentences, phrases, mask = 'MASK'):
    masked_sentences = []
    for phrase_, sentence_ in zip(phrases, sentences):
        masked_sentences.append(sentence_[0].replace(phrase_[0], mask))
    return masked_sentences

class ObjectDetectors():
    def __init__(self, detector_name):
        from tools.detectron_loader import DetectronModelZoo #! detectron2 does not work on 'clip' env
        from fasterrcnn.fasterrcnn_detector import FasterRCNN
        supported_detectors = ['yolo', 'fasterrcnn_res50_cc', 'fasterrcnn_res101_vg', 'detectron_frn_cc']
        self.detector_name = detector_name
        if detector_name not in supported_detectors:
            raise Exception(f'Detector name not found.\nCurrent avaliable names are: {supported_detectors}')

        print('Loading detector...')
        # load image detector
        if detector_name == 'yolo':
            detector = torch.hub.load('ultralytics/yolov5', 'yolov5x6') # https://github.com/ultralytics/yolov5 for details
        elif detector_name == 'fasterrcnn_res50_cc':
            detector = fasterrcnn_resnet50_fpn(pretrained=True) # trained on COCO
            detector.eval()
            detector.cuda()
            self.img_trans = torchvision.transforms.ToTensor()
        elif detector_name == 'fasterrcnn_res101_vg': # specific for model trained on Visual Genome
            detector = FasterRCNN()
            detector.load_fasterrcnn()
        elif detector_name == 'detectron_frn_cc':
            detector = DetectronModelZoo()
            detector.load_model()
        
        self.detector = detector

    def get_proposals(self, img_path):
        image = Image.open(img_path)
        if self.detector_name == 'yolo':
            detector_result = self.detector(img_path)
            bbox_frame = detector_result.pandas().xyxy[0] # NOTE sometimes there will be no proposals
            proposal_box = get_yolo_proposals(bbox_frame)
            object_names = bbox_frame.name.tolist()
        elif self.detector_name == 'fasterrcnn_res101_vg':
            proposal_box, object_names = self.detector.detect(img_path)
            proposal_box = proposal_box.numpy()
        elif self.detector_name.startswith("detectron"):
            proposal_box, object_names = self.detector.detect(img_path)
        return proposal_box, object_names
    
def get_key_point(box=None, heatmap=None, method='max'):
    if box is not None:
        xmin, ymin, xmax, ymax = box
        key_point = (xmax-xmin)/2, (ymax-ymin)/2
    elif heatmap is not None:
        if method == 'max':
            key_point = np.unravel_index(heatmap.argmax(), heatmap.shape)
        elif method == 'min':
            key_point = np.unravel_index(heatmap.argmin(), heatmap.shape)
    else:
        raise NotImplementedError
    return key_point