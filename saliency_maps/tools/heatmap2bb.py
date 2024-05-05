""" Get BB from heatmap"""
################################################################################
from tkinter import X
import cv2
import copy
import numpy as np
from skimage.transform import resize
from tools.iou_tool import xywh2xyxy
import torch
from typing import Any, Tuple, List

#* when using this ranking method, make sure the order of (box_list, area_list, intensity_list) is the consistent
class Heatmap2BB(object):
    # when grid size is 1, the result of choices is (intense-base box, area-base box)
    # when alpha = 1, result is area-base box
    # when alpha = 0, result is intense-base box 
    @staticmethod
    def get_pred_boxes(cam: np.ndarray, grid_size: float = 1, to_xyxy: bool= True, alpha: float or None= None, threshold: float = 1.0) -> List[list]:
        contours = Heatmap2BB.get_contours(cam, threshold)
        candidate_boxes, box_areas = Heatmap2BB.get_bounding_boxes(contours, cam, to_xyxy)
        mean_intense = Heatmap2BB.mean_contour_intensity(contours, cam)
        scores = Heatmap2BB.rank_bbs(box_areas, mean_intense, grid_size, alpha)
        return Heatmap2BB.choose_boxes(candidate_boxes, scores)
    
    @staticmethod
    def choose_boxes(boxes: List[list], score: np.ndarray):
        choices = score.argmax(axis=1)
        return [boxes[choice] for choice in choices]
    
    @staticmethod
    def mean_contour_intensity(contours: List, graycam: np.ndarray) -> list:
        if not contours:
            return [1e-3]
        cnt_mean = []
        for cnt in contours:
            mask = np.zeros(graycam.shape, np.uint8)
            cv2.drawContours(mask, [cnt], 0, (1), 1)
            # mask = mask/255
            cnt_cam = mask * graycam
            # cnt_area = cv2.contourArea(cnt) # contourArea sometimes return 0, so replace with mask.sum()
            cnt_area = mask.sum()
            if cnt_area == 0:
                print(cnt_area)
            cnt_mean.append(cnt_cam.sum()/cnt_area)
        return cnt_mean
        
    @staticmethod
    def get_contours(graycam: np.ndarray, threshold: float= 1.0) -> list:
        gray = copy.deepcopy(graycam)
        gray = cv2.normalize(src=gray, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        if threshold == 1.0:
            binary_map = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        else:
            binary_map = cv2.threshold(gray, round(threshold*255), 255, cv2.THRESH_BINARY)[1]
        contours, hierarchy = cv2.findContours(np.uint8(binary_map), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    @staticmethod
    def get_bounding_boxes(contours: list, cam: np.ndarray, to_xyxy: bool = True) -> Tuple[list, list]:
        box_areas = []
        bb_list = []
        cam_size = cam.shape[0] * cam.shape[1]
        if not contours:
            bb_list.append([0, 0, 1, 1]) # set a dummy value in case of no prediction
            box_areas.append(1/cam_size)
        for cntr in contours:
            x,y,w,h = cv2.boundingRect(cntr)
            box_areas.append(w*h/cam_size)
            bb_loc = [x,y, x+w, y+h] if to_xyxy else [x, y, w, h]
            bb_list.append(bb_loc)
        return (bb_list, box_areas)
    
    @staticmethod
    def rank_bbs(box_areas: list, intensity: list, grid_size: float, alpha: float or None) -> np.ndarray:
        alpha = np.arange(0, 1+grid_size, grid_size).reshape((-1, 1)) if alpha is None else np.array([alpha]).reshape((-1, 1))
        areas = np.array(box_areas).reshape((1, -1))
        intense = np.array(intensity).reshape((1, -1))
        # areas = - 0.1 / np.log(areas+1e-6)
        # intense = - 0.1 / np.log(intense)
        return alpha * areas + (1-alpha) * intense
    
    #! EXPERIMENTAL: scale intensity and area such that they are comparable
    @staticmethod
    def _get_pred_boxes_experimental(cam: np.ndarray, grid_size: float = 1, to_xyxy: bool= True, alpha: float or None= None, threshold: float = 1.0) -> List[list]:
        contours = Heatmap2BB.get_contours(cam, threshold)
        candidate_boxes, box_areas = Heatmap2BB.get_bounding_boxes(contours, cam, to_xyxy)
        mean_intense = Heatmap2BB.mean_contour_intensity(contours, cam)
        scores = Heatmap2BB._rank_bbs_experimental(box_areas, mean_intense, grid_size, alpha)
        return Heatmap2BB.choose_boxes(candidate_boxes, scores)
    @staticmethod
    def _rank_bbs_experimental(box_areas: list, intensity: list, grid_size: float, alpha: float or None) -> np.ndarray:
        alpha = np.arange(0, 1+grid_size, grid_size).reshape((-1, 1)) if alpha is None else np.array([alpha]).reshape((-1, 1))
        areas = np.array(box_areas).reshape((1, -1))
        intense = np.array(intensity).reshape((1, -1))
        areas = - 0.1 / np.log(areas+1e-6)
        intense = - 0.1 / np.log(intense)
        return alpha * areas + (1-alpha) * intense
    
        
def graycam2bb(graycam, thresh_val=0.5, num_bbs=1, to_xyxy=True, return_mask=False):
    if thresh_val == 1:
        thresh_val = 'ostu'    
    gray = copy.deepcopy(graycam)
    gray = cv2.normalize(src=gray, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    if thresh_val == 'ostu':
        threshold, binary_map  = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        threshold, binary_map = cv2.threshold(gray, round(thresh_val*255), 255, cv2.THRESH_BINARY)
    
    contours, hierarchy = cv2.findContours(np.uint8(binary_map), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    box_areas = []
    bb_list = []
    if len(contours) == 0:
        bb_list.append([0, 0, 1, 1]) # set a default value in case of no prediction
        box_areas.append(1)
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        box_areas.append(w*h)
        if to_xyxy:
            bb_loc = [x,y, x+w, y+h]
        else:
            bb_loc = [x, y, w, h]
        bb_list.append(bb_loc)
    if num_bbs == 1:
        max_idx = np.array(box_areas).argmax()
        bb_loc = bb_list[max_idx]
        rt_box = bb_loc
    else:
        rt_box = bb_list
    
    if return_mask:
        return rt_box, binary_map
    else:
        return rt_box  
    
def binary_area(graycam, thresh_val=1, box=None, in_persantage=True):
    if thresh_val == 1:
        thresh_val = 'ostu' 
    gray = copy.deepcopy(graycam)
    gray = cv2.normalize(src=gray, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    if thresh_val == 'ostu':
        binary_map = cv2.threshold(gray,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    else:
        binary_map = cv2.threshold(gray, round(thresh_val), 1, cv2.THRESH_BINARY)[1]
    # binary_map = binary_map.astype(bool).astype(np.uint8)
    if box is not None:
        box = box.astype(int)
        area = binary_map[box[1]:box[3], box[0]:box[2]].sum()
        total = (box[2]-box[0])*(box[3]-box[1])
    else:
        total = binary_map.shape[0]*binary_map.shape[1]
        area = binary_map.sum()
    return area if not in_persantage else area/total

def heatmap_density(graycam, boxes=None, in_persantage=True):
    gray = copy.deepcopy(graycam)
    if gray.max() > 1 or gray.min() < 0:
        heatmap = cv2.normalize(src=gray, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    else:
        heatmap = gray
    density = []
    if boxes is None:
        boxes = [[0, 0, heatmap.shape[1], heatmap.shape[0]]]

    for box in boxes:
        box = box.astype(int)
        area = heatmap[box[1]:box[3], box[0]:box[2]].sum()
        total = (box[2]-box[0])*(box[3]-box[1])
        if in_persantage:
            density.append(area/total)
        else:
            density.append(area)

    return torch.tensor(density)

def stable_sigmoid(x):
    return np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))

def hila_heatmap_transform(graycam: np.ndarray, model_input_size: tuple, raw_image_size: tuple):
    gray = copy.deepcopy(graycam)
    if gray.dtype == np.float16:
        gray = gray.astype(float)
    # normalize
    gray = cv2.normalize(src=gray, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    # Otsu (line 7)
    binary_map = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    # resize to size of model input (line 8)
    resized_map = cv2.resize(binary_map, model_input_size, interpolation=cv2.INTER_LINEAR)
    # sigmoid (line 9)
    resized_map = stable_sigmoid(resized_map)
    resized_map = np.where(resized_map > 0.5, resized_map, 0)
    # resize to original image input (line 10)
    resized_map = cv2.resize(binary_map, model_input_size, interpolation=cv2.INTER_NEAREST)
    
    return resized_map

def heatmap2binary(x: np.ndarray):
    if x.dtype == np.float16:
        x = x.astype(float)
    return cv2.normalize(src=x, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)