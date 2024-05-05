# evaluation method of InfoGround https://github.com/BigRedT/info-ground
import torch
import numpy as np


def resize_box(box, box_size, target_size):
    x_scale = target_size[0]/box_size[0]
    y_scale = target_size[1]/box_size[1]
    resize_box = [box[0]*x_scale, box[1]*y_scale, box[2]*x_scale, box[3]*y_scale]
    return resize_box

def return_max_iou(pred_box, gt_boxes):
    iou_list = []
    for gt_box in gt_boxes:
        iou = compute_iou(pred_box, gt_box)
        iou_list.append(iou)
    iou_array = np.array(iou_list)
    return iou_array.max(), gt_boxes[iou_array.argmax()]

def xywh2xyxy(box):
    if torch.is_tensor(box[0]):
        box = list(map(lambda x: x.item(), box))
    x,y,w,h = box[0], box[1], box[2]+box[0], box[3]+box[1]
    return [x,y,w,h]

def xyxy2xywh(box):
    return [box[0], box[1], box[2]-box[0], box[3]-box[1]]
    

def compute_recall(pred_boxes,gt_boxes,k=1):
    recalled = [0]*k
    pred_box = [None]*k
    gt_box = [None]*k
    iou_list = []
    for i,pred_box_ in enumerate(pred_boxes):
        if i>=k:
            break

        for gt_box_ in gt_boxes:
            iou = compute_iou(pred_box_,gt_box_)
            iou_list.append(iou)  #! This is problematic when k > 1
            if iou >= 0.5:
                recalled[i] = 1
                pred_box[i] = pred_box_
                gt_box[i] = gt_box_
                break
            
    max_recall = 0
    for i in range(k):
        max_recall = max(recalled[i],max_recall)
        recalled[i] = max_recall

    return torch.tensor(recalled), pred_box, gt_box, max(iou_list)

def get_set_recall(pred_boxes, gt_boxes):
    recalled = [0]*len(pred_boxes)
    iou_list = []
    for i,pred_box in enumerate(pred_boxes):
        max_iou = 0
        for gt_box in gt_boxes:
            iou = compute_iou(pred_box, gt_box)
            max_iou = max(max_iou, iou)
            if iou >= 0.5:
                recalled[i] = 1
                # break
        iou_list.append(max_iou)
    return recalled, iou_list

def compute_area(bbox,invalid=None):
    x1,y1,x2,y2 = bbox

    if (x2 <= x1) or (y2 <= y1):
        area = invalid
    else:
        area = (x2 - x1 + 1) * (y2 - y1 + 1)

    return area

def compute_iou(bbox1,bbox2,verbose=False):
    x1,y1,x2,y2 = bbox1
    x1_,y1_,x2_,y2_ = bbox2
    
    x1_in = max(x1,x1_)
    y1_in = max(y1,y1_)
    x2_in = min(x2,x2_)
    y2_in = min(y2,y2_)

    intersection = compute_area(bbox=[x1_in,y1_in,x2_in,y2_in],invalid=0.0)
    area1 = compute_area(bbox1, invalid=0.0)
    area2 = compute_area(bbox2, invalid=0.0)
    union = area1 + area2 - intersection
    iou = intersection / (union + 1e-6)

    if verbose:
        return iou, intersection, union

    return iou 

def compute_best_candidate(proposals, gt_boxes):
    max_iou = 0
    recallable = False
    max_iou_list = []
    for candidate in proposals:
        iou_list = torch.tensor([compute_iou(candidate, gt_box) for gt_box in gt_boxes])
        if iou_list.max() > 0.5:
            recallable = True
        max_iou_list.append(iou_list.max())
    best_choice = torch.tensor(max_iou_list).topk(1)
    return recallable, best_choice

def choose_best_candidate(proposals, reference, by='iou'):
    if by == 'iou':
        iou_list = torch.tensor([compute_iou(candidate, reference) for candidate in proposals])
        max_iou, choice = iou_list.topk(1)
    elif by == 'center':
        reference_center = [0.5*(reference[0]+reference[2]), 0.5*(reference[1]+reference[3])]
        proposal_center  = [[0.5*(box[0]+box[2]), 0.5*(box[1]+box[3])] for box in proposals]
        
        reference_center = torch.tensor(reference_center).float().unsqueeze(0)
        proposal_center  = torch.tensor(proposal_center).float()
        center_diff      = torch.cdist(proposal_center, reference_center)
        max_iou, choice = center_diff.topk(1, dim=0)
        max_iou = max_iou[0]
        choice  = choice[0]
    return max_iou, choice

def evaluate_choice(socres, proposals, gt_boxes, topk=1, largest=True):
    confidence, chosen_index = socres.topk(topk, largest=largest)
    chosen_boxes = [proposals[i] for i in chosen_index]
    choice_recall, pred_box, gt_box = compute_recall(chosen_boxes, gt_boxes, topk)
    # recallable , best_candidate_iou = compute_best_candidate(proposals, gt_boxes)

    # return choice_recall, recallable, best_candidate_iou
    return choice_recall[topk-1].item()


#     # compute IoU of proposed bb and ground truth bb
# def computeIoU(true_bb, true_bb_type='xyxy', *bb):
#     if len(bb) == 1:
#         bb = bb[0] # unpack bb
#     if true_bb_type != 'xywh':
#         x,y,w,h = true_bb[0], true_bb[1], true_bb[0]+true_bb[2], true_bb[1]+true_bb[3]
#     else:
#         x,y,w,h = true_bb[0], true_bb[1], true_bb[2], true_bb[3]
#     Xa = max(x, bb[0])
#     Xb = min(w, bb[2])
#     Ya = max(y, bb[1])
#     Yb = min(h, bb[3])
#     interArea = max(0, Xb-Xa+1) * max(0, Yb-Ya+1)
#     trueArea  = (w-x+1)*(h-y+1)
#     bbArea    = (bb[2]-bb[0]+1)*(bb[3]-bb[1]+1)
#     IoU = interArea/(trueArea+bbArea-interArea)
#     return IoU