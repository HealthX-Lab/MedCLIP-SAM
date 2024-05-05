# Modified from segment-anything repo

import cv2 
from segment_anything import SamPredictor, sam_model_registry
import argparse
import json
import os
from typing import Any, Dict, List
import numpy as np
import torch

_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0

parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--mask-input",
    type=str,
    required=True,
    help="Path to either a single crf mask image or folder of crf mask images.",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help=(
        "Path to the directory where masks will be output. Output will be either a folder "
        "of PNGs per image or a single json with COCO-style masks."
    ),
)

parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)

parser.add_argument(
    "--multi-contour",
    action="store_true",
    help="Whether to evaluate multiple contours in the mask. Default is False.",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

parser.add_argument(
    "--convert-to-rle",
    action="store_true",
    help=(
        "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
        "Requires pycocotools."
    ),
)

def write_mask_to_folder(mask , t_mask, path: str) -> None:
    file = t_mask.split("/")[-1]
    filename = f"{file}"
    cv2.imwrite(os.path.join(path, filename), mask * 255)

    return


def scoremap2bbox(scoremap, threshold, multi_contour_eval=False):
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)
    _, thr_gray_heatmap = cv2.threshold(
        src=scoremap_image,
        thresh=int(threshold * np.max(scoremap_image)),
        maxval=255,
        type=cv2.THRESH_BINARY)
    contours = cv2.findContours(
        image=thr_gray_heatmap,
        mode=cv2.RETR_TREE,
        method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

    if len(contours) == 0:
        return np.asarray([[0, 0, width, height]]), 1

    if not multi_contour_eval:
        contours = [max(contours, key=cv2.contourArea)]

    estimated_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x0, y0, x1, y1 = x, y, x + w, y + h
        x1 = min(x1, width - 1)
        y1 = min(y1, height - 1)
        estimated_boxes.append([x0, y0, x1, y1])

    return estimated_boxes, contours


def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=args.device)
    output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"

    predictor = SamPredictor(sam)
    
    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        targets = [
            f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
        ]
        targets = [os.path.join(args.input, f) for f in targets]

    if not os.path.isdir(args.mask_input):
        targets_mask = [args.mask_input]
    else:
        targets_mask = [
            f for f in os.listdir(args.mask_input) if not os.path.isdir(os.path.join(args.mask_input, f))
        ]
        targets_mask = [os.path.join(args.mask_input, f) for f in targets_mask]


    os.makedirs(args.output, exist_ok=True)

    for t,t_mask in zip(targets,targets_mask):
        print(f"Processing '{t}'...")
        image = cv2.imread(t)
        if image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(t_mask, cv2.IMREAD_GRAYSCALE)

        if(args.multi_contour):
            boxes, _ = scoremap2bbox(mask, 0, multi_contour_eval=True)
            predictor.set_image(image)
            boxes = np.array(boxes)
            boxes = torch.tensor(boxes, device=predictor.device)  
            boxes = predictor.transform.apply_boxes_torch(boxes, image.shape[:2])  

            masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=boxes,
            multimask_output=False,
        )
            masks = masks.cpu().numpy()
            masks = masks.sum(axis=0).clip(0, 1)
        else:
            boxes, _ = scoremap2bbox(mask, 0, multi_contour_eval=False)
            predictor.set_image(image)
            boxes = np.array(boxes)
            masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )

        base = os.path.basename(t)
        base = os.path.splitext(base)[0]
        save_base = os.path.join(args.output, base)
        if output_mode == "binary_mask":
            masks = np.squeeze(masks).astype(float)
            write_mask_to_folder(masks, t_mask,args.output)
        else:
            save_file = save_base + ".json"
            with open(save_file, "w") as f:
                json.dump(masks, f)
    print("Done!")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)


