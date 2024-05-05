import PIL
from PIL import Image, ImageDraw, ImageFont, ImageColor
# from skimage.util import img_as_float
# from skimage import color
import skimage
import os, textwrap
import copy
from tqdm import tqdm
from numpy import character
import cv2
import numpy as np


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception("The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def draw_text(img, text_list, text_size=14, inside = False):
    try:
        ft = ImageFont.truetype("font/arial.ttf", text_size)
    except:
        ft = ImageFont.truetype("arial.ttf", text_size)
    # ft = ImageFont.truetype("/Library/fonts/Arial.ttf", 14)
    img_width, img_height = img.size
    reformated_lines = []
    text_lines_height = []
    for idx, line in enumerate(text_list):
        line_width, line_height = ft.getsize(line)
        if line_width > img_width: # wrap text to multiple lines
            character_width = line_width/(len(line))
            characters_per_line = round(img_width/character_width) - 2
            sperate_lines = textwrap.wrap(line, characters_per_line)
            text_lines_height.append(len(sperate_lines))
            line = '\n'.join(sperate_lines)
        else:
            text_lines_height.append(1)
        reformated_lines.append(line)

    total_lines = sum(text_lines_height)
    if inside:
        text_img = Image.new('RGB', (img_width, line_height*total_lines+10), (255,255,255))
        img.paste(text_img, (0,0, img_width, line_height*total_lines+10))
        new_img = img
        draw = ImageDraw.Draw(new_img)
        y_text = 0
        for idx, line in enumerate(reformated_lines):
            draw.text((3, y_text), line, 'black', font=ft)
            y_text += (line_height+2)*text_lines_height[idx]

    else: 
        new_img = Image.new('RGB', (img_width, img_height+line_height*total_lines+10), (255,255,255))
        new_img.paste(img, (0,0, img_width, img_height))
        draw = ImageDraw.Draw(new_img)

        y_text = img_height
        for idx, line in enumerate(reformated_lines):
            draw.text((3, y_text), line, 'black', font=ft)
            y_text += (line_height+2)*text_lines_height[idx]

    return new_img

def draw_box(img, box, color='red', width=3):
    new_img = copy.deepcopy(img)
    draw = ImageDraw.Draw(new_img)
    draw.rectangle(box, outline=color, width=width)
    return new_img

def generate_text_img(text, text_color, mode='RGB', text_size = 12, img_width=False):
    ft = ImageFont.truetype("arial.ttf", text_size)
    w, h = ft.getsize(text)
    # compute # of lines
    # lines = math.ceil(img_width / width) +     
    height = h
    if len(mode) == 1: # L, 1
        background = (255)
        color = (0)
    if len(mode) == 3: # RGB
        background = (255, 255, 255)
        color = (0,0,0)
    if len(mode) == 4: # RGBA, CMYK
        background = (255, 255, 255, 255)
        color = (0,0,0,0)
    if img_width:
        textImage = Image.new(mode, (img_width, height), background)
    else:
        textImage = Image.new(mode, (w, height), background)
    draw = ImageDraw.Draw(textImage)  

    # ipdb.set_trace()
    # tx_w, tx_h = ft.getsize(text)
    draw.text((5, 0), text, text_color, font=ft)

    return textImage

"""
Concatenate images with imagick
Input:
# dir_list: A list of dirs, each dir must have same amount of images and same name
# out_dir : output folder 
Output:
A set concatenate images
"""
def concat_imgs(dir_list,
                out_dir='visualization_samples/clip_context/s_iMs_M_montage'):
    os.makedirs(out_dir, exist_ok=True)
    img_list = os.listdir(dir_list[0])
    num_of_dirs = len(dir_list)
    montage_file_list = []
    for img_name in img_list:
        montage_files = ''.join(
            f'{dir_list[i]}/{img_name} ' for i in range(num_of_dirs)
        )

        montage_file_list.append(montage_files)
    for files, img_name in tqdm(zip(montage_file_list, img_list), total=len(img_list)):
        os.system(f'montage -quiet {files}-tile {num_of_dirs}x1 -geometry +0+0 {out_dir}/{img_name}')
        
        
# cat images and draw boxes, will be deprecated in future.
def concat_images(raw_image, cam_img, bbox, gt_boxes, show_box=False, resize_image=False, img_size=None):
    input_size = raw_image.size
    if show_box:
        box_img = draw_box(raw_image, bbox)
        if gt_boxes is not None and len(gt_boxes) != 0:
            for gt_box in gt_boxes:
                if resize_image:
                    from tools.iou_tool import resize_box
                    gt_box = resize_box(gt_box, box_size=img_size, target_size=input_size)
                box_img = draw_box(box_img, gt_box, color='green') 
            cat_img = Image.new('RGB', (input_size[0]*3, input_size[1]))
        else:
            cat_img = Image.new('RGB', (input_size[0]*2, input_size[1]))
    else:
        cat_img = Image.new('RGB', (input_size[0]*2, input_size[1]))
    cat_img.paste(raw_image, (0,0))
    cat_img.paste(cam_img, (input_size[0],0))
    if show_box:
        cat_img.paste(box_img, (input_size[0]*2, 0))
    return cat_img

class Drawer:
    def __init__(self, show_box=True, show_cam=True):
        self.show_box = show_box
        self.show_cam = show_cam
    
    @staticmethod
    def draw_boxes(image, boxes, color, tags=None, width=1, text_size=12, loc='above'):
        if tags is not None:
            try:
                font = ImageFont.truetype("font/arial.ttf", text_size)
            except:
                font = ImageFont.truetype("arial.ttf", text_size)
            if len(boxes) != len(tags):
                raise ValueError('boxes and tags must have same length')

        for idx, box in enumerate(boxes):
            image = draw_box(image, box, color=color, width=width)
            if tags is not None:
                tag = tags[idx]
                draw = ImageDraw.Draw(image, 'RGBA')
                tag_width, tag_height = font.getmask(tag).size
                color_rgba = ImageColor.getrgb(color) + (127,)
                if loc == 'above':
                    textbb_loc = [box[0], box[1]-tag_height, box[0]+tag_width, box[1]]
                    text_loc   = (box[0], box[1]-tag_height)
                else:
                    textbb_loc = [box[0], box[1], box[0]+tag_width, box[1]+tag_height]
                    text_loc   = (box[0], box[1])
                draw.rectangle(textbb_loc, fill=color_rgba)
                draw.text(text_loc, tag, fill='white', font=font)
                
        return image

    @staticmethod
    def draw_text(image, text_list): # draw text as extra box in image
        image = draw_text(image, text_list)
        return image
    
    @staticmethod
    def overlay_cam_on_image(image, cam, use_rgb=False, color_map=cv2.COLORMAP_JET):
        float_img = skimage.util.img_as_float(image)
        if len(float_img.shape) == 2: # for gray image
            float_img = skimage.color.gray2rgb(float_img)
        cam_img_array = show_cam_on_image(float_img, cam, use_rgb=use_rgb, colormap=color_map)
        return Image.fromarray(cam_img_array)
        
    @staticmethod
    def concat(target_image_list: PIL.Image, horizontal: bool = True) -> PIL.Image:
        width, height = target_image_list[0].size
        num_imgs = len(target_image_list)
        cat_img = Image.new('RGB', (width*num_imgs, height)) if horizontal else Image.new('RGB', (width, height*num_imgs))
        for idx, img in enumerate(target_image_list):
            if horizontal:
                cat_img.paste(img, (width*idx, 0))
            else:
                cat_img.paste(img, (0, height*idx))
        return cat_img
    
    @staticmethod
    def paste_patch_to_image(box, img_patch, org_image):
        image = copy.deepcopy(org_image)
        image_width, image_height = image.size
        x1, y1, x2, y2 = box
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(x2, image_width)
        y2 = min(y2, image_height)
        img_patch = img_patch.resize((x2-x1, y2-y1))
        image.paste(img_patch, (x1, y1))
        return image
    
    @staticmethod
    def concat_imgs_in_folders(dir1, dir2, out_dir, horizontal=True):
        os.makedirs(out_dir, exist_ok=True), 
        img_list = os.listdir(dir1)
        for img_name in img_list:
            img1 = Image.open(f'{dir1}/{img_name}')
            img2 = Image.open(f'{dir2}/{img_name}')
            cat_img = CLIPDrawer.concat([img1, img2], horizontal=horizontal)
            cat_img.save(f'{out_dir}/{img_name}')
    
    @staticmethod
    def check_name_matching(folder1, folder2):
        img_list1 = os.listdir(folder1)
        img_list2 = os.listdir(folder2)
        diff = set(img_list1) - set(img_list2)
        if len(diff) > 0:
            print(f'{len(diff)} images are not in {folder2}')
            print(diff)
        diff = set(img_list2) - set(img_list1)
        if len(diff) > 0:
            print(f'{len(diff)} images are not in {folder1}')
            print(diff)
        return len(diff) == 0