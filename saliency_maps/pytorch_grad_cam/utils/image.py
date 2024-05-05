import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, ToTensor
import copy

def preprocess_image(img: np.ndarray, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) -> torch.Tensor:
    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return preprocessing(img.copy()).unsqueeze(0)


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


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
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def normalize2D(cam):
    cam -= np.min(cam)
    cam /= (1e-8 + np.max(cam))
    return cam  

def resize2D(cam, target_size):
    if cam.dtype == np.float16:
        cam = np.float32(cam)
    cam = cv2.resize(cam, target_size)
    return cam

def scale_cam_image(cam, target_size=None):
    result = []
    for img in cam:
        img = img - np.min(img)
        img = img / (1e-8 + np.max(img))
        if target_size is not None:
            if img.dtype == np.float16:
                img = np.float32(img)
            img = cv2.resize(img, target_size)
        result.append(img)
    result = np.float32(result)

    return result

def scale_accross_batch_and_channels(tensor, target_size):
    batch_size, channel_size = tensor.shape[:2]
    reshaped_tensor = tensor.reshape(
        batch_size * channel_size, *tensor.shape[2:])
    result = scale_cam_image(reshaped_tensor, target_size)
    result = result.reshape(
        batch_size,
        channel_size,
        target_size[1],
        target_size[0])
    return result


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
    float_map = normalize2D(binary_map.astype(float))
    model_size_map = cv2.resize(float_map, model_input_size, interpolation=cv2.INTER_LINEAR)
    # sigmoid (line 9)
    model_size_map = stable_sigmoid(model_size_map)
    model_size_map = np.where(model_size_map > 0.5, model_size_map, 0)
    # resize to original image input (line 10)
    input_size_map = cv2.resize(model_size_map, raw_image_size, interpolation=cv2.INTER_NEAREST)
    
    return input_size_map

def cls_reshpae(tensor):
    result = tensor[0:1,:,:]

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.permute(2, 0, 1)
    return result.unsqueeze(0).numpy()