from typing import List
from pytorch_grad_cam import *

import torch, torchvision
import numpy as np
import cv2
# from lime.lime import LIME

        
def hilacam(image, text, model, device, index=None, cam_size=None):
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1)
    if index is None:
        # locate the largest score of img-text pair
        index = np.argmax(logits_per_image.cpu().data.numpy(), axis=-1)
    one_hot = np.zeros((1, logits_per_image.size()[-1]), dtype=np.float32)
    # create a tensor equal to the clip score
    one_hot[0, index] = 1
    one_hot = torch.from_numpy(one_hot)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()
    # back propergate to the network
    one_hot.requires_grad_(True)
    one_hot.backward(retain_graph=True)
    # create a diagonal matrix
    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    # R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype) #* change to cpu to resolve memory overflow
    # weighted activation
    for blk in image_attn_blocks:
        grad = blk.attn_grad
        cam = blk.attn_probs
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=0)
        # R += torch.matmul(cam, R)
        R += torch.matmul(cam.detach().cpu(), R)
    R[0, 0] = 0
    image_relevance = R[0, 1:]

    length = image_relevance.shape[-1]
    heatmap_size = int(length**0.5)
    image_relevance = image_relevance.reshape(1, 1, heatmap_size, heatmap_size)
    image_relevance = torch.nn.functional.interpolate(image_relevance.float(), size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())    

    return (
        cv2.resize(image_relevance, cam_size)
        if cam_size is not None
        else image_relevance
    )

class CAMWrapper(object):
    CAM_LIST = ['gradcam', 'scorecam', 'gradcam++', 'ablationcam', 'xgradcam', 'eigencam', 'eigengradcam', 'layercam', 'hilacam', 'groupcam', 'sscam1', 'sscam2', 'rawcam', 'testhila', 'gradientcam', 'gscorecam', 'vitgradcam', 'rise', 'gscorecambeta', 'testcam']
    CAM_DICT = {"gradcam": GradCAM,    # consider as baseline
        "scorecam": ScoreCAM,         # Good but slow
        "gradcam++": GradCAMPlusPlus, # focus barely shift from different inputs
        "ablationcam": AblationCAM,   # slightly worse than scorecam but has similar computing complexity
        "xgradcam": XGradCAM,         # bad for CLIP
        "eigencam": EigenCAM,         # does not work for CLIP
        "eigengradcam": EigenGradCAM, # not good for CLIP
        "layercam": LayerCAM,         # sometimes very focus, sometimes very noisy
        "hilacam": hilacam,
        "groupcam": GroupCAM,         # slightly noiser than gradcam
        "sscam1": SSCAM1,             # doesn't work for clip, extremly slow
        "sscam2": SSCAM2,             # doesn't work for clip, extremly slow
        "testhila": HilaCAM,
        "gscorecam": GScoreCAM,
        "rise": RiseCAM,
        "gscorecambeta": GScoreCAMBeta,
        "testcam": TestCAM,
    }
    def __init__(self, model, target_layers, tokenizer, cam_version, preprocess=None, target_category=None, is_clip=True,
                 mute=False, cam_trans=None, is_transformer=False, model_domain = "biomedclip", **kwargs):
        """[summary]

        Args:
            model (model): [description]
            target_layers (model layer): List[layers]
            drop (bool, optional): [description]. Defaults to False.
            cam_version (str, optional): [description]. Defaults to 'gradcam'.
            target_category (int or tensor, optional): [description]. Defaults to None.
            mute (bool, optional): [description]. Defaults to False.
            channel_frame (csv, optional): [description]. Defaults to None.
            channels (int, optional): [description]. Defaults to None.
            cam_trans (function, optional): [description]. Defaults to None.

        Raises:
            Exception: [description]
        """
        self.mute = mute
        self.model = model
        self.version = cam_version
        self.target_layers = target_layers
        self.target_category = target_category
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.cam_trans = cam_trans
        self.is_transformer = is_transformer
        self.is_clip = is_clip
        self.channels = None
        self.__dict__.update(kwargs)
        self.model_domain = model_domain

        if self.version not in self.CAM_LIST:
            raise ValueError(f"CAM version not found. Please choose from: {self.CAM_LIST}")
        # define cam
        self._load_cam()
    
    def _select_channels(self, text):
        if self.channel_dict is not None and text in self.channel_dict.keys():
            return self.channel_dict[text][:self.topk]
        else:
            return None
        

    def _load_channel_from_csv(self, channel_frame): #!deprecated
        import pandas as pd
        channelFrame = pd.read_csv(channel_frame, index_col=0)
        return channelFrame[channelFrame.columns[0]].to_list()

    # load cam
    def _load_cam(self):
        if self.version == 'hilacam':
            self.cam = self.CAM_DICT[self.version]
        elif self.version == 'testhila':
            target_layer = self.model.visual.attnpool
            self.cam = self.CAM_DICT[self.version](model=self.model, target_layers=self.target_layers, use_cuda=True, clip=self.is_clip , reshape_transform=self.cam_trans, hila=True,model_domain=self.model_domain)

        elif self.version in ['scorecam', 'gscorecam']:
            batch_size = self.batch_size if hasattr(self, "batch_size") else 128
            self.cam = self.CAM_DICT[self.version](model=self.model, target_layers=self.target_layers, 
                                        use_cuda=True, is_clip=self.is_clip , reshape_transform=self.cam_trans, drop=self.drop, 
                                        mute=self.mute, channels=self.channels, topk=self.topk, batch_size=batch_size, is_transformer=self.is_transformer,model_domain=self.model_domain)
        elif self.version == 'groupcam':
            self.cam = self.CAM_DICT[self.version](self.model, self.target_layers[0], cluster_method='k_means', is_clip=self.is_clip,model_domain=self.model_domain)
        elif self.version == 'layercam':
            self.cam = self.CAM_DICT[self.version](model=self.model, target_layers=self.target_layers, 
                                        use_cuda=True, is_clip=self.is_clip , reshape_transform=self.cam_trans,model_domain=self.model_domain)
        elif self.version.startswith('sscam'):
            self.cam = self.CAM_DICT[self.version](model=self.model, is_clip=self.is_clip,model_domain=self.model_domain)
        elif self.version == 'rise':
            img_size = self.dataset_size if hasattr(self, 'dataset_size') else (384, 384)
            mask_path = f'data/rise_mask_{img_size[0]}x{img_size[1]}.npy'
            self.cam = self.CAM_DICT[self.version](model=self.model, image_size=img_size, mask_path=mask_path, batch_size=64,model_domain=self.model_domain)
        else:
            self.cam = self.CAM_DICT[self.version](model=self.model, target_layers=self.target_layers, 
                                        use_cuda=True, is_clip=self.is_clip , reshape_transform=self.cam_trans, is_transformer=self.is_transformer,
                                        model_domain=self.model_domain)

    def getCAM(self, input_img, input_text, cam_size, target_category,model_domain):
        cam_input = (input_img, input_text) if self.is_clip else input_img
        self.cam.img_size = cam_size
        if self.version == 'hilacam':
            grayscale_cam = self.cam(input_img, input_text, self.model, 'cuda', cam_size=cam_size, index=target_category)
        elif self.version == 'groupcam':
            grayscale_cam = self.cam(cam_input, class_idx=target_category)
            grayscale_cam = np.nan_to_num(grayscale_cam, nan=0.0)
        elif self.version.startswith('sscam'):
            grayscale_cam = self.cam(input_img, input_text, class_idx=target_category, 
                                     param_n=35, mean=0, sigma=2, mute=self.mute)
        elif self.version == 'layercam':
            grayscale_cam = self.cam(input_tensor=cam_input, targets=target_category)
            grayscale_cam = grayscale_cam[0, :]
        elif self.version == 'rise':
            grayscale_cam = self.cam(inputs=cam_input, targets=target_category, image_size=cam_size)
        # elif self.version == 'lime':
        #     grayscale_cam = self.cam(inputs=cam_input, target=target_category, image_size=(224, 224), image=kwargs['image'])
        else:
            grayscale_cam = self.cam(input_tensor=cam_input, targets=target_category)
            grayscale_cam = grayscale_cam[0, :]
        return grayscale_cam
    
    def __call__(self, inputs, label, heatmap_size,model_domain):

        if isinstance(inputs, tuple):
            img, text = inputs[0], inputs[1]
        else:
            img = inputs
            text = None

        if self.preprocess is not None:
            img = self.preprocess(img)
        # tokenize text
        text_token = None if self.tokenizer is None else self.tokenizer(text).cuda()
        if len(img.shape) < 4:
            img = img.unsqueeze(0)
        if not img.is_cuda:
            img = img.cuda()
        if hasattr(self, "channel_dict"):
            # self.cam.channels = self.channel_dict[text]
            self.cam.channels = self._select_channels(text)

        return self.getCAM(img, text_token, heatmap_size, label,model_domain)
    
    def getLogits(self, img, text):
        with torch.no_grad():
            if self.preprocess is not None:
                img = self.preprocess(img)
            img_per_text, text_per_img = self.model(img.unsqueeze(0).cuda(), self.tokenizer(text).cuda())
        return img_per_text, text_per_img


def get_heatmap_from_mask(
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
    # heatmap = np.float32(heatmap) / 255

    # if np.max(img) > 1:
    #     raise Exception("The input image should np.float32 in the range [0, 1]")

    # cam = heatmap + img
    # cam = cam / np.max(cam)
    return heatmap


def CLIP_topk_channels(src_path='data/featuremap_search', cat_name='all', topk=125):
    from sklearn import metrics
    from tools.utils import getFileList
    print("Getting channel dictionary ...")
    files = getFileList(src_path, suffix='.npy', if_path=False)
    xrange = np.arange(0.05, 1, 0.05)
    channels = {}
    for npy_file in files:
        class_name = npy_file.split('_')[0]
        ious = np.load(f'{src_path}/{npy_file}')
        auc = [metrics.auc(xrange, iou) for iou in ious]
        auc = torch.tensor(auc)
        top_values, top_index = auc.topk(topk)
        channels[class_name] = top_index
    return channels if cat_name=='all' else {cat_name:channels[cat_name]}


def load_cam(model: torch.nn.Module, 
                cam_version: str,
                target_layers: List[torch.nn.Module], 
                preprocess: torchvision.transforms.Compose,
                cam_trans: torch.nn.Module, # for transformer
                drop: bool,
                topk: int, 
                is_clip: bool,
                tokenizer: None or torch.nn.Module = None,
                use_channel_dict: bool=False, 
                channel_search_path: str = None,
                is_transformer: bool = False):
    if use_channel_dict:
        if channel_search_path.endswith('.json'):
            import json
            channel_dict = json.load(open(channel_search_path, 'r'))
        else:
            channel_dict = CLIP_topk_channels(channel_search_path, cat_name='all', topk=topk)
    else:
        channel_dict = {}
    
    return CAMWrapper(model, 
                            target_layers=target_layers, 
                            tokenizer=tokenizer, 
                            cam_version=cam_version, 
                            preprocess=preprocess, 
                            cam_trans=cam_trans, 
                            is_clip=is_clip,
                            topk=topk,
                            drop=drop, 
                            channels=None, 
                            channel_dict=channel_dict,
                            mute=True)