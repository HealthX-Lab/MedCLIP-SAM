import torch
import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize
from tqdm import tqdm
import cv2
import os

class RISE(nn.Module):
    def __init__(self, model, input_size, gpu_batch=100):
        super(RISE, self).__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch

    def generate_masks(self, N, s, p1, savepath='masks.npy'):
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')

        self.masks = np.empty((N, *self.input_size))

        for i in tqdm(range(N), desc='Generating filters'):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            self.masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                         anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        np.save(savepath, self.masks)
        self.masks = torch.from_numpy(self.masks).float()
        self.masks = self.masks.cuda()
        self.N = N
        self.p1 = p1

    def load_masks(self, filepath):
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).float().cuda()
        self.N = self.masks.shape[0]

    def forward(self, x):
        if isinstance(x, tuple):
            multi_input = True
            x, y = x[0], x[1]
            y = y.repeat((self.gpu_batch, 1))
        else:
            multi_input = False
        N = self.N
        _, _, H, W = x.size()
        # Apply array of filters to the image
        # stack = torch.mul(self.masks, x.data) #! To save GPU memory, save the stack in cpu and convert it to gpu when used
        stack = torch.mul(self.masks.cpu(), x.data.cpu())
        
        # p = nn.Softmax(dim=1)(model(stack)) processed in batches
        p = []
        with torch.no_grad():
            for i in range(0, N, self.gpu_batch):
                if not multi_input:
                    p.append(self.model(stack[i:i + self.gpu_batch].cuda()))
                else:
                    p.append(self.model(stack[i: i + self.gpu_batch].cuda(), y)[0])
        p = torch.cat(p)
        # Number of classes
        CL = p.size(1)
        sal = torch.matmul(p.data.transpose(0, 1).float(), self.masks.view(N, H * W))
        sal = sal.view((CL, H, W))
        sal = sal / N / self.p1
        return sal


class RiseCAM(object):
    def __init__(self, model: torch.nn.Module, image_size: tuple, batch_size: int, mask_path: str= None, N: int = 8000):
        self.mask_path = mask_path
        self.image_size = image_size
        self.model = model
        self.explainer = RISE(model, image_size, batch_size)
        self.explainer.N = N
        self._get_rise_masks()
        
    def _get_rise_masks(self):
        if self.mask_path is not None and os.path.isfile(self.mask_path):
            self.explainer.load_masks(self.mask_path)
            self.explainer.p1 = 0.1
        else:
            self.explainer.generate_masks(N=self.explainer.N, s=8, p1=0.1, savepath=f'data/rise_mask_{self.image_size[0]}x{self.image_size[1]}.npy')
    
    def scale_and_normalize(self, heatmap, image_size=None):
        if heatmap.dtype == 'float16':
            heatmap = heatmap.astype(np.float32)
        heatmap = heatmap.cpu().numpy()
        heatmap = cv2.resize(heatmap, self.image_size) if image_size is None else cv2.resize(heatmap, image_size)
        heatmap = heatmap - np.min(heatmap)
        heatmap = heatmap/ (np.max(heatmap) + 1e-9)
        return np.float32(heatmap)
        
    
    def __call__(self, inputs, targets=None, image_size=None):
        if isinstance(inputs, tuple):
            # image_input, text_input = inputs[0], inputs[1]
            # logits_per_image, logits_per_text = self.model(image_input, text_input)
            max_class = 0 # in the CLIP case, only one class input. Thus the dimension of heatmaps will be 1xwxh 
        elif targets is None:
            logits = self.model(inputs)
            max_class = logits.argmax()
        else:
            max_class = targets
        saliency_maps = self.explainer(inputs)
        saliency = saliency_maps[max_class]

        return self.scale_and_normalize(saliency, image_size)