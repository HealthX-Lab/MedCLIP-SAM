import cv2
import numpy as np
import torch
import tqdm
from pytorch_grad_cam.base_cam import BaseCAM
import random

class HilaCAM(BaseCAM):
    def __init__(self, model, target_layer, use_cuda=False, reshape_transform=None, is_clip=False, drop=False, mute=False, channels=None, hila=True):
        super(HilaCAM, self).__init__(model, target_layer, use_cuda, 
            reshape_transform=reshape_transform, is_clip=is_clip, drop=drop, mute=mute, hila=hila)
        self.channels = channels

    def get_cam_weights(self,
                        input_tensor,
                        target_category,
                        activations,
                        grads
                        ):
        torch.cuda.empty_cache()
        
        if self.is_clip:
            # img_size = img_size
            img_tensor = input_tensor[0]
        else:
            img_tensor = input_tensor
        img_size = img_tensor.shape[-2 : ]
        upsample = torch.nn.UpsamplingBilinear2d(img_size)
        activation_tensor = torch.from_numpy(activations)
        if self.cuda:
            activation_tensor = activation_tensor.cuda()

        upsampled = upsample(activation_tensor)

        maxs = upsampled.view(upsampled.size(0),
            upsampled.size(1), -1).max(dim=-1)[0]
        mins = upsampled.view(upsampled.size(0),
            upsampled.size(1), -1).min(dim=-1)[0]
        maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
        upsampled = (upsampled - mins) / (maxs - mins) #NOTE this could lead to division by zero

        input_tensors = img_tensor[:, None, :, :]*upsampled[:, :, None, :, :]

        if hasattr(self, "batch_size"):
            BATCH_SIZE = self.batch_size
        else: 
            BATCH_SIZE = 16
        if self.drop:
            #randomly drop 90% of the in_channels
            k = 300
            indices = torch.tensor(random.sample(range(input_tensors.shape[1]), k))
            # indices = torch.tensor(indices)
            input_tensors = input_tensors[:,indices]
        if self.channels is not None:
            indices = self.channels
            input_tensors = input_tensors[:, indices]


        scores = []

        with torch.no_grad():
            for batch_index, tensor in enumerate(input_tensors):
                category = target_category[batch_index]
                for i in tqdm.tqdm(range(0, tensor.size(0), BATCH_SIZE), disable=self.mute):
                    batch = tensor[i : i + BATCH_SIZE, :]
                    if self.clip:
                        outputs =  self.model(batch, input_tensor[1])[0].cpu().numpy()[:, category]
                    else:
                        outputs = self.model(batch).cpu().numpy()[:, category]
                    scores.extend(outputs)
        scores = torch.Tensor(scores)
        if scores.isnan().any():
            scores = scores.nan_to_num(nan=0.0) # fix nan bug in clip implementation

        if self.drop or self.channels is not None:
            # place the 10%(Or handpick channels) score back to the weight
            emtpy_score = torch.zeros(activations.shape[1])
            emtpy_score[indices] = scores
            scores = emtpy_score.view(activations.shape[0], activations.shape[1])
        else:
            scores = scores.view(activations.shape[0], activations.shape[1])



        weights = torch.nn.Softmax(dim=-1)(scores).numpy()
        
        return weights