import cv2
import numpy as np
import torch
import tqdm
from pytorch_grad_cam.base_cam import BaseCAM
import random

class GScoreCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=True, reshape_transform=None, is_clip=False, drop=False, mute=True, topk=None, channels=None, batch_size: int = 128, is_transformer: bool = False,
                 model_domain=None):
        super(GScoreCAM, self).__init__(model, target_layers, use_cuda, 
            reshape_transform=reshape_transform, is_clip=is_clip, drop=drop, mute=mute, batch_size=batch_size, is_transformer=is_transformer)
        self.topk = topk
        self.use_bot  = False
        self.drop = drop
        self.mute = mute
        self.model_domain = model_domain

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads
                        ):
        torch.cuda.empty_cache()
        with torch.no_grad():
        # with open('context_text.json', 'r'):
            if self.is_clip:
                # img_size = img_size
                img_tensor, text_tensor = input_tensor[0], input_tensor[1]
            else:
                img_tensor = input_tensor
            img_size = img_tensor.shape[-2 : ]
            upsample = torch.nn.UpsamplingBilinear2d(img_size)
            activation_tensor = torch.from_numpy(activations)
            # if self.cuda:
            #     activation_tensor = activation_tensor.cuda()

            upsampled = upsample(activation_tensor.float())

            maxs = upsampled.view(upsampled.size(0),
                upsampled.size(1), -1).max(dim=-1)[0]
            mins = upsampled.view(upsampled.size(0),
                upsampled.size(1), -1).min(dim=-1)[0]
            maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
            upsampled = (upsampled - mins) / (maxs - mins +1e-6) #NOTE this could lead to division by zero

            input_tensors = img_tensor[:, None, :, :].cpu()*upsampled[:, :, None, :, :]

            BATCH_SIZE = self.batch_size if hasattr(self, "batch_size") else 64
            k = 300 if self.topk is None else self.topk
            #* testing for different vairance
            #^ average gradient 
            importance = torch.from_numpy(grads).float().mean(axis=(2,3))
            #^ max pooling 
            # maxpool = torch.nn.MaxPool2d(grads.shape[2:])
            # importance = maxpool(torch.from_numpy(abs(grads)).float()).view((1, -1))
            #^ average pooling
            # averagepool = torch.nn.AvgPool2d(grads.shape[2:])
            # importance = averagepool(torch.from_numpy(abs(grads)).float()).view((1, -1))

            
            if self.use_bot:
                indices_top = importance.topk(k)[1][0]
                indices_bot = importance.topk(k, largest=False)[1][0]
                indices = torch.cat([indices_top, indices_bot])
            else:
                indices = importance.topk(k)[1][0]

            scores = []
            top_tensors = input_tensors[:,indices]
            if isinstance(target_category, int):
                target_category = [target_category]
            for category, tensor in zip(target_category, top_tensors):
                for i in tqdm.tqdm(range(0, tensor.size(0), BATCH_SIZE), disable=self.mute):
                    batch = tensor[i : i + BATCH_SIZE, :]
                    if self.is_clip:
                        outputs =  self.model(batch.cuda(), text_tensor.cuda())[0].cpu().numpy()[:, category]
                    else:
                        outputs = self.model(batch.cuda()).cpu().numpy()[:, category]
                    scores.extend(outputs)
                    
            scores = torch.Tensor(scores)
            if scores.isnan().any():
                scores = scores.nan_to_num(nan=0.0) # fix nan bug in clip implementation

            # place the chosen scores back to the weight
            emtpy_score = torch.zeros(activations.shape[1])
            emtpy_score[indices] = scores          
            scores = emtpy_score.view(activations.shape[0], activations.shape[1])


            weights = torch.nn.Softmax(dim=-1)(scores)
            if self.use_bot:
                bot_mask    = torch.ones(activations.shape[1])
                bot_mask[indices_bot] = -1
                bot_mask = bot_mask.view(activations.shape[0], activations.shape[1])
                weights = weights * bot_mask
            return weights.numpy()
        
