import numpy as np
from pytorch_grad_cam.base_cam import BaseCAM
import torch
from tqdm import tqdm

class ScoreHooker:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layer, reshape_transform):
        self.model = model
        self.activations = []
        self.reshape_transform = reshape_transform
        target_layer.register_forward_hook(self.save_activation)

    def save_activation(self, module, input, output):
        activation = output
        
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def __call__(self, x):
        
        return self.model(*x) if isinstance(x, tuple) else self.model(x)

class TestCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None, is_clip=False, topk=300, batch_size=128, is_transformer=False):
        super(
            TestCAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform,
            is_clip=is_clip,
            topk=topk,
            batch_size=batch_size,
            is_transformer=is_transformer)
        self.topk = topk
        self.is_clip = is_clip
        self.batch_size = batch_size
        self.use_bot  = False
        self.mute = True
    
    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        # torch.cuda.empty_cache()
        # # get_scores = ScoreHooker(self.model, self.model.visual.attnpool.c_proj, None)
        # # output, _ = get_scores(input_tensor)
        
        # with torch.no_grad():
        # # with open('context_text.json', 'r'):
        #     if self.is_clip:
        #         # img_size = img_size
        #         img_tensor, text_tensor = input_tensor[0], input_tensor[1]
        #     else:
        #         img_tensor = input_tensor
        #     img_size = img_tensor.shape[-2 : ]
        #     upsample = torch.nn.UpsamplingBilinear2d(img_size)
        #     activation_tensor = torch.from_numpy(activations)
        #     # if self.cuda:
        #     #     activation_tensor = activation_tensor.cuda()

        #     upsampled = upsample(activation_tensor.float())

        #     maxs = upsampled.view(upsampled.size(0),
        #         upsampled.size(1), -1).max(dim=-1)[0]
        #     mins = upsampled.view(upsampled.size(0),
        #         upsampled.size(1), -1).min(dim=-1)[0]
        #     maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
        #     upsampled = (upsampled - mins) / (maxs - mins +1e-6) #NOTE this could lead to division by zero

        #     input_tensors = img_tensor[:, None, :, :].cpu()*upsampled[:, :, None, :, :]

        #     BATCH_SIZE = self.batch_size if hasattr(self, "batch_size") else 128
        #     k = 300 if self.topk is None else self.topk
        #     #* value of importance define the ranking system (could also be used as weighting)
        #     #^ average gradient 
        #     grad_avg = torch.from_numpy(grads).float().mean(axis=(2,3))
        #     #^ max gradient
        #     grad_max = torch.from_numpy(grads).float().amax(dim=(2,3))
        #     #^ variance of gradient
        #     grad_var = torch.from_numpy(grads).float().var(axis=(2,3))
            
        #     importance = grad_max
            
        #     #* choose topk by importance
        #     if self.use_bot:
        #         indices_top = importance.topk(k)[1][0]
        #         indices_bot = importance.topk(k, largest=False)[1][0]
        #         indices = torch.cat([indices_top, indices_bot])
        #     else:
        #         indices = importance.topk(k)[1][0]

        #     #* Score-CAM base scoring system
        #     #^ origin Score-cam
        #     scores = []
        #     top_tensors = input_tensors[:,indices]
        #     if isinstance(target_category, int):
        #         target_category = [target_category]
        #     for category, tensor in zip(target_category, top_tensors):
        #         for i in tqdm(range(0, tensor.size(0), BATCH_SIZE), disable=self.mute):
        #             batch = tensor[i : i + BATCH_SIZE, :]
        #             if self.is_clip:
        #                 outputs =  self.model(batch.cuda(), text_tensor.cuda())[0].cpu().numpy()[:, category]
        #             else:
        #                 outputs = self.model(batch.cuda()).cpu().numpy()[:, category]
        #             scores.extend(outputs)
                    
        #     scores = torch.Tensor(scores)
        #     if scores.isnan().any():
        #         scores = scores.nan_to_num(nan=0.0) # fix nan bug in clip implementation
        #     #^ simply sum the topk scores
        #     # scores = torch.ones(len(indices))
            
        #     # place the chosen scores back to the weight
        #     emtpy_score = torch.zeros(activations.shape[1])
        #     #* original gscorecam
        #     # emtpy_score[indices] = scores          
        #     # scores_all = emtpy_score.view(activations.shape[0], activations.shape[1])
        #     # weights = torch.nn.Softmax(dim=-1)(scores_all)
        #     #* variance #1: softmax over chosen channels
        #     # scores = torch.nn.Softmax(dim=-1)(scores)            
        #     # emtpy_score[indices] = scores          
        #     # weights = emtpy_score.view(activations.shape[0], activations.shape[1])
        #     #* variance #2: combine gradient with weights
        #     # scores = (torch.nn.Softmax(dim=-1)(scores) + torch.nn.Softmax(dim=-1)(importance[:,indices]))/2
        #     # emtpy_score[indices] = scores          
        #     # weights = emtpy_score.view(activations.shape[0], activations.shape[1])            
        #     #* variance #3: combine softmax(gradient) with logit
        #     # scores = torch.nn.Softmax(dim=-1)(scores * torch.nn.Softmax(dim=-1)(importance))
        #     # emtpy_score[indices] = scores          
        #     # weights = emtpy_score.view(activations.shape[0], activations.shape[1])          
        #     #* variance #4: combine softmax(logit) with gradient
        #     # scores = torch.nn.Softmax(dim=-1)(importance * torch.nn.Softmax(dim=-1)(scores))  
        #     # emtpy_score[indices] = scores          
        #     # weights = emtpy_score.view(activations.shape[0], activations.shape[1])
        #     #* variance #5: simply multiply the gradient by the score
        #     emtpy_score[indices] = scores          
        #     scores_all = emtpy_score.view(activations.shape[0], activations.shape[1]) * importance
        #     weights = torch.nn.Softmax(dim=-1)(scores_all)
            
             


        #     if self.use_bot:
        #         bot_mask    = torch.ones(activations.shape[1])
        #         bot_mask[indices_bot] = -1
        #         bot_mask = bot_mask.view(activations.shape[0], activations.shape[1])
        #         weights = weights * bot_mask
            weights = np.ones(3072)/3072
            return weights