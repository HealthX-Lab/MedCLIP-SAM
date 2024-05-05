import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import random

"""
Modification for CLIP
"""

class BaseCAM(object):
    """ Base class for Class activation mapping.
        : Args
            - **model_dict -** : Dict. Has format as dict(type='vgg', arch=torchvision.models.vgg16(pretrained=True),
            layer_name='features',input_size=(224, 224)).
    """

    def __init__(self, model, mute=False, is_clip=False):
        self.mute = mute
        self.model = model
        if torch.cuda.is_available():
          self.model.cuda()
        self.gradients = dict()
        self.activations = dict()
        self.is_clip = is_clip

        def backward_hook(module, grad_input, grad_output):
            if torch.cuda.is_available():
              self.gradients['value'] = grad_output[0].cuda()
            else:
              self.gradients['value'] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            if torch.cuda.is_available():
              self.activations['value'] = output[0].cuda()
            else:
              self.activations['value'] = output[0]
            return None

        try:
            target_layer = model.visual.layer4[-1]
        except:
            raise Exception('Target layer not found.')

        target_layer.register_full_backward_hook(forward_hook)#
        target_layer.register_forward_hook(backward_hook)


    def forward(self, input, text, class_idx=None, retain_graph=False):
        return None

    def __call__(self, input, text, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)

class SSCAM1(BaseCAM):

    """x
        SSCAM1, inherit from BaseCAM
    """

    def __init__(self, model):
        super().__init__(model)
        self.drop = True

    def forward(self, inputs, class_idx=None, raw_size=None, param_n=16, mean=0, sigma=0.2, retain_graph=False, mute=False):
        if isinstance(inputs, tuple):
            img, text = inputs[0], inputs[1]
        else:
            img = inputs
        
        b, c, h, w = img.size()
        
        # prediction on raw img
        if self.is_clip:
            logit = self.model(img, text)[0]
        else:
            logit = self.model(img)
        
        if class_idx is None:
            predicted_class = logit.max(1)[-1]
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            predicted_class = torch.LongTensor([class_idx])
            score = logit[:, class_idx].squeeze()
        
        # logit = F.softmax(logit) # softmaxed logit never been in used

        if torch.cuda.is_available():
            predicted_class= predicted_class.cuda()
            score = score.cuda()
        #   logit = logit.cuda()

        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)
        activations = self.activations['value']
        b1, k, u, v = activations.size()
        
        BATCH_SIZE = 16

        if torch.cuda.is_available():
            activations = activations.cuda()
          

        #HYPERPARAMETERS (can be modified for better/faster explanations)
        #mean = 0
        #param_n = 35
        #param_sigma_multiplier = 2
        if self.drop:
            #randomly drop 90% of the in_channels
            k = 300
            indices = torch.tensor(random.sample(range(activations.shape[1]), k))
            # indices = torch.tensor(indices)
            chosen_activations = activations[:,indices]
        else:
            chosen_activations = activations
        # score_saliency_map = torch.zeros((chosen_activations.size(1), 1, h, w)).cuda()
        weight_list = []
        with torch.no_grad():
            for i in tqdm(range(0, chosen_activations.size(1), BATCH_SIZE), disable=self.mute):
                batch = chosen_activations[0, i : i + BATCH_SIZE, :]
                # upsampling
                saliency_map = torch.unsqueeze(batch, 1)
                
                saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

                if saliency_map.max() == saliency_map.min():
                    continue

                x = saliency_map               

                if (torch.max(x) - torch.min(x)).item() == 0:
                    continue
                else:
                    sigma = sigma / (torch.max(x) - torch.min(x)).item()
                
                noisy_list = []
                
                noise_weight = torch.zeros((batch.size(0), 1)).cuda()
                # Adding noise to the upsampled activation map `x`
                for _ in range(param_n):

                    noise = Variable(x.data.new(x.size()).normal_(mean, sigma**2))
                    
                    noisy_img = x + noise

                    noisy_list.append(noisy_img)
                    
                    output = self.model(noisy_img * input, text)[0]
                    # output = F.softmax(output) # softmax a single value is one
                    noise_weight += output/param_n
                weight_list.append(noise_weight)
            # score_list.append(score)
              
            # Averaging the scores to introduce smoothing
            weights = torch.cat(weight_list)
            weights = weights.nan_to_num(nan=0.0)
            score_saliency_map = weights.unsqueeze(0).unsqueeze(-1) * chosen_activations
                
        if self.drop:
            # place the 10% the activation map back
            emtpy_activation = torch.zeros(activations.shape).cuda()
            emtpy_activation[0, indices] = score_saliency_map[0]
            score_saliency_map = emtpy_activation

        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

        if score_saliency_map_min == score_saliency_map_max:
            return None

        score_saliency_map = (score_saliency_map - score_saliency_map_min).div(score_saliency_map_max - score_saliency_map_min).data

        if raw_size is not None:
            import cv2
            import numpy as np
            score_saliency_map = score_saliency_map[0][0].cpu().numpy().astype(np.float32)
            if np.isnan(score_saliency_map).any(): # in casdoge the normalization has division by zero(or near zero) errors, replace all infinite number with zeros.
                score_saliency_map = np.nan_to_num(score_saliency_map, nan=0.0)
            score_saliency_map = cv2.resize(score_saliency_map, raw_size)

        return score_saliency_map

    def __call__(self, input, text, class_idx=None, retain_graph=False, raw_size=None, param_n=35, mean=0, sigma=0.2, mute=False):
        return self.forward(input, text, class_idx, retain_graph=retain_graph, raw_size=raw_size,
                            param_n=param_n, mean=mean, sigma=sigma, mute=mute)


class SSCAM2(BaseCAM):

    """
        SSCAM2, inherit from BaseCAM
    """

    def __init__(self, model_dict):
        super().__init__(model_dict)

    def forward(self, inputs, class_idx=None, param_n=35, mean=0, sigma=2, retain_graph=False):
        if isinstance(inputs, tuple):
            img, text = inputs[0], inputs[1]
        else:
            img = inputs
       
        b, c, h, w = img.size()
        
        # prediction on raw input
        if self.is_clip:
            logit = self.model(img, text)[0]
        else:
            logit = self.model(img)
        
        if class_idx is None:
            predicted_class = logit.max(1)[-1]
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            predicted_class = torch.LongTensor([class_idx])
            score = logit[:, class_idx].squeeze()
        
        # logit = F.softmax(logit)

        if torch.cuda.is_available():
          predicted_class= predicted_class.cuda()
          score = score.cuda()
        #   logit = logit.cuda()                

        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)
        activations = self.activations['value']
        b1, k, u, v = activations.size()
        
        score_saliency_map = torch.zeros((1, 1, h, w))

        if torch.cuda.is_available():
          activations = activations.cuda()
          score_saliency_map = score_saliency_map.cuda()

        #HYPERPARAMETERS (can be modified for better/faster explanations)
        #mean = 0
        #param_n = 35
        #param_sigma_multiplier = 2
        

        with torch.no_grad():
          for i in range(k):

              # upsampling
              saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
              
              saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

              if saliency_map.max() == saliency_map.min():
                continue

              # Normalization
              norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

              x = input * norm_saliency_map              

              if (torch.max(x) - torch.min(x)).item() == 0:
                continue
              else:
                sigma = sigma / (torch.max(x) - torch.min(x)).item()
              
              score_list = []
              noisy_list = []

              # Adding noise to the normalized input mask `x`
              for i in range(param_n):

                noise = Variable(x.data.new(x.size()).normal_(mean, sigma**2))
                
                noisy_img = x + noise

                noisy_list.append(noisy_img)
                
                noisy_img = noisy_img.cuda()
                output = self.model(noisy_img)
                output = F.softmax(output)
                score = output[0][predicted_class]
                score_list.append(score)

              # Averaging the scores to introduce smoothing               
              score = sum(score_list) / len(score_list)
              score_saliency_map +=  score * saliency_map
                
        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

        if score_saliency_map_min == score_saliency_map_max:
            return None

        score_saliency_map = (score_saliency_map - score_saliency_map_min).div(score_saliency_map_max - score_saliency_map_min).data

        return score_saliency_map

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)