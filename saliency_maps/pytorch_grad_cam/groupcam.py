import torch
import torch.nn.functional as F
from kornia.filters.gaussian import gaussian_blur2d
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from sklearn.cluster import KMeans
    from sklearn.cluster import AgglomerativeClustering
import numpy as np
blur = lambda x: gaussian_blur2d(x, kernel_size=(51, 51), sigma=(50., 50.))

class BaseCAM(object):
    def __init__(self, model, target_layer, **kwargs):
        # super(BaseCAM, self).__init__()
        self.model = model.eval()
        self.gradients = {}
        self.activations = {}

        # for module in self.model.named_modules():
        #     if module[0] == target_layer:
        #         module[1].register_forward_hook(self.forward_hook)
        #         module[1].register_backward_hook(self.backward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)#
        target_layer.register_forward_hook(self.forward_hook)


    def backward_hook(self, module, grad_input, grad_output):
        self.gradients['value'] = grad_output[0]

    def forward_hook(self, module, input, output):
        self.activations['value'] = output

    def forward(self, x, class_idx=None, retain_graph=False):
        raise NotImplementedError

    def __call__(self, x, class_idx=None, retain_graph=False):
        return self.forward(x, class_idx, retain_graph)


def group_cluster(x, group=32, cluster_method='k_means'):
    # x : (torch tensor with shape [1, c, h, w])
    xs = x.detach().cpu()
    b, c, h, w = xs.shape
    xs = xs.reshape(b, c, -1).reshape(b*c, h*w)
    if cluster_method == 'k_means':
        n_cluster = KMeans(n_clusters=group, random_state=0).fit(xs)
    elif cluster_method == 'agglomerate':
        n_cluster = AgglomerativeClustering(n_clusters=group).fit(xs)
    else:
        assert NotImplementedError

    labels = n_cluster.labels_
    del xs
    return labels


def group_sum(x, n=32, cluster_method='k_means'):
    b, c, h, w = x.shape
    group_idx = group_cluster(x, group=n, cluster_method=cluster_method)
    init_masks = [torch.zeros(1, 1, h, w).to(x.device) for _ in range(n)]
    for i in range(c):
        idx = group_idx[i]
        init_masks[idx] += x[:, i, :, :].unsqueeze(1)
    return init_masks


class GroupCAM(BaseCAM):
    def __init__(self, 
                 model, 
                 target_layers, 
                #  use_cuda, 
                 groups=32, 
                 cluster_method=None, 
                 is_clip=False):
        super(GroupCAM, self).__init__(model, 
                                       target_layers, 
                                    #    use_cuda, 
                                       groups=groups, 
                                       cluster_method=cluster_method, 
                                       is_clip=is_clip)
        assert cluster_method in [None, 'k_means', 'agglomerate']
        self.cluster = cluster_method
        self.groups = groups
        self.is_clip = is_clip


    def forward(self, inputs, class_idx=None, retain_graph=False, raw_size=None):
        if isinstance(inputs, tuple):
            img, text = inputs[0], inputs[1]
            text = text.cuda()
        else:
            img = inputs
            
        img = img.cuda()
        
        b, c, h, w = img.size()

        if self.is_clip:
            logit = self.model(img,text)
            logit = logit[0]
        else:
            logit = self.model(img)

        if class_idx is None:
            predicted_class = logit.max(1)[-1]
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            predicted_class = torch.LongTensor([class_idx])
            score = logit[:, class_idx].squeeze()

        predicted_class = predicted_class.cuda()
        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value'].data
        activations = self.activations['value'].data

        b, k, u, v = activations.size()

        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)
        activations = weights * activations # this is GradCAM

        if self.cluster is None:
            saliency_map = activations.chunk(self.groups, 1)
            # parallel implement
            saliency_map = torch.cat(saliency_map, dim=0)
            saliency_map = saliency_map.sum(1, keepdim=True)
        else:
            saliency_map = group_sum(activations, n=self.groups, cluster_method=self.cluster)
            saliency_map = torch.cat(saliency_map, dim=0)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

        norm_saliency_map = saliency_map.reshape(self.groups, -1)
        inter_min = norm_saliency_map.min(dim=-1, keepdim=True)[0]
        inter_max = norm_saliency_map.max(dim=-1, keepdim=True)[0]
        norm_saliency_map = (norm_saliency_map-inter_min) / (inter_max - inter_min + 1e-6)
        norm_saliency_map = norm_saliency_map.reshape(self.groups, 1, h, w)

        # modify for clip
        with torch.no_grad():
            if self.is_clip:
                org_score = self.model(blur(img), text)[0] 
                # some sort of enhancement by bluring the none important area
                blur_x = img * norm_saliency_map + blur(img) * (1 - norm_saliency_map)
                blur_score = self.model(blur_x.cuda(), text)[0]
                score = F.relu(blur_score - org_score).unsqueeze(-1).unsqueeze(-1)
            else:
                org_score = self.model(blur(img))[:,class_idx] 
                # some sort of enhancement by bluring the none important area
                blur_x = img * norm_saliency_map + blur(img) * (1 - norm_saliency_map)
                blur_score = self.model(blur_x.cuda())[:,class_idx]
                score = F.relu(blur_score - org_score).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        #     score = F.sigmoid(blur_score - org_score).unsqueeze(-1).unsqueeze(-1)
        self.weights = score.cpu().numpy()
        score_saliency_map = torch.sum(saliency_map * (score+1e-6), dim=0)[0]

        # with torch.no_grad():
        #     base_line = F.softmax(self.model(blur(img).cuda(), text.cuda())[0], dim=-1)[0][predicted_class]
        #     blur_x = img * norm_saliency_map + blur(img) * (1 - norm_saliency_map)
        #     output = self.model(blur_x.cuda(), text.cuda())[0]
        # output = F.softmax(output, dim=-1)
        # score = output[:, predicted_class] - base_line.unsqueeze(0).repeat(self.groups, 1)
        # score = F.relu(score).unsqueeze(-1).unsqueeze(-1)
        # score_saliency_map = torch.sum(saliency_map * score, dim=0, keepdim=True)

        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min().cpu().numpy(), score_saliency_map.max().cpu().numpy()
        # score_saliency_map = (score_saliency_map - score_saliency_map_min) / (
        #         score_saliency_map_max - score_saliency_map_min)

        if score_saliency_map_min == score_saliency_map_max:
            return None

        if raw_size is not None:
            import cv2
            score_saliency_map = cv2.resize(score_saliency_map.cpu().numpy().astype(np.float32), raw_size)
        else:
            score_saliency_map = score_saliency_map.cpu().numpy()
        score_saliency_map = score_saliency_map - np.min(score_saliency_map)    
        if np.max(score_saliency_map) == 0:
            score_saliency_map = score_saliency_map/ (np.max(score_saliency_map) + 1e-6)
        else:
            score_saliency_map = score_saliency_map / np.max(score_saliency_map)
        return score_saliency_map

    def __call__(self, inputs, class_idx=None, retain_graph=False, raw_size=None):
        return self.forward(inputs, class_idx, retain_graph, raw_size)
