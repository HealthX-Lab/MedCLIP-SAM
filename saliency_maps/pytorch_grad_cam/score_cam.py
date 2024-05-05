import torch
import tqdm
import random
from pytorch_grad_cam.base_cam import BaseCAM


class ScoreCAM(BaseCAM):
    def __init__(self, 
                 model, 
                 target_layers, 
                 use_cuda=False, 
                 reshape_transform=None, 
                 is_clip=False, 
                 drop=False, 
                 mute=False, 
                 topk:int or None=None,
                 channels=None,
                 batch_size: int = 128,
                 is_transformer: bool = False,
                 model_domain=None):
        super(ScoreCAM, self).__init__(model, 
                                       target_layers, 
                                       use_cuda, 
                                       reshape_transform=reshape_transform, 
                                       is_clip=is_clip, 
                                       drop=drop, 
                                       mute=mute,
                                       batch_size=batch_size,
                                       is_transformer=is_transformer
                                       )
        self.topk = topk
        self.channels = channels
        self.drop = drop
        self.mute = mute
        self.is_clip = is_clip
        self.model_domain = model_domain
        # self.batch_size = batch_size

        if len(target_layers) > 0:
            print("Warning: You are using ScoreCAM with target layers, "
                  "however ScoreCAM will ignore them.")

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        targets,
                        activations,
                        grads):
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
        upsampled = (upsampled - mins) / (maxs - mins)

        input_tensors = img_tensor[:, None, :, :].cpu()*upsampled[:, :, None, :, :]

        BATCH_SIZE = self.batch_size if hasattr(self, "batch_size") else 128
        # BATCH_SIZE = 32

        if self.drop:
            #randomly select k channels
            k = self.topk
            indices = torch.tensor(random.sample(range(input_tensors.shape[1]), k))
            # indices = torch.tensor(indices)
            input_tensors = input_tensors[:,indices]
        elif self.channels is not None and self.channels:
            indices = self.channels
            input_tensors = input_tensors[:, indices]


        scores = []
        if isinstance(targets, int):
            targets = [targets]
        with torch.no_grad():
            for target, tensor in zip(targets, input_tensors):
                for i in tqdm.tqdm(range(0, tensor.size(0), BATCH_SIZE), disable=self.mute):
                    batch = tensor[i: i + BATCH_SIZE, :]
                    if self.is_clip:
                        outputs =  self.model(batch.cuda(), text_tensor.cuda())[0].cpu().numpy()[:, target]
                    else:
                        outputs = self.model(batch.cuda()).cpu().numpy()[:, target]
                    # outputs = [target(o).cpu().item() for o in self.model(batch)]
                    scores.extend(outputs)
            scores = torch.Tensor(scores)
            if scores.isnan().any():
                scores = scores.nan_to_num(nan=0.0) # fix nan bug in clip implementation
            
            if self.drop or (self.channels is not None and self.channels):
                # place the 10%(Or handpick channels) score back to the weight
                emtpy_score = torch.zeros(activations.shape[1])
                emtpy_score[indices] = scores
                scores = emtpy_score.view(activations.shape[0], activations.shape[1])
            else:
                scores = scores.view(activations.shape[0], activations.shape[1])
            
            scores = scores.view(activations.shape[0], activations.shape[1])
            weights = torch.nn.Softmax(dim=-1)(scores)
            return weights.numpy()
