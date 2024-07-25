import numpy as np
import torch
import ttach as tta
from typing import Callable, List, Tuple
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
from pytorch_grad_cam.utils.image import scale_cam_image, normalize2D, resize2D, cls_reshpae
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2

class BaseCAM:
    def __init__(self,
                 model: torch.nn.Module,
                 target_layers: List[torch.nn.Module],
                 use_cuda: bool = True,
                 reshape_transform: Callable = None,
                 compute_input_gradient: bool = False,
                 uses_gradients: bool = True,
                 is_clip: bool = False,
                 is_transformer: bool = False,
                 model_domain: str = None,
                 **kwargs,
                 ) -> None:
        self.model = model.eval()
        self.target_layers = target_layers
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform=None) #! set reshape_transform to None, because we will use the reshape_transform of inside BaseCAM to aviod unexpected function call of reshape_transform 
        
        allowed_keys = {'drop', 'topk', 'img_size', 'channels', 'groups'}
        self.__dict__.update((key, value) for key, value in kwargs.items() if key in allowed_keys)
        self.is_clip = is_clip
        self.is_transformer = is_transformer
        self.model_domain = model_domain

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    def get_cam_weights(self,
                        input_tensor: torch.Tensor,
                        target_layers: List[torch.nn.Module],
                        # targets: List[torch.nn.Module],
                        targets: np.array,
                        activations: torch.Tensor,
                        grads: torch.Tensor) -> np.ndarray:
        raise NotImplementedError("Base cam should not be used by itself.")

    def dim_mapper(self, activation_shape: tuple, weight_shape: tuple) -> str:
        dim_s = 'ncwh'
        weight_length = len(weight_shape)
        start_dim = activation_shape.index(weight_shape[0])
        return dim_s[start_dim: start_dim+weight_length]
    
    
    def getRawActivation(self, input_tensor, target_category=None, img_size=None):
        if self.is_clip:
            output = self.activations_and_grads(input_tensor)[1] # per text logit of clip
        else:
            output = self.activations_and_grads(input_tensor)

        # self.model.zero_grad()
        # output.backward(retain_graph=True)

        activations = self.activations_and_grads.activations[-1].cpu().data.numpy()
        # grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()

        cam = activations[0]

        cam = np.maximum(cam, 0)

        result = []
        # fix bug in cv2 that it does not support type 23. (float16)
        if cam.dtype == 'float16':
            cam = cam.astype(np.float32)
        input_shape = img_size if self.is_clip else input_tensor.shape[-2:][::-1]
        for img in cam:
            img = cv2.resize(img, input_shape)
            img = img - np.min(img)
            img = img/ (np.max(img) + 1e-8) 
            result.append(img)
        result = np.float32(result)
        return result


    def get_cam_image(self,
                      input_tensor: torch.Tensor,
                      target_layer: torch.nn.Module,
                    #   targets: List[torch.nn.Module],
                      targets: np.array,
                      activations: torch.Tensor,
                      grads: torch.Tensor,
                      eigen_smooth: bool = False) -> np.ndarray:

        self.weights = self.get_cam_weights(input_tensor,
                                       target_layer,
                                       targets,
                                       activations,
                                       grads) # usually (n, c), but with einsum, now is support any shapes that match the corresponding dimension of activation. i.e., (c, w, h)
        # weighted_activations = weights[:, :, None, None] * activations
        weight_dim = self.dim_mapper(activations.shape, self.weights.shape)
        weighted_activations = np.einsum(f"{weight_dim},ncwh->ncwh", self.weights.astype(np.float32), activations.astype(np.float32))

        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def forward(self,
                input_tensor: torch.Tensor,
                # targets: List[torch.nn.Module],
                targets: np.array,
                eigen_smooth: bool = False) -> np.ndarray:


        if self.is_clip:
            if self.cuda:
                input_tensor = (input_tensor[0].cuda(), input_tensor[1].cuda())


            if self.compute_input_gradient:
                input_tensor = (torch.autograd.Variable(input_tensor[i],
                                                    requires_grad=True)   
                                for i in len(input_tensor))
            outputs = self.activations_and_grads(input_tensor)[1] # per text logit of clip    
        else:
            if self.cuda:
                input_tensor = input_tensor.cuda()

            if self.compute_input_gradient:
                input_tensor = torch.autograd.Variable(input_tensor,
                                                    requires_grad=True)


            outputs = self.activations_and_grads(input_tensor)


        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            # targets = [ClassifierOutputTarget(category) for category in target_categories]
            targets = target_categories
        else:
            target_categories = targets

        if self.uses_gradients:
            self.model.zero_grad()
            # loss = sum(target(output) for target, output in zip(targets, outputs))
            loss = sum(outputs[:, target_categories])
            loss.requires_grad_(True)
            loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   targets,
                                                   eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)

    def get_target_width_height(self,
                                input_tensor: torch.Tensor) -> Tuple[int, int]:
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(
            self,
            input_tensor: torch.Tensor,
            # targets: List[torch.nn.Module],
            targets: np.array,
            eigen_smooth: bool) -> np.ndarray:
        if(self.model_domain == "biomedclip"):
            for a in range(len(self.activations_and_grads.activations)):
                self.activations_and_grads.activations[a] = self.activations_and_grads.activations[a].permute(1,0,2) 
            for a in range(len(self.activations_and_grads.gradients)):
                self.activations_and_grads.gradients[a] = self.activations_and_grads.gradients[a].permute(1,0,2)
        activations_list = [self.reshape_transform(a).numpy() if self.reshape_transform is not None else a.numpy()
                            for a in self.activations_and_grads.activations]
        self.activations = activations_list
        if self.is_transformer: # use the gradient of CLS token for gradient
            grads_list = [cls_reshpae(g) for g in self.activations_and_grads.gradients]
        else:
            grads_list = [self.reshape_transform(g).numpy() if self.reshape_transform is not None else g.numpy()
                            for g in self.activations_and_grads.gradients]
        
        if hasattr(self, 'img_size'):
            target_size = self.img_size
        else:
            target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(input_tensor,
                                     target_layer,
                                     targets,
                                     layer_activations,
                                     layer_grads,
                                     eigen_smooth)
            cam = np.maximum(cam, 0) #* The difference between having this line is small, i.e., in COCO val, with this line is 20.83% and without this line is 20.86% 
            # normalize and resize cam
            cam = np.array([normalize2D(img) for img in cam])
            if target_size is not None: #* no resize such that we can post process it later
                cam = np.array([resize2D(img, target_size) for img in cam])
            # scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(cam[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer: np.ndarray) -> np.ndarray:
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        # return scale_cam_image(result)
        return np.array([normalize2D(cam) for cam in result])

    def forward_augmentation_smoothing(self,
                                       input_tensor: torch.Tensor,
                                    #    targets: List[torch.nn.Module],
                                    targets: np.array,
                                       eigen_smooth: bool = False) -> np.ndarray:
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        cams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor,
                               targets,
                               eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(self,
                 input_tensor: torch.Tensor,
                 targets: List[torch.nn.Module] = None,
                 aug_smooth: bool = False,
                 eigen_smooth: bool = False) -> np.ndarray:

        # Smooth the CAM result with test time augmentation
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(
                input_tensor, targets, eigen_smooth)

        return self.forward(input_tensor,
                            targets, eigen_smooth)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True
