import numpy as np
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection

# https://ieeexplore.ieee.org/document/9462463


class LayerCAM(BaseCAM):
    def __init__(
            self,
            model,
            target_layers,
            use_cuda=False,
            reshape_transform=None,
            is_clip=False,
            ):
        super(
            LayerCAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform,
            is_clip=is_clip,
            )


    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        self.weights = np.maximum(grads, 0)
        spatial_weighted_activations = self.weights * activations

        if eigen_smooth:
            cam = get_2d_projection(spatial_weighted_activations)
        else:
            cam = spatial_weighted_activations.sum(axis=1)
        return cam
