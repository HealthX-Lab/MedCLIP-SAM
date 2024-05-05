import torch
import pickle
import shutil
import os.path as osp
import warnings
from functools import partial
from collections import OrderedDict

#* For CLIP ViT
def reshape_transform(tensor, height=None, width=None):
    if height or width is None:
        grid_square = len(tensor) - 1
        if grid_square ** 0.5 % 1 == 0:
            height = width = int(grid_square**0.5)
        else:
            raise ValueError("Heatmap is not square, please set height and width.")
    result = tensor[1:, :, :].reshape(
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.permute(2, 0, 1)
    return result.unsqueeze(0)

def load_clip(clip_version, resize='adapt', custom=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(clip_version)
    if 'vit' in clip_version.lower() and not custom: #* This is no necessary, for experimental usage, hila CLIP will hook all attentions.
        from hila_clip import clip
        clip_model, preprocess = clip.load(clip_version, device=device, jit=False)

    elif custom:
        from hila_clip import clip
        clip_model, preprocess = clip.load(clip_version, device=device, jit=False)            

    else:
        import clip
        clip_model, preprocess = clip.load(clip_version, device=device)

    if clip_version.startswith("RN"):
        target_layer = clip_model.visual.layer4[-1]
        cam_trans = None
    else:
        target_layer = clip_model.visual.transformer.resblocks[-1]
        cam_trans = reshape_transform

    if resize == 'raw': # remove clip resizing
        if not custom:
            raise Exception("Raw input needs to use custom clip.") 
        preprocess.transforms.pop(0)
        preprocess.transforms.pop(0)
    elif resize == 'adapt': # adapt to clip size
        from torchvision import transforms
        crop_size = preprocess.transforms[1].size # resize to crop size so that no information will be cropped
        preprocess.transforms.insert(0, transforms.Resize(crop_size))
    # clip_model = torch.nn.DataParallel(clip_model)
    return clip_model, preprocess, target_layer, cam_trans, clip

def load_checkpoint(fpath):
    r"""Load checkpoint.

    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.

    Args:
        fpath (str): path to checkpoint.

    Returns:
        dict

    Examples::
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError("File path is None")

    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))

    map_location = None if torch.cuda.is_available() else "cpu"

    try:
        checkpoint = torch.load(fpath, map_location=map_location)

    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )

    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise

    return checkpoint

def load_clip_from_checkpoint(checkpoint, model):
    r"""Load pretrianed weights to model.

    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.

    Examples::
        >>> weight_path = 'log/my_model/model-best.pth.tar'
        >>> load_pretrained_weights(model, weight_path)
    """
    checkpoint = load_checkpoint(checkpoint)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]  # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            f"Cannot load {checkpoint} (check the key names manually)"
        )
    else:
        print(f"Successfully loaded pretrained weights from {checkpoint}")
        if len(discarded_layers) > 0:
            print(
                f"Layers discarded due to unmatched keys or size: {discarded_layers}"
            )
        return model
    

# def load_clip_from_checkpoint(checkpoint, model):
#     # checkpoint = torch.load(checkpoint, map_location='cpu')
#     checkpoint = load_checkpoint(checkpoint)

#     # # Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"] 
#     # checkpoint['model_state_dict']["input_resolution"] = model.input_resolution #default is 224
#     # checkpoint['model_state_dict']["context_length"] = model.context_length # default is 77
#     # checkpoint['model_state_dict']["vocab_size"] = model.vocab_size 

#     model.load_state_dict(checkpoint['model_state_dict'])
#     return model