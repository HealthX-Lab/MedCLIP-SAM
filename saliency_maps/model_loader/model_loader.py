from model_loader.clip_loader import load_clip
import torchvision
from torchvision.transforms import transforms

def load_model(model_name: str, is_clip: bool =False, custom_clip: bool = False, image_size: tuple = (224, 224)):
    if is_clip:
        model, preprocess, target_layer, cam_trans, clip = load_clip(model_name, custom=custom_clip)
        tokenizer = clip.tokenize
    else:
        model = torchvision.models.__dict__[model_name](pretrained=True)
        target_layer = model.layer4[-1] # for resnet50, should be 'layer4'
        cam_trans = None
        tokenizer = None
        preprocess = transforms.Compose([transforms.Resize(image_size),
                                            # transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])])
    return model, preprocess, target_layer, cam_trans, tokenizer