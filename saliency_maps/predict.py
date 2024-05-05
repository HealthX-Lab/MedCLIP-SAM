import subprocess

import numpy as np
from cog import BasePredictor, Input, Path
from PIL import Image

from model_loader.clip_loader import load_clip
from tools.cam import CAMWrapper
from tools.drawing import Drawer


class Predictor(BasePredictor):
    def setup(self):

        subprocess.run(["mkdir", "/root/.cache/clip"])
        subprocess.run(["mv", "RN50x16.pt", "/root/.cache/clip"])
        subprocess.run(["mv", "ViT-B-16.pt", "/root/.cache/clip"])

    def predict(
        self,
        input_image: Path = Input(description="Path to an image to investigate clip on"),
        input_text: str = Input(description="What to look for in the image", default="An object"),
        clip_version: str = Input(description="Name of the clip model",
                                    default="RN50x16",
                                    choices=["RN50x16", "ViT-B/16"],
        ),
        drop: bool = Input(description="Use only part of the channels", default=True),
        topk: int = Input(description="Number of channels used by gscorecam (ignored when drop=False)",
                            ge=1,
                            le=3072.0,
                            default=300,
        ),
        overlay_output: bool = Input(description="Show the output heatmap on-top of the input", default=True),
    ) -> Path:
        cam_version = "gscorecam"
        resize = "adapt"
        custom_clip = False
        drop = bool(drop)  # Whether to drop the channels in scorecam"
        batch_size = 128  # "Batch size for scorecam based methods."
        is_clip = True  # "Whether to use clip model."

        clip_model, preprocess, target_layer, cam_trans, clip = load_clip(
            str(clip_version), resize=resize, custom=custom_clip
        )

        cam = CAMWrapper(
            clip_model,
            target_layers=[target_layer],
            tokenizer=clip.tokenize,
            cam_version=str(cam_version),
            preprocess=preprocess,
            topk=int(topk),
            drop=drop,
            channels={},
            cam_trans=cam_trans,
            is_clip=is_clip,
            batch_size=batch_size,
        )

        input_image = Image.open(str(input_image))
        if input_image.mode != "RGB":
            input_image = input_image.convert("RGB")

        output = cam((input_image, str(input_text)), 0, input_image.size)

        if bool(overlay_output):
            output = Drawer.overlay_cam_on_image(input_image, output, use_rgb=True)
        else:
            print(output.shape)
            output = Image.fromarray(output * 255).convert("L")

        output_path = "output.png"
        output.save(output_path)
        return Path(output_path)
