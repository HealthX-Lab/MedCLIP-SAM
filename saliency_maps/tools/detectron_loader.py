from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
import cv2
import numpy as np
class DetectronModelZoo():
    def __init__(self,
                threshold=0.5, 
                # cfg_name="COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
                cfg_name="PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml"
                ):
        self.cfg_name = cfg_name
        self.threshold = threshold

    def load_model(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.cfg_name))
        # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.cfg_name)
        # predictor = DefaultPredictor(cfg)
        self.detector = DefaultPredictor(cfg)
        self.dataset_classes = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
        # import ipdb; ipdb.set_trace()

    def detect(self, img_path):
        im = cv2.imread(img_path)
        prediction_result = self.detector(im)
        proposals = prediction_result["instances"].pred_boxes.tensor.cpu().numpy()
        if len(proposals) == 0: # set default candidate to center of the image
             proposals = np.array([[im.shape[0]/3, im.shape[1]/3, im.shape[0]*2/3, im.shape[1]*2/3]])
        class_names = [
            self.dataset_classes[i]
            for i in prediction_result["instances"].pred_classes.cpu().numpy()
        ]

        return proposals, class_names

if __name__ == "__main__":
    detector = DetectronModelZoo()
    detector.load_model()
    proposals, class_names = detector.detect('sample.jpg')
    import ipdb; ipdb.set_trace()

    