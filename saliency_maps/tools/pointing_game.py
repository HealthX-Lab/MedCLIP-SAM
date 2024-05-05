
from torchray.benchmark.pointing_game import PointingGame
import xml.etree.ElementTree as xET
import xmltodict
from easydict import EasyDict
import torch

class PointingGameImageNet(PointingGame):
    """Pointing game benchmark on standard datasets.

    Args:
        dataset (:class:`torchvision.VisionDataset`): The dataset.
        tolerance (int): the tolerance for the pointing game. Default: ``15``.
        difficult (bool): whether to use the difficult subset.
            Default: ``False``.
    """

    def __init__(self, tolerance=15, xml_path='/home/peijie/dataset/ILSVRC2012/val_bbox'):
        num_classes = 1000


        super(PointingGameImageNet, self).__init__(
            num_classes=num_classes, tolerance=tolerance)
        self.dataset = 'imagenet'
        self.box_path = xml_path
            
    def get_mask(self, img_name):
        name = img_name.split('.')[0]
        file_name = f'{self.box_path}/{name}.xml'
        with open(file_name, 'r') as f:
            xml_dict = xmltodict.parse(f.read())
        annotations = EasyDict(xml_dict['annotation'])

        width, height = int(annotations.size.width), int(annotations.size.height)
        
        # genereate bool mask
        mask = torch.zeros(width, height)
        
        objects = annotations.object
        if not isinstance(objects, list):
            objects = [objects]
        # bboxes = []
        for one_object in objects:
            box_d = one_object.bndbox
            box = [int(box_d.xmin), int(box_d.ymin), int(box_d.xmax), int(box_d.ymax)]
            # bboxes.append(box)
            mask[box[0]:box[2]+1, box[1]:box[3]+1] = 1
        mask = mask.bool()
        return mask
        

    def evaluate(self, img_name, label, class_id, point):
        """Evaluate an label-class-point triplet.

        Args:
            label (dict): a label in VOC or Coco detection format.
            class_id (int): a class id.
            point (iterable): a point specified as a pair of u, v coordinates.

        Returns:
            int: +1 if the point hits the object, -1 if the point misses the
                object, and 0 if the point is skipped during evaluation.
        """

        # Skip if testing on the EBP difficult subset and the image/class pair
        # is an easy one.
        mask = self.get_mask(img_name)

        assert mask is not None
        return super(PointingGameImageNet, self).evaluate(mask, point)

