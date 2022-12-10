from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper

class MyTrainer(DefaultTrainer):
    """
    Overriding the build_train_loader method
    from the Default Trainer that includes augmentations """
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=[T.ResizeShortestEdge(short_edge_length=(640, 672, 704, 736, 768, 800), max_size=1333, sample_style='choice'),
                                                                  T.RandomFlip(prob=0.3, horizontal=True, vertical=False),
                                                                 T.RandomLighting(0.7),
                                                                  T.RandomContrast(0.6, 1.3),
                                                                  T.RandomSaturation(0.8, 1.4)])
        return build_detection_train_loader(cfg, mapper=mapper)
