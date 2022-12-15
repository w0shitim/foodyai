from pycocotools.coco import COCO
import torch, detectron2

from detectron2.data.datasets import register_coco_instances


def data_path():
    '''
    create coco file from annotations.json
    '''
    train_annotations_path = '/home/ali_zolfagharian/raw_data/annotations.json'
    train_images_path = '/home/ali_zolfagharian/raw_data/images'

    val_annotations_path = '/home/ali_zolfagharian/raw_data/annotations.json'
    val_images_path = '/home/ali_zolfagharian/raw_data/images'

    train_coco = COCO(train_annotations_path)

    register_coco_instances("training_dataset", {},train_annotations_path, train_images_path)
    register_coco_instances("validation_dataset", {},val_annotations_path, val_images_path)

    print("Registered coco instances")

    return None
