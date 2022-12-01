
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg

from detectron2.engine import DefaultTrainer

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

import os

from pycocotools.coco import COCO
from detectron2.data.datasets import register_coco_instances

train_annotations_path = '/home/patricia/code/w0shitim/foodyai/raw_data/Training_2/annotations.json'
train_images_path = '/home/patricia/code/w0shitim/foodyai/raw_data/Training_2/images'

val_annotations_path = '/home/patricia/code/w0shitim/foodyai/raw_data/Validation_2/annotations.json'
val_images_path = '/home/patricia/code/w0shitim/foodyai/raw_data/Validation_2/images'

train_coco = COCO(train_annotations_path)

register_coco_instances("training_dataset", {},train_annotations_path, train_images_path)
register_coco_instances("validation_dataset", {},val_annotations_path, val_images_path)

print("Registered coco instances")

def custom_config(training_dataset = ("training_dataset",),
                  num_workers = 2,
                  trained_model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                  num_classes = 323,
                  batch_size = 128,
                  ims_per_batch = 10,
                  learning_rate = 0.00025,
                  max_iter = 2,
                  output_dir = "logs/"):

    """Initialize pre-trained model"""

    cfg = get_cfg()
    cfg.DATALOADER.NUM_WORKERS = num_workers

    # Get configuration from model_zoo. Check the model zoo and use any of the models
    cfg.merge_from_file(model_zoo.get_config_file(trained_model))

    # Loading pre trained weights # Let training initialize from model zoo
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(trained_model)

    # Model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size

    # Solver
    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.BASE_LR = learning_rate
    cfg.SOLVER.MAX_ITER = max_iter

    #Datasets
    cfg.DATASETS.TRAIN = training_dataset
    cfg.DATASETS.TEST = ()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.OUTPUT_DIR = output_dir

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    trainer.train()

    return cfg, trainer



def evaluate_model(validation_dataset = "validation_dataset",
                   model_path = "model_final.pth",
                   thresh_test = 0.5):

    """Evaluate model with COCOEvaluater"""

    cfg, trainer = custom_config()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh_test

    evaluator = COCOEvaluator(validation_dataset, output_dir="./output")
    val_loader = build_detection_test_loader(cfg, validation_dataset)
    valResults = inference_on_dataset(trainer.model, val_loader, evaluator)

    return valResults, cfg, trainer

if __name__ == '__main__':
    custom_config()
    evaluate_model()
