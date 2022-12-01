
from detectron2.config import get_cfg

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

import os

cfg = get_cfg()


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01 # set a custom testing threshold



evaluator = COCOEvaluator("validation_dataset", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "validation_dataset")

#Need to import trainer from model.py
valResults = inference_on_dataset(trainer.model, val_loader, evaluator)
