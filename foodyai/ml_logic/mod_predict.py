import os
import json
import importlib
import numpy as np
import cv2
import torch
from detectron2.engine import DefaultPredictor

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.structures import Boxes, BoxMode
from detectron2.config import get_cfg
import pycocotools.mask as mask_util


def prediction_setup(self):
    """
    Setting the config parameters
    """

    self.config = self.get_detectron_config()
    self.model_name = self.config["model_type"]
    self.model = importlib.import_module(f"detectron2.{self.model_name}")
    self.class_to_category = self.get_class_to_category()
    args = default_argument_parser()
    args = args.parse_args(
        """
        --num-gpus 1
        --config-file COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml
        --resume
        MODEL.WEIGHTS weights/model_final.pth
        TEST.AUG.ENABLED False"""
        .split())
    cfg = setup(args)
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    if cfg.TEST.AUG.ENABLED:
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model, batch_size=1)
    self.predictor = Predictor(cfg, model)


def prediction(self, image_path):
    print("Generating for", image_path)
    # read the image
    img = cv2.imread(image_path)
    prediction = self.predictor(img)

    annotations = []
    instances = prediction["instances"]
    if len(instances) > 0:
        scores = instances.scores.tolist()
        classes = instances.pred_classes.tolist()
        bboxes = BoxMode.convert(
            instances.pred_boxes.tensor.cpu(),
            BoxMode.XYXY_ABS,
            BoxMode.XYWH_ABS,
        ).tolist()

        masks = []
        if instances.has("pred_masks"):
            for mask in instances.pred_masks.cpu():
                _mask = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
                _mask["counts"] = _mask["counts"].decode("utf-8")
                masks.append(_mask)

        for idx in range(len(instances)):
            category_id = self.class_to_category[str(classes[idx])] # json converts int keys to str
            output = {
                "image_id": int(os.path.basename(image_path).split(".")[0]),
                "category_id": category_id,
                "bbox": bboxes[idx],
                "score": scores[idx],
            }
            if len(masks) > 0:
                output["segmentation"] = masks[idx]
            annotations.append(output)

    # You can return single annotation or array of annotations in your code.
    return annotations

def get_class_to_category(self):
    class_to_category = {}
    with open("utils/class_to_category.json") as fp:
        class_to_category = json.load(fp)
    return class_to_category

def get_detectron_config(self):
    with open("aicrowd.json") as fp:
        config = json.load(fp)
    return config
