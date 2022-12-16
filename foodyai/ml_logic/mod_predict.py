import os
import json
import importlib
import numpy as np
import pandas as pd
import cv2
#import torch
from detectron2.engine import DefaultPredictor

#from detectron2.evaluation import COCOEvaluator, inference_on_dataset
#from detectron2.data import build_detection_test_loader
from detectron2.structures import Boxes, BoxMode
from detectron2.config import get_cfg
import pycocotools.mask as mask_util

from foodyai.gc_bucket.data import get_class


def prediction_setup(threshold:float, model_path:str, config_path:str,model="model_zoo"):
    """
    Setting the config parameters for the prediction
    - threshold is a float between 0 and 1
    - model_path is the path to the model_final.pth
    - config path is the pâath to the config of the model
    - model is the name of the model used from detectron. By default, model_zoo
    Config and model are both output after training the model
    """
    model_name = model
    model = importlib.import_module(f"detectron2.{model_name}")

    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 323

    cfg.MODEL.DEVICE = "cpu" #if MacOS
    #cfg.MODEL.DEVICE = "cuda" #if Windows (but install cuda in requirements)

    predictor = DefaultPredictor(cfg)

    return predictor



def prediction(predictor:DefaultPredictor,image_path:str,class_to_category:dict,output_filepath:str):
    """
    run the prediction on one image given as input
    - predictor is the output of prediction_setup function
    - image_path is the path to the image to predict
    - class_to_category is a dictionary output from the get_class_to_category function
    - output_file_path is the path you want the prediction output to be saved
    it will be a json file with imageid, categoryid, score, bbox, segmentation
    """
    print("Generating for", image_path)

    img = cv2.imread(image_path)
    prediction = predictor(img)

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
            category_id = class_to_category[str(classes[idx])] # json converts int keys to str
            output = {
                "image_id": int(os.path.basename(image_path).split(".")[0]),
                "category_id": category_id,
                "bbox": bboxes[idx],
                "score": scores[idx],
            }
            if len(masks) > 0:
                output["segmentation"] = masks[idx]
            annotations.append(output)

    with open(output_filepath, "w") as fp:
        json.dump(annotations, fp)

    df = pd.read_json(output_filepath)

    return df[['image_id','category_id','score']]

def get_class_to_category():
    """
    open the class_to_category json file
    """
    config_path = './raw_data/class_to_category.json'
    if os.path.isfile(config_path) == False:
        get_class(download_to_disk = True,
                destination_file_name = './raw_data/class_to_category.json')
        class_to_cat = './raw_data/class_to_category.json'
    else:
        class_to_cat = './raw_data/class_to_category.json'

    #class_to_category = {}
    with open(class_to_cat) as fp:
        class_to_category = json.load(fp)
    return class_to_category
