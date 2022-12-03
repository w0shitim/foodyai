import torch, detectron2

# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import pandas as pd
import os, json, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
#from detectron2.utils.visualizer import Visualizer
#from detectron2.data import MetadataCatalog, DatasetCatalog

# For reading annotations file
from pycocotools.coco import COCO

from google.cloud import storage

client = storage.Client()
bucket = client.bucket('foodygs')
blob = bucket.blob('Nutrition/nutrition.csv')

contents = blob.download_as_string()

print("Downloaded storage object") #{} from bucket {} as the following string: {}.".format(blob_name, bucket_name, contents)




blob_train_annotations = bucket.blob('foodyai_data/Training_2/annotations.json')
blob_train_images = bucket.blob('foodyai_data/Training_2/images')

blob_val_annotations = bucket.blob('foodyai_data/Validation_2/annotations.json')
blob_val_images = bucket.blob('foodyai_data/Validation_2/images')

# Loading the datasets in coco format and registering them as instances
train_annotations_path = blob_train_annotations.download_as_string()
train_images_path = blob_train_images.download_as_string()

val_annotations_path = blob_val_annotations.download_as_string()
val_images_path = blob_val_images.download_as_string()

train_coco = COCO(train_annotations_path)

from detectron2.data.datasets import register_coco_instances

register_coco_instances("training_dataset", {},train_annotations_path, train_images_path)
register_coco_instances("validation_dataset", {},val_annotations_path, val_images_path)


bucket_name = 'your-bucket-name'
prefix = 'your-bucket-directory/'
dl_dir = 'your-local-directory/'

storage_client = storage.Client()
bucket = storage_client.get_bucket(bucket_name)
blobs = bucket.list_blobs(prefix=prefix)  # Get list of files
for blob in blobs:
    filename = blob.name.replace('/', '_')
    blob.download_to_filename(dl_dir + filename)  # Download


train_annotations_path = '/home/patricia/code/w0shitim/foodyai/raw_data/Training_2/annotations.json'
train_images_path = '/home/patricia/code/w0shitim/foodyai/raw_data/Training_2/images'

val_annotations_path = '/home/patricia/code/w0shitim/foodyai/raw_data/Validation_2/annotations.json'
val_images_path = '/home/patricia/code/w0shitim/foodyai/raw_data/Validation_2/images'

train_coco = COCO(train_annotations_path)

register_coco_instances("training_dataset", {},train_annotations_path, train_images_path)
register_coco_instances("validation_dataset", {},val_annotations_path, val_images_path)

print("Registered coco instances")
