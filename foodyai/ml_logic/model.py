
from google.cloud import storage

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


def download_file(bucket_name = 'foodygs',
                              blob_name = 'Nutrition/nutrition.csv',
                              download_to_disk = False,
                              destination_file_name = '../raw_data/data.csv'):

    """Download a file from Google Cloud Storage.
    If download_to_disk = False then it will save to memory.
    If download_to_disk = True then it will save to your local disk.
    """

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(blob_name)

    if download_to_disk == True:

        blob.download_to_filename(destination_file_name)
        print(
            "Downloaded storage object {} from bucket {} to local file {}.".format(
            blob_name, bucket_name, destination_file_name
        )
    )
    contents = ''

    if download_to_disk == False:
        print(f"retrieving {blob_name} from gcloud")
        contents = blob.download_as_string()
        print(f"received... test")
    return contents

print("Files downloaded")


def blob_coco_register():
    print('----- blob coco register -----')
    storage_client = storage.Client()
    bucket = storage_client.bucket('foodygs')
    print('-- storage and bucket has been defined')

    str_folder_name_on_gcs = 'foodyai_data/Training_2/images/'
    blob_train_annotations = bucket.blob('foodyai_data/Training_2/annotations.json')
    blobs = bucket.list_blobs(delimiter='/',prefix=str_folder_name_on_gcs)
    print('-- training annotations and images blobs has been created')

    blob_val_annotations = bucket.blob('foodyai_data/Validation_2/annotations.json')
    blobs_val = bucket.list_blobs(delimiter='/',prefix='foodyai_data/Validation_2/images/')
    print('-- validation annotations and images blobs has been created')

    print('-- loading the datasets in coco format and registering them as instances')
    train_annotations_path = blob_train_annotations.download_as_string()
    val_annotations_path = blob_val_annotations.download_as_string()
    print('-- annotation downloaded as string')

    train_coco = COCO(train_annotations_path)
    print('-- made annotation coco format')

    print('-- register coco instances training dataset begin')
    register_coco_instances("training_dataset", {},train_annotations_path, blobs)

    print('-- register coco instances validation dataset begin')

    register_coco_instances("validation_dataset", {},val_annotations_path, blobs_val)
    print('-- register coco instances end ')



def custom_config(training_dataset = ("training_dataset",),
                  num_workers = 2,
                  trained_model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                  num_classes = 323,
                  batch_size = 128,
                  ims_per_batch = 10,
                  learning_rate = 0.00025,
                  max_iter = 20,
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
    download_file()
    blob_coco_register()
    custom_config()
    #evaluate_model()
