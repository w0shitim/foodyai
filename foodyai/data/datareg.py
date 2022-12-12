# in this file there are three function, for retrieving basic foodygs dataframes, images and downloading file

from google.cloud import storage
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import urllib3
import urllib.request
import os
import cv2
from tempfile import NamedTemporaryFile
from google.cloud import storage
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances

from pycocotools.coco import COCO
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO
import json




def get_dataframes():
    #retrieving basic needed data from gcloud for foodyai project
    jsonSeries = pd.read_json('gs://foodygs/foodyai_data/Training_2/annotations.json',typ='series')
    # jsonseries has three dataframe and it could be retrieved as the following
    categories = pd.DataFrame(jsonSeries.categories)  
    print(f'categories information containing {len(categories)} rows has been received')
    images = pd.DataFrame(jsonSeries.images)
    print(f'image information containing {len(images)} rows has been received')
    annotations = pd.DataFrame(jsonSeries.annotations)    
    print(f'annotations information containing {len(annotations)} rows has been received')
    nutrition = pd.read_csv('gs://foodygs/Nutrition/nutrition.csv', sep=",", index_col=0)
    print(f'nutrition information containing {len(nutrition)} has been received')
    
    return categories,images,annotations,nutrition


def get_an_image(img_url_and_name='https://storage.cloud.google.com/foodygs/foodyai_data/Training_2/images/006316.jpg'):
    #receiving one image from gcloud by a given address
    
    # resp = urllib.request.urlopen(img_url_and_name)
    # image = np.asarray(bytearray(resp.read()), dtype="uint8")
    # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.imread(img_url_and_name, 0)
    print(type(image))
    return image

# get_an_image('https://storage.cloud.google.com/foodygs/foodyai_data/Training_2/images/006316.jpg')




# setup_logger()

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


    # contents = download_file(bucket_name = 'foodygs',
    #                               blob_name = 'Nutrition/nutrition.csv', 
    #                               download_to_disk = False).decode("utf-8")


    # contents = StringIO(contents)
    # data = pd.read_csv(contents, sep=",").reset_index()

    # print(f"example: the serving size of the Cornstarch is {data[data['name']=='Cornstarch']['serving_size'][0]}")



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
    
    
    




    
    
# storage_client = storage.Client()
# bucket = storage_client.bucket('foodygs')


# # train_annotations_path = 'Training_2/annotations.json' 
# # train_images_path = 'Training_2/images'

# # val_annotations_path = 'Validation_2/annotations.json'
# # val_images_path = 'Validation_2/images'

# # train_coco = COCO(train_annotations_path)

# str_folder_name_on_gcs = 'foodyai_data/Training_2/images/'

# blob_train_annotations = bucket.blob('foodyai_data/Training_2/annotations.json')
# blobs = bucket.list_blobs(delimiter='/',prefix=str_folder_name_on_gcs)

# # blob_train_images = bucket.list_blobs(str_folder_name_on_gcs)
# # blob_val_annotations = bucket.blob('foodyai_data/Validation_2/annotations.json')
# # blob_val_images = bucket.list_blobs('foodyai_data/Validation_2/images/')
# # Loading the datasets in coco format and registering them as instances
# print('-----')
# train_annotations_path = blob_train_annotations.download_as_string()
# print('annotation retrieved')
# # print(blob_train_images)
# # train_images_path = blob_train_images.download_as_string()
# i=0
# # for page in blobs.pages:
# #     for blob in page:
# #         i+=1
# #         print(i)
# #         blob.download_as_string()
#     # if not blob.name.endswith('/'):
#     #     # This blob is not a directory!
#     #     # print(f'Downloading file [{blob.name}]')
#     #     print(i)
#     #     blob.download_as_string()
        
        
# # val_annotations_path = blob_val_annotations.download_as_string()
# # val_images_path = blob_val_images.download_as_string()

# train_coco = COCO(train_annotations_path)
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("training_dataset", {},train_annotations_path, blobs)
# # register_coco_instances("validation_dataset", {},val_annotations_path, val_images_path)

