# Get some objects from Google Cloud Storage Bucket

After training the model, the output is 2 files:
- model_final.pth
- config.yml
Both necessary to run prediction.

## load_model.py
It loads both files from the GCS Bucket. There are 2 choice of import:
- download in this package (in this case raw_data)
- or keep it in memory the time of the prediciton

## data.py
This python file get 2 important json file from the GCS buckets necessary for the predictions:
- class_to_category.json needed to create 1 class per category for the prediction
- annotations.json which is a file containing image information (category id and name, image id, segmentation, bbox) used to extract the category id and name to get nutrition fact
