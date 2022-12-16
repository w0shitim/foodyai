# Foodyai interface

This file is to run 3 functions:
- train()
- evaluate()
- predict()

These 3 functions are re using the functions created in the ml_logic folder.

## train()
This function has been run only in the Google Cloud Virtual Machine. Since the pretrained model (detectron2) requires GPU.

CPU can be used but it is not recommended since it will take ages to train.

To avoid mistake, the function parameter is set to false for training it again.

The model was trained with data augmentation. However, the ETA will considerably increase.

## evaluate()
Has been run in the vm only since the evaluation is done one almost 1000 images.

It is then preferable to run this part in a Google Cloud Virtual Machine.

## predict()
The prediction is more special.
The process is the following:
- define the image path
- the model path and config path are given. Due to the size of the model, we saved it on our Google Cloud Bucket and import to our package when needed (2 possibilites, download to raw_data or to memory then discard it)
- set up the prediction parameters (specific to the pre trained model)
- run the prediction. the output is a data frame containing: image id, category id, score, box, and segmentation
- from the annotations.json file, create a dataframe category_id, category_name
- perform a merge between the 2 dataframe to keep only the category predicted and get the name out of it
- turn the names into a list then string
- give it to the spoonacular API to detect food item
- extract nutrition fact into a dataframe

## Conclusion
For the prediction, while requesting the API there surely is a better way to do it (in this case we are requesting the API too much). But in a short amount of time, that is the best trick we could do.
