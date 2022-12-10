import pandas as pd

from foodyai.ml_logic.mod_predict import *

from foodyai.ml_logic.category import *


def train ():
    pass

def evaluate():
    pass

def predict():
    """
    input an image and output a category name
    re use function defined in ml_logic
    """

    threshold = 0
    model_path = "logs/model_final.pth"
    config_path = "logs/config.yml"

    predictor = prediction_setup(threshold,
                                 model_path,
                                 config_path,
                                 model="model_zoo")


    class_to_category = get_class_to_category()

    image_path = ''
    output_filepath = 'logs/predict_rslt.json'

    df_pred = prediction(predictor,
               image_path,
               class_to_category,
               output_filepath)

    df_category = get_category()

    df_rslt = df_pred.merge(df_category, on='category_id', how='left')
    df_rslt = df_rslt[['category_id','name_readable']]
