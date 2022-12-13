#import pandas as pd

from foodyai.ml_logic.model import *
from foodyai.ml_logic.mod_predict import *
from foodyai.ml_logic.category import *
from foodyai.data.data_source import data_path
from foodyai.ml_logic.data_aug import MyTrainer


def train(data_aug=False,train_again=False):
    '''
    train the model if not already trained
    '''

    #get the data from google cloud stroage bucket
    data_path()

    #train the model if model_final.pth doesn't exist yet
    if train_again == True:
        #model_path = 'logs/model_final.pth'
        #if os.path.isfile(model_path) == False:

        cfg = custom_config(training_dataset = ("training_dataset",),
                  num_workers = 2,
                  trained_model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                  num_classes = 323,
                  batch_size = 128,
                  ims_per_batch = 10,
                  learning_rate = 0.00025,
                  max_iter = 50000)

        #by default, data augmentation is set to False because of increase training time
        if data_aug==False:
            model_train(output_dir = "logs/",
                    trainer_to_choose = DefaultTrainer,
                    cfg=cfg)

        else:
            model_train(output_dir = "logs/",
                    trainer_to_choose = MyTrainer,
                    cfg=cfg)


def evaluate():
    '''
    evaluate the preformance of the model
    try different thresh_test to find sweet spot (best performance)
    '''
    valResults, cfg, trainer = evaluate_model(validation_dataset = "validation_dataset",
                   model_path = "model_final.pth",
                   thresh_test = 0.8)

    return valResults

def predict(image_path:str):
    """
    input an image and output a category name
    re use function defined in ml_logic
    """

    #setup parameters for predictor
    threshold = 0
    model_path = "logs/model_final.pth"
    config_path = "logs/config.yml"

    #run predictor
    predictor = prediction_setup(threshold,
                                 model_path,
                                 config_path,
                                 model="model_zoo")


    #open class_to_category json file
    class_to_category = get_class_to_category()

    #setup parameters for prediction
    output_filepath = 'logs/predict_rslt.json'

    #run prediction
    df_pred = prediction(predictor,
               image_path,
               class_to_category,
               output_filepath)

    #get the category as dataframe
    df_category = get_category()

    #left join to get only food category from prediction output
    df_rslt = df_pred.merge(df_category, on='category_id', how='left')
    df_rslt = df_rslt[['category_id','name_readable']]

    #turn name_readble into list
    categories = df_rslt['name_readable'].tolist()

    #clean the list to remove useless terms
    cat_cleaned = preprocessing(categories)

    #define params for API

    params = {'apiKey': API_KEY}

    #use api to detect food items
    food_items = detect_food(cat_cleaned,API_KEY,BASE_URL,params)

    #get food information from API call
    lst_info = get_food_info(food_items,BASE_URL)

    #list of all nutrition facts wanted
    lst_nut_fact = ['sodium','Saturated Fat','Carbohydrates','Fiber','Calories','Cholesterol']

    #output dataset of nutrition fact
    df_nut = get_nutrition(lst_nut_fact,lst_info)

    return df_nut

if __name__ == '__main__':
    train()
    #predict()
    #evaluate()
