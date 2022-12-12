import pandas as pd


from foodyai.ml_logic.mod_predict import *
from foodyai.ml_logic.category import *


def train():
    pass

def evaluate():
    pass

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
