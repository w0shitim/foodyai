from pycocotools.coco import COCO
import pandas as pd
import requests

import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer as wnl

TRAIN_ANNOTATIONS_PATH = 'Training_2/annotations.json'
train_coco = COCO(TRAIN_ANNOTATIONS_PATH)

def get_category():
    """
    from the coco file containing all the annotations
    create a dataframe having the category id and the category name
    """
    category_ids = sorted(train_coco.getCatIds())
    categories = train_coco.loadCats(category_ids)
    df = pd.DataFrame(categories)

    df_cat = df.rename(columns = {'id':'category_id'})

    return df_cat

def preprocessing(categories:list)->str:
    '''
    give a list of string --> each string is the category predicted
    clean each sentence in the column reviews for our dataframe "data"
    remove whitespaces, lowercase characers, remove numbers, remove punctuation ,tokenize, lemmatize
    '''
    cat_cleaned = []

    for sentence in categories:

        sentence = sentence.strip() #remove whitespaces
        sentence = sentence.lower() # lower chara
        sentence = ''.join(char for char in sentence if not char.isdigit()) # remove numbers

        for punctuation in string.punctuation:
            sentence = sentence.replace(punctuation, '') # remove punctuation

        tokenized_sentence = word_tokenize(sentence) # tokenize sentence
        stop_words = set(stopwords.words('english')) # define stopwords

        tokenized_sentence_cleaned = [w for w in tokenized_sentence if not w in stop_words] # remove stopwords

        verb_lemmatized = [wnl().lemmatize(word, pos = "v") for word in tokenized_sentence_cleaned] # lemmatize for verbs
        lemmatized = [wnl().lemmatize(word, pos = "n") for word in verb_lemmatized] # lemmatize for nouns on top of lemmatize for verbs

        sentence = ' '.join(lemmatized)

        cat_cleaned.append(sentence)

    cat_cleaned = ' '.join(cat_cleaned)

    return cat_cleaned


def detect_food(text:str,API_KEY:str,BASE_URL:str,params:dict):
    """
    detect food item in a sentence
    input:
    - sentence with all categories
    """

    endpoint = "food/detect"
    url_query = {"text":text}

    response_item = requests.post(BASE_URL+endpoint, data=url_query, params=params)
    item_json = response_item.json()

    item_lst = []

    for i in range(len(item_json['annotations'])):
        item_lst.append(item_json['annotations'][i]['annotation'])

    return item_lst

def get_food_info(item_lst:list,BASE_URL:str):
    """

    """
    lst_info = []
    endpoint = "recipes/parseIngredients"

    for i in range(len(item_lst)):
        ingredientList=item_lst[i]
        servings=1
        includeNutrition=True
        url_query = {"ingredientList": ingredientList, "servings": servings,"includeNutrition": includeNutrition}
        response_info = requests.post(BASE_URL+endpoint, data=url_query, params=params)
        lst_info.append(response_info.json()[0])

def get_nutrition(nut_key:list,nut_info:list)->pd.core.frame.DataFrame:

    """
    The input is a list of string containing nutritional fact wanted (nut_key).
    With the list of dictionary containing food information from the API call (nut_info).

    The output is a dataframe containing nutrition fact for a food category.
    """

    df_nut = pd.DataFrame()
    lst_nut = []
    dicts = {}
    list_nut_fact = [name.lower() for name in nut_key]

    for i in range(len(nut_info)):

        dicts['name'] = nut_info[i]['name']
        dicts['amount'] = nut_info[i]['nutrition']['weightPerServing']['amount']
        dicts['unit'] = nut_info[i]['nutrition']['weightPerServing']['unit']

        nut = nut_info[i]['nutrition']['nutrients']
        for j in range(len(nut)):
            if nut_info[i]['nutrition']['nutrients'][j]['name'].lower() in list_nut_fact:
                dicts[nut_info[i]['nutrition']['nutrients'][j]['name'].lower()+'_amount'] = nut_info[i]['nutrition']['nutrients'][j]['amount']
                dicts[nut_info[i]['nutrition']['nutrients'][j]['name'].lower()+'unit'] = nut_info[i]['nutrition']['nutrients'][j]['unit']

        lst_nut.append(dicts)
        dicts={}


    df_nut = pd.DataFrame(lst_nut)

    return df_nut
