from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from foodyai.interface.main import predict

app = FastAPI()
app.state.model = './raw_data/model_path'

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict(image_path: str):
    '''
    pred is a dataframe containing nutrition fact for each food item
    '''
    pred = predict(image_path)

    return pred.to_dict()

@app.get("/")
def root():

    return {
    'greeting': 'Welcolme to the Foodyai API'
    }
