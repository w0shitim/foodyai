from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from foodyai.ml_logic.mod_predict import prediction_setup, prediction, get_class_to_category, get_detectron_config

app = FastAPI()
app.state.model = 'xx'

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict(img_path: str):
    pass
