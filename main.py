from fastapi import FastAPI, UploadFile
from contextlib import asynccontextmanager
from lib.models import TrashClassifierNeuralNetwork
from lib.prediction import TrashClassifierPredictor
from fastapi.middleware.cors import CORSMiddleware
from dotenv import dotenv_values


config = dotenv_values(".env")


prediction_model_file_path = config['PREDICTION_MODEL_FILE_PATH']
prediction_models = {}

@asynccontextmanager
async def trash_classifier_lifespan(app: FastAPI):
    # Cargar modelo desde el archivo
    trash_classifier_neural_network = TrashClassifierNeuralNetwork.fromFile(
        prediction_model_file_path)
    prediction_models['trash_classifier_predictor'] = TrashClassifierPredictor(trash_classifier_neural_network)
    yield
    # Eliminar modelo de memoria
    del prediction_models['trash_classifier_predictor']


app = FastAPI(lifespan=trash_classifier_lifespan)


origins = [
    config['FRONTEND_ORIGIN'],
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=['POST'],
    allow_headers=["*"],
)


@app.post("/trash-classifier-predictions", status_code=201)
async def classify_trash_from_image(image_file: UploadFile):
    # Obtener contenido de la imagen
    image_file_contents = await image_file.read()
    # Realizar predicción
    trash_classifier_predictor = prediction_models['trash_classifier_predictor']
    predicted_trash_type = trash_classifier_predictor.predict_on_api_loaded_image(
        image_file_contents)
    return {"type": predicted_trash_type}