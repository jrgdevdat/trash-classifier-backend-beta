import cv2
from keras.applications.vgg16 import preprocess_input
import numpy as np

class TrashClassifierPredictor:

    def __init__(self, trash_classifier_model):
        self.trash_classifier_model = trash_classifier_model

    def __preprocess_image(self, image_file_path):
        # Preprocess input image
        trash_image = cv2.imread(image_file_path, 1)
        # Convertir a RGB
        trash_image = cv2.cvtColor(trash_image, cv2.COLOR_BGR2RGB)
        # Redimensionar a (224, 224, 3)
        trash_image = cv2.resize(trash_image, (224, 224), interpolation=cv2.INTER_AREA)
        # Preprocesamiento VGG16
        trash_image = preprocess_input(trash_image)
        return trash_image

    def __create_input_from_preprocessed_image(self, preprocessed_image):
        # Preparar arreglo con inputs
        input = np.zeros((1,) + preprocessed_image.shape, dtype=np.float32)
        input[0] = preprocessed_image
        return input

    def __format_prediction(self, trash_image_prediction):
        trash_type_index_label = np.argmax(trash_image_prediction)
        if trash_type_index_label == 0:
            return 'Material no reciclable (sin manejo especializado)'
        elif trash_type_index_label == 1:
            return 'Material no reciclable (requiere manejo especializado)'
        else:
            return 'Material reciclable'

    def predict(self, image_file_path):
        # Preprocess image
        preprocessed_trash_image = self.__preprocess_image(image_file_path)
        # Prepare prediction input
        prediction_input = self.__create_input_from_preprocessed_image(preprocessed_trash_image)
        # Get prediction output
        prediction_output = self.trash_classifier_model.predict_on_input(prediction_input)
        trash_image_prediction = prediction_output[0]
        # Format prediction
        trash_image_formatted_prediction = self.__format_prediction(trash_image_prediction)
        return trash_image_formatted_prediction
