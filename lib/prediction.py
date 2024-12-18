import cv2
from keras.applications.vgg16 import preprocess_input
import numpy as np
from lib.preprocessing import InputsPreprocessor, TrashClassifierDataSequence 


class TrashClassifierPredictor:

    def __init__(self, trash_classifier_model):
        self.trash_classifier_model = trash_classifier_model

    def __preprocess_image(self, trash_image):
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

    def __predict(self, trash_image):
        # Preprocess image
        preprocessed_trash_image = self.__preprocess_image(trash_image)
        # Prepare prediction input
        prediction_input = self.__create_input_from_preprocessed_image(preprocessed_trash_image)
        # Get prediction output
        prediction_output = self.trash_classifier_model.predict_on_input(prediction_input)
        trash_image_prediction = prediction_output[0]
        # Format prediction
        trash_image_formatted_prediction = self.__format_prediction(trash_image_prediction)
        return trash_image_formatted_prediction
    
    def predict_on_image_file(self, image_file_path):
        # Cargar imagen
        trash_image = cv2.imread(image_file_path, 1)
        return self.__predict(trash_image)

    def predict_on_api_loaded_image(self, loaded_image_contents):
        image_contents_np_array = np.fromstring(loaded_image_contents, np.uint8)
        prediction_image = cv2.imdecode(image_contents_np_array, cv2.IMREAD_COLOR)
        return self.__predict(prediction_image)

    def predict_on_dataset(self, dataset_dir_path, dataset_length):

        # Crear matriz de confusión (esperado/predicho)
        confusion_matrix = np.zeros((3, 3))

        # Crear secuencia para consumir los datos del dataset
        inputs_preprocessor = InputsPreprocessor()
        batch_size = 1
        data_sequence = TrashClassifierDataSequence(
            dataset_dir_path,
            dataset_length,
            inputs_preprocessor,
            batch_size,
        )

        for i in range(dataset_length):
            # Obtener datos de entrada y salida del elemento del dataset
            element_input, element_output = data_sequence[i]
            # Realizar predicción sobre el elemento
            prediction_output = self.trash_classifier_model.predict_on_input(element_input)
            # Guardar el resultado en la matriz de confusión
            expected_label_index = np.argmax(element_output[0])
            predicted_label_index = np.argmax(prediction_output[0])
            confusion_matrix[expected_label_index, predicted_label_index] += 1 

        return confusion_matrix
    
    def __del__(self):
        del self.trash_classifier_model
