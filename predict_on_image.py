from lib.models import TrashClassifierNeuralNetwork
from lib.prediction import TrashClassifierPredictor

# Varibles importantes
image_file_path = 'test.JPG'
model_file_path = 'model/model-trained.h5'

# Cargar modelo
trash_classifier_neural_network = TrashClassifierNeuralNetwork.fromFile(model_file_path)

# Predecir sobre la imagen
trash_classifier_predictor = TrashClassifierPredictor(trash_classifier_neural_network)
prediction = trash_classifier_predictor.predict(image_file_path)

# Imprimir etiqueta predicha
print(prediction)
