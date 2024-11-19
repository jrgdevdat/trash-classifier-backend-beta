from lib.models import TrashClassifierNeuralNetwork
from lib.prediction import TrashClassifierPredictor

# Varibles importantes
image_file_path = 'test.JPG'
model_file_path = 'model/model-trained.h5'

# Load model
trash_classifier_neural_network = TrashClassifierNeuralNetwork.fromFile(model_file_path)

# Predict in image
trash_classifier_predictor = TrashClassifierPredictor(trash_classifier_neural_network)
prediction = trash_classifier_predictor.predict(image_file_path)

# Print prediction
print(prediction)
