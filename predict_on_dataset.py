import os
from lib.datasets import DatasetFileReader
from lib.models import TrashClassifierNeuralNetwork
from lib.prediction import TrashClassifierPredictor


# Varibles importantes
dataset_dir_path = 'dataset/preprocessed'
model_file_path = 'model/model-trained.h5'

# Cargar modelo
trash_classifier_neural_network = TrashClassifierNeuralNetwork.fromFile(model_file_path)
trash_classifier_predictor = TrashClassifierPredictor(trash_classifier_neural_network)

# Leer información del dataset
dataset_file_reader = DatasetFileReader()
dataset_file_path = os.path.join(dataset_dir_path, 'dataset_info.txt')
dataset_info = dataset_file_reader.read_dataset_file(dataset_file_path)
print(dataset_info)

# Dataset de entrenamiento
train_dir_path = os.path.join(dataset_dir_path, 'train')
train_dataset_length = dataset_info.num_train_samples
train_dataset_confusion_matrix = trash_classifier_predictor.predict_on_dataset(
    train_dir_path, train_dataset_length)
print('\nConfusion matrix for training set')
print(train_dataset_confusion_matrix)

# Dataset de validación
dev_dir_path = os.path.join(dataset_dir_path, 'dev')
dev_dataset_length = dataset_info.num_dev_samples
dev_dataset_confusion_matrix = trash_classifier_predictor.predict_on_dataset(
    dev_dir_path, dev_dataset_length)
print('\nConfusion matrix for dev set')
print(dev_dataset_confusion_matrix)

# Dataset de prueba
test_dir_path = os.path.join(dataset_dir_path, 'test')
test_dataset_length = dataset_info.num_test_samples
test_dataset_confusion_matrix = trash_classifier_predictor.predict_on_dataset(
    test_dir_path, test_dataset_length)
print('\nConfusion matrix for test set')
print(test_dataset_confusion_matrix)
