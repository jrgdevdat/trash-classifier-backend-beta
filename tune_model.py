import os
from lib.datasets import DatasetFileReader
from lib.tuning import TuningHyperparametersConditions, TrashClassifierNeuralNetworkTuner


# Varibles importantes
dataset_dir_path = 'dataset/preprocessed'
model_dir_path = 'models_tuned'
number_of_models = 2
epochs = 1
batch_size = 1

# Leer información del dataset
dataset_file_reader = DatasetFileReader()
dataset_file_path = os.path.join(dataset_dir_path, 'dataset_info.txt')
dataset_info = dataset_file_reader.read_dataset_file(dataset_file_path)
print(dataset_info)

# Crear condiciones de hiperparámetros para el ajuste
min_learning_rate_exponent = -6
max_learning_rate_exponent = 1
min_hidden_units = 1
max_hidden_units = 1
min_decoder_layers = 1
max_decoder_layers = 1
encoder_trainable = False

tuning_hyperparameters_conditions = TuningHyperparametersConditions(
    min_learning_rate_exponent,
    max_learning_rate_exponent,
    min_hidden_units,
    max_hidden_units,
    min_decoder_layers,
    max_decoder_layers,
    encoder_trainable,
)

# Crear tuner
trash_classifier_neural_network_tuner = TrashClassifierNeuralNetworkTuner(
    dataset_dir_path, model_dir_path, dataset_info)

# Realizar ajuste
trash_classifier_neural_network_tuner.perform_tuning(
    number_of_models, epochs, batch_size, tuning_hyperparameters_conditions
)