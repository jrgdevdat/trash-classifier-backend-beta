import os
from lib.datasets import DatasetFileReader
from lib.hyperparameters import HyperparametersConfiguration
from lib.preprocessing import InputsPreprocessor, TrashClassifierDataSequence
from lib.models import TrashClassifierInnerModelArchitect, TrashClassifierNeuralNetwork


# Varibles importantes
dataset_dir_path = 'dataset/preprocessed'
model_dir_path = 'model'
epochs = 1
batch_size = 1

# Leer información del dataset
dataset_file_reader = DatasetFileReader()
dataset_file_path = os.path.join(dataset_dir_path, 'dataset_info.txt')
dataset_info = dataset_file_reader.read_dataset_file(dataset_file_path)
print(dataset_info)

# Crear configuración de hiperparámetros
learning_rate = 0.10
hidden_units = 2
decoder_layers = 1
encoder_trainable = False
hyperparameters_config = HyperparametersConfiguration(learning_rate, hidden_units, decoder_layers, encoder_trainable)

# Crear modelo interno de la red neuronal
trash_classifier_inner_model_architect = TrashClassifierInnerModelArchitect()
trash_classifier_inner_model = trash_classifier_inner_model_architect.create_inner_model(hyperparameters_config)

# Crear red neuronal
trash_classifier = TrashClassifierNeuralNetwork(trash_classifier_inner_model, hyperparameters_config)
print(trash_classifier)
trash_classifier.print_summary()
trash_classifier.compile()

# Entrenar red neuronal
training_inputs_preprocessor = InputsPreprocessor()
train_dir_path = os.path.join(dataset_dir_path, 'train')
training_sequence = TrashClassifierDataSequence(
    train_dir_path,
    dataset_info.num_train_samples,
    training_inputs_preprocessor,
    batch_size,
)

validation_inputs_preprocessor = InputsPreprocessor()
dev_dir_path = os.path.join(dataset_dir_path, 'dev')
validation_sequence = TrashClassifierDataSequence(
    dev_dir_path,
    dataset_info.num_dev_samples,
    validation_inputs_preprocessor,
    batch_size,
)

trash_classifier.fit_sequence(
    training_sequence,
    validation_sequence,
    epochs,
)

# Evaluar red neuronal
evaluation_inputs_preprocessor = InputsPreprocessor()
test_dir_path = os.path.join(dataset_dir_path, 'test')
evaluation_sequence = TrashClassifierDataSequence(
    test_dir_path,
    dataset_info.num_test_samples,
    evaluation_inputs_preprocessor,
    batch_size,
)

trash_classifier.evaluate_sequence(evaluation_sequence)

# Guardar red neuronal
trash_classifier.save_to_h5(model_dir_path)

# Eliminar red neuronal
del trash_classifier