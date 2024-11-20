from .models import TrashClassifierInnerModelArchitect, TrashClassifierNeuralNetwork
from .hyperparameters import HyperparametersConfiguration
from .preprocessing import InputsPreprocessor, TrashClassifierDataSequence
import os
from keras import backend as K


class TuningHyperparametersConditions:

    def __init__(
            self, min_learning_rate_exponent, max_learning_rate_exponent, min_hidden_units, max_hidden_units, min_decoder_layers, max_decoder_layers, encoder_trainable):
        self.min_learning_rate_exponent = min_learning_rate_exponent
        self.max_learning_rate_exponent = max_learning_rate_exponent
        self.min_hidden_units = min_hidden_units
        self.max_hidden_units = max_hidden_units
        self.min_decoder_layers = min_decoder_layers
        self.max_decoder_layers = max_decoder_layers
        self.encoder_trainable = encoder_trainable


class TrashClassifierNeuralNetworkTuner:

    def __init__(self, dataset_dir_path, model_dir_path, dataset_info):
        self.dataset_dir_path = dataset_dir_path
        self.model_dir_path = model_dir_path
        self.dataset_info = dataset_info

    def perform_tuning(self, number_of_models, epochs, batch_size, tuning_hyperparameters_conditions):

        for model_number in range(number_of_models):

            # Crear configuración de hiperparámetros aleatoria
            hyperparameters_config = HyperparametersConfiguration.fromRandomness(
                tuning_hyperparameters_conditions)

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
            train_dir_path = os.path.join(self.dataset_dir_path, 'train')
            training_sequence = TrashClassifierDataSequence(
                train_dir_path,
                self.dataset_info.num_train_samples,
                training_inputs_preprocessor,
                batch_size,
            )

            validation_inputs_preprocessor = InputsPreprocessor()
            dev_dir_path = os.path.join(self.dataset_dir_path, 'dev')
            validation_sequence = TrashClassifierDataSequence(
                dev_dir_path,
                self.dataset_info.num_dev_samples,
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
            test_dir_path = os.path.join(self.dataset_dir_path, 'test')
            evaluation_sequence = TrashClassifierDataSequence(
                test_dir_path,
                self.dataset_info.num_test_samples,
                evaluation_inputs_preprocessor,
                batch_size,
            )

            trash_classifier.evaluate_sequence(evaluation_sequence)

            # Guardar red neuronal
            model_name = 'model-trained-{}'.format(model_number)
            trash_classifier.save_to_h5(self.model_dir_path, model_name)

            # Eliminar red neuronal
            K.clear_session()
            del trash_classifier