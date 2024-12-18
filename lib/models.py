from datetime import datetime
from keras.optimizers import Adam
from keras.layers import Input, Dense
from keras.initializers import RandomNormal
from keras.models import Model, load_model
from keras.applications.vgg16 import VGG16
import math
import os


class TrashClassifierInnerModelArchitect:

    def create_inner_model(self, hyperparameters_config):

        image_shape = (224, 224, 3)        
        hidden_units = hyperparameters_config.hidden_units # Número de unidades en las capas ocultas        

        # Definición de inputs
        image_input = Input(shape=image_shape, name='Trash_image')

        # Creación del clasificador seleccionado
        vgg_16 = VGG16(include_top=False, weights='imagenet', input_shape=image_shape, pooling='max')
        encoder_out = vgg_16(image_input)

        # Creación del decodificador
        number_of_decoder_layers = hyperparameters_config.decoder_layers 
        for layer_index in range(number_of_decoder_layers):
            if layer_index == 0:
                hidden_out = Dense(hidden_units, activation='tanh',
                                   kernel_initializer=RandomNormal(stddev=math.sqrt(1 / 512.0)))(encoder_out)
            else:
                hidden_out = Dense(hidden_units, activation='tanh',
                                   kernel_initializer=RandomNormal(stddev=math.sqrt(1 / hidden_units)))(hidden_out)

        # Agregando capa de salida
        output_layer_name = 'trash_type'
        # Hay 3 etiquetas posibles
        property_classes = 3
        prop_out = Dense(property_classes, activation='softmax', name=output_layer_name)(hidden_out)

        model = Model(inputs=image_input, outputs=prop_out)

        # Congelar pesos del codificador si fue especificado
        if not hyperparameters_config.encoder_trainable:
            vgg_16.trainable = False

        return model


class TrashClassifierNeuralNetwork:

    def __init__(self, inner_model, hyperparameters_configuration, creation_date=None):
        self.inner_model = inner_model
        if creation_date is None:
            self.creation_date = '_'.join(str(datetime.now()).split('.')[0].split(' '))
        else:
            self.creation_date = creation_date
        if hyperparameters_configuration is not None:
            self.hyperparameters_configuration = hyperparameters_configuration
            self.model_id = 'date,{};lr,{};hu,{};dl,{};et,{}'.format(
                self.creation_date,
                self.hyperparameters_configuration.learning_rate,
                self.hyperparameters_configuration.hidden_units,
                self.hyperparameters_configuration.decoder_layers,
                self.hyperparameters_configuration.encoder_trainable,
        )

    def print_summary(self):
        self.inner_model.summary()

    def compile(self):
        # Preparar diccionarios con la información de la función de error y métricas a aplicar
        loss = {}
        metrics = {}
        output_layer_name = 'trash_type'
        loss[output_layer_name] = 'categorical_crossentropy'
        metrics[output_layer_name] = ['categorical_accuracy']
        optimizer = Adam(learning_rate=self.hyperparameters_configuration.learning_rate, epsilon=10 ** -8)
        self.inner_model.compile(optimizer, loss=loss, metrics=metrics)
    
    def __del__(self):
        del self.inner_model

    def __str__(self):
        header = '******** Trash Classifier Neural Network Info\n'
        model_id_line = 'Model id: {}\n'.format(self.model_id)
        creation_date_line = 'Creation date: {}\n'.format(self.creation_date)
        return '{}{}{}{}'.format(
            header, model_id_line, creation_date_line, self.hyperparameters_configuration)
    
    def fit_sequence(self, training_sequence, validation_sequence, epochs, initial_epoch=0):
        # Entrenar modelo interno
        print('**** Training about to start\tinitial epoch: {}\tfinal_epoch: {}'.format(
            initial_epoch, epochs))
        return self.inner_model.fit(
            x=training_sequence, epochs=epochs, verbose=2, validation_data=validation_sequence, initial_epoch=initial_epoch)
    
    def evaluate_sequence(self, evaluation_sequence):
        print('**** Evaluation about to start')
        scores = self.inner_model.evaluate(x=evaluation_sequence, verbose=2)
        # Imprimir cada valor de loss y métricas
        for score_name, score_value in zip(self.inner_model.metrics_names, scores):
            print(score_name, score_value)

    def save_to_h5(self, target_dir_path, model_name=None):
        model_file_name = 'model-trained' if model_name is None else model_name
        h5_file_path = '{}.h5'.format(os.path.join(target_dir_path, model_file_name))
        self.inner_model.save(h5_file_path)

    @classmethod
    def fromFile(cls, model_h5_file_path):
        # Cargar modelo
        inner_model = load_model(model_h5_file_path)
        # Crear red neuronal
        return cls(inner_model, None)
    
    def predict_on_input(self, prediction_input):
        return self.inner_model.predict(prediction_input, batch_size=1, verbose=2)
