import numpy as np
import cv2
from keras.utils import Sequence, to_categorical
import os
import math
from keras.applications.vgg16 import preprocess_input


class InputsPreprocessor:

    def get_inputs(self, features):

        # - Imagen del elemento
        trash_image = features['Trash image']

        # -- Redimensionar a (224, 224, 3)
        new_height = 224
        new_width = 224
        trash_image = cv2.resize(trash_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # -- Preprocesamiento adicional necesario para VGG16
        trash_image = preprocess_input(trash_image)

        # Preparar arreglo con inputs
        input_image = np.zeros((1,) + trash_image.shape, dtype=np.float32)
        input_image[0] = trash_image

        inputs = input_image

        return inputs

    def get_inputs_shape(self):
        return (224, 224, 3)


class TrashClassifierDataSequence(Sequence):

    def __init__(self, samples_dir_path, number_of_samples, inputs_preprocessor, batch_size):
        self.samples_dir_path = samples_dir_path
        self.number_of_samples = number_of_samples
        self.inputs_preprocessor = inputs_preprocessor
        self.batch_size = batch_size
    
    def __read_sample(self, sample_file_path):
        with np.load(sample_file_path, allow_pickle=True) as sample_file:
            features, labels = sample_file['features'], sample_file['labels']
            return features.item(), labels.item()

    def __len__(self):
        return math.ceil(self.number_of_samples / float(self.batch_size))

    def __prepare_batch_output(self, actual_batch_size):
        num_of_property_classes = 3
        output = np.zeros((actual_batch_size, num_of_property_classes), dtype=np.float32)
        return output

    def __getitem__(self, idx):

        # Definir índices del batch
        initial_batch_npz_index = idx * self.batch_size
        final_batch_npz_index = (idx + 1) * self.batch_size
        # - Asegurarse que el índice final no pase el límite
        final_batch_npz_index = min(final_batch_npz_index, self.number_of_samples)

        # Definir variables donde guardar las entradas y salidas
        actual_batch_size = final_batch_npz_index - initial_batch_npz_index        
        # - Preparar arreglo con el batch de entradas
        inputs_element_shape = self.inputs_preprocessor.get_inputs_shape()
        inputs = np.zeros((actual_batch_size,) + inputs_element_shape, dtype=np.float32)
        
        # Preparar el batch de salida
        output = self.__prepare_batch_output(actual_batch_size)

        # Iterar por cada archivo npz del batch
        current_batch_element_index = 0
        for current_npz_index in range(initial_batch_npz_index, final_batch_npz_index):

            # Leer archivo .npz
            sample_file = 'sample{}.npz'.format(current_npz_index)
            sample_file_path = os.path.join(self.samples_dir_path, sample_file)
            features, labels = self.__read_sample(sample_file_path)

            # Obtener inputs
            if self.batch_size > 1:
                inputs[current_batch_element_index] = self.inputs_preprocessor.get_inputs(features)[0]
            else:
                inputs = self.inputs_preprocessor.get_inputs(features)
            
            # Crear arreglo donde guardar la etiqueta de la clase a predecir
            num_of_property_classes = 3
            output_array = np.zeros((1,1), dtype=np.uint16)
            output_array[0, 0] = labels['trash-type']
            output[current_batch_element_index] = to_categorical(output_array, num_classes=num_of_property_classes)[0]

            # Aumentar índice del elemento del batch actual
            current_batch_element_index += 1

        # Retornar batch
        return inputs, output