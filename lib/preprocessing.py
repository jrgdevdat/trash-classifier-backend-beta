import numpy as np
import cv2
from keras.utils import Sequence, to_categorical
import os
import math


class InputsPreprocessor:

    def get_inputs(self, features):

        # - Imagen del elemento

        # -- Agregar relleno blanco para completar resolución máxima
        element_image = features['Trash image']
        element_image = element_image[:646, :1280, :]
        prep_element_image = np.full((646, 1280, 3), 255, dtype=np.uint8)
        element_image_shape = element_image.shape
        prep_element_image[:element_image_shape[0], :element_image_shape[1], :] = element_image[:,:,:]

        # -- Redimensionar a la mitad (323, 640, 3) manteniendo relación de aspecto
        new_height = 323
        new_width = 640
        prep_element_image = cv2.resize(prep_element_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # -- Normalizar
        prep_element_image = prep_element_image.astype('float32')
        prep_element_image /= 255

        # Preparar arreglo con inputs
        input_element_image = np.zeros((1,) + prep_element_image.shape, dtype=np.float32)
        input_element_image[0] = prep_element_image

        inputs = input_element_image

        return inputs

    def get_inputs_shape(self):
        return (323, 640, 3)


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

    def __prepare_batch_outputs_list(self, actual_batch_size):
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
        
        # Preparar la lista con el batch de cada salida
        outputs = self.__prepare_batch_outputs_list(actual_batch_size)

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
            outputs[current_batch_element_index] = to_categorical(output_array, num_classes=num_of_property_classes)[0]

            # Aumentar índice del elemento del batch actual
            current_batch_element_index += 1

        # Retornar batch
        return inputs, outputs