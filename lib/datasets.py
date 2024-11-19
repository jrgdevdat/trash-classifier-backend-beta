import os
import random
import numpy as np
import cv2


class DatasetFileReader:

    def read_dataset_file(self, dataset_file_path):
        with open(dataset_file_path) as dataset_file:
            # Leer número de ejemplos para entrenamiento en el dataset
            num_train_samples = int(dataset_file.readline()[:-1])
            # Leer número de ejemplos para validación en el dataset
            num_dev_samples = int(dataset_file.readline()[:-1])
            # Leer número de ejemplos para prueba en el dataset
            num_test_samples = int(dataset_file.readline()[:-1])
            return DatasetInfo(num_train_samples, num_dev_samples, num_test_samples)


class DatasetInfo:

    def __init__(self, num_train_samples, num_dev_samples, num_test_samples):
        self.num_train_samples = num_train_samples
        self.num_dev_samples = num_dev_samples
        self.num_test_samples = num_test_samples

    def __str__(self):        
        header = '**** Info about this dataset:\n'
        train_samples_line = 'Number of train samples: {}\n'.format(self.num_train_samples)
        dev_samples_line = 'Number of validation samples: {}\n'.format(self.num_dev_samples)
        test_samples_line = 'Number of test samples: {}\n'.format(self.num_test_samples)
        return '{}{}{}{}'.format(header, train_samples_line, dev_samples_line, test_samples_line)


class DatasetElement:

    def __init__(self, image_file_path, output_file_path):
        self.image_file_path = image_file_path
        self.output_file_path = output_file_path

    def __str__(self):
        str_repr = '({}, {})'.format(self.image_file_path, self.output_file_path)
        return str_repr


class Dataset:

    def __init__(self, images_dir, target_dir, images_files):
        self.images_dir = images_dir
        self.target_dir = target_dir
        self.images_files = images_files

    def __str__(self):
        str_repr = 'Dataset with target dir {} and {} elements with images source dir {}'.format(
                    self.target_dir, self.__len__(), self.images_dir)
        return str_repr

    def __len__(self):
        return len(self.images_files)

    def __getitem__(self, index):
        try:
            image_file = self.images_files[index]
        except IndexError:
            raise IndexError("Dataset with target dir: {} and {} elements was called with an index value of {}".format(
                             self.target_dir, self.__len__(), index))
        else:
            image_file_path = os.path.join(self.images_dir, image_file)
            output_file = 'sample{}'.format(index)
            output_file_path = os.path.join(self.target_dir, output_file)
            return DatasetElement(image_file_path, output_file_path)


class ProjectDatasetsCreator:

    def create_datasets(self, images_dir, target_dir, train_prop, dev_prop):
        # Obtener los nombres de las imágenes y el número
        images_files = os.listdir(images_dir)
        images_files_number = len(images_files)
        # Cambiar el orden de los archivos de imágenes aleatoriamente
        images_files = random.sample(images_files, images_files_number)
        
        # Separar archivos en archivos de entrenamiento, validación y prueba
        train_sep_index = int(images_files_number * train_prop)
        val_sep_index = train_sep_index + int(images_files_number * dev_prop)
        images_train_files = images_files[:train_sep_index]
        images_val_files = images_files[train_sep_index: val_sep_index]
        images_test_files = images_files[val_sep_index:]

        # Crear datasets
        train_dir = os.path.join(target_dir, 'train')
        train_dataset = self.create_dataset(images_dir, train_dir, images_train_files)
        val_dir = os.path.join(target_dir, 'dev')
        val_dataset = self.create_dataset(images_dir, val_dir, images_val_files)
        test_dir = os.path.join(target_dir, 'test')
        test_dataset = self.create_dataset(images_dir, test_dir, images_test_files)
        return [train_dataset, val_dataset, test_dataset]

    def create_dataset(self, images_dir, target_dir, images_files):
        # Crear directorio objetivo, si este no existe
        try:
            os.mkdir(target_dir)
        except FileExistsError:
            pass
        return Dataset(images_dir, target_dir, images_files)


class DatasetSample:

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def save(self, output_file_path):
        np.savez_compressed(output_file_path, features=self.features, labels=self.labels)


class DatasetSampleBuilder:

    def __init__(self, dataset_element):        
        self.dataset_element = dataset_element
        self.dataset_sample = DatasetSample(None, None)

    def create_features(self):
        features = {}
        # Cargar imagen
        image_file_path = self.dataset_element.image_file_path
        trash_image = cv2.imread(image_file_path, 1)
        # Convertir a RGB
        trash_image = cv2.cvtColor(trash_image, cv2.COLOR_BGR2RGB)
        # Redimensionar a (224, 224, 3)
        features['Trash image'] = cv2.resize(trash_image, (224, 224), interpolation=cv2.INTER_AREA)
        # Guardar features
        self.dataset_sample.features = features

    def create_labels(self, df_dataset_labels):
        labels = {}
        # Buscar en el dataframe utilizando el nombre del archivo
        image_file_path = self.dataset_element.image_file_path
        image_file_name = os.path.basename(image_file_path)
        df_dataset_element_label = df_dataset_labels[df_dataset_labels['photo'] == image_file_name]
        dataset_element_label = df_dataset_element_label['label'].iloc[0]
        labels['trash-type'] = dataset_element_label
        # Guardar labels
        self.dataset_sample.labels = labels

    def get_sample(self):
        return self.dataset_sample


class DatasetFileCreator:

    def create_dataset_file(self, dataset_file_path, datasets):
        with open(dataset_file_path, 'w') as dataset_file:
            for dataset in datasets:
                dataset_file.write('{}\n'.format(len(dataset)))
