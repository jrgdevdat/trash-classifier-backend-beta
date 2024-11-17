
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
