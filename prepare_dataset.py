import random
from lib.datasets import ProjectDatasetsCreator, DatasetSampleBuilder, DatasetFileCreator
import os
import pandas as pd


# Variables
dataset_labels_file = 'dataset/original/dataset_labels.csv'
images_files_dir = 'dataset/original/images'
dataset_target_dir = 'dataset/preprocessed'
train_prop = 0.7
dev_prop = 0.15

# Semilla para garantizar reproducibilidad
random.seed(0)

# Cargar dataframe con la información de las etiquetas
df_dataset_labels = pd.read_csv(dataset_labels_file)

# Creación de los datasets (separación de los archivos)
project_datasets_creator = ProjectDatasetsCreator()

datasets = project_datasets_creator.create_datasets(
    images_files_dir, dataset_target_dir, train_prop, dev_prop)

# Preprocesar dataset
for dataset in datasets:
    
    dataset_length = len(dataset)
    
    for i in range(dataset_length):
        
        dataset_element = dataset[i]
        
        # Construir features y labels
        dataset_sample_builder = DatasetSampleBuilder(dataset_element)
        dataset_sample_builder.create_features()
        dataset_sample_builder.create_labels(df_dataset_labels)

        # Obtener resultado
        dataset_sample = dataset_sample_builder.get_sample()

        # Guardar resultado
        dataset_sample.save(dataset_element.output_file_path)

        # Imprimir datos del sample creado
        print(dataset_element)

# Crear archivo informativo del dataset creado
dataset_file_creator = DatasetFileCreator()
dataset_file_path = os.path.join(dataset_target_dir, 'dataset_info.txt')
dataset_file_creator.create_dataset_file(dataset_file_path, datasets)
