# Creación y configuración del ambiente virtual

## Requerimientos
El ambiente fue creado con [uv](https://github.com/astral-sh/uv)

## Crear ambiente con uv
Ejecutar el siguiente comando en el directorio raíz del proyecto:
```bash
uv venv --python 3.9.20
```

## Instalar dependencias
Una vez activado el ambiente virtual, ejecutar el siguiente comando en la raíz del proyecto:
```bash
uv pip sync requirements.txt
```

# Preparar dataset para entrenamiento

Para transformar las imágenes originales en los archivos que se consumen al entrenar el modelo se debe ejecutar el siguiente script:
```bash
uv run prepare_dataset.py
```

# Realizar entrenamiento

Para entrenar un modelo se ejecuta el siguiente comando una vez configuradas las variables generales e hiperparámetros: 
```bash
uv run train_model.py
```

# Ajuste de hiperparámetros
```bash
uv run tune_model.py
```

# Inferencia sobre imagen
```bash
uv run predict_on_image.py
```

# Inferencia sobre dataset (cálculo de la matriz de confusión)
```bash
uv run predict_on_dataset.py
```

# Iniciar backend
```bash
fastapi dev main.py
```
**Nota:** Esperar a que salga el mensaje *Application startup complete* con el fin
de tener el modelo correctamente cargado.