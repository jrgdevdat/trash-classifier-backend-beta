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

# Realizar entrenamiento

Para entrenar un modelo se ejecuta el siguiente comando una vez configuradas las variables generales e hiperparámetros: 
```bash
uv run train_model.py
```