
# Descripción del Proyecto

Este proyecto tiene como objetivo predecir las notas de los estudiantes utilizando técnicas de inteligencia artificial. El modelo está diseñado para analizar diversos factores que pueden influir en el rendimiento académico y proporcionar predicciones precisas que pueden ayudar a los educadores a identificar áreas de mejora y tomar decisiones informadas.
# Instrucciones para la instalación y ejecución

## Instalación de dependencias

Para instalar las dependencias necesarias, ejecute el siguiente comando:

```bash
pip install -r requirements.txt
```

## Activacion del entorno de python venv
En windows
```bash
venv/Scripts/activate 
```

## Ejecucion del entrenamiento y validación del modelo para generar el modelo joblib y los encoders de las variables categoricas
```bash
python -m app.ml_models.score_prediction_model
```

## Ejecución del proyecto

Para ejecutar el proyecto en el puerto 8000, utilice el siguiente comando:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Asegúrese de que `uvicorn` esté instalado y configurado correctamente en su entorno.
