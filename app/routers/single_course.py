from pathlib import Path
from typing import Any, List, Optional
from io import BytesIO
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import SQLModel
from app.models.Student import StudentModel

import pandas as pd
import joblib

MODEL_DIR = Path().resolve().joinpath("linear_regression_model.joblib") or Path().resolve().resolve().joinpath("linear_regression_model.joblib")

# Cargar el modelo entrenado
model = joblib.load(MODEL_DIR)

# Cargar los codificadores
label_encoders = {
    'gender': joblib.load("gender_encoder.joblib"),
    'race_ethnicity': joblib.load("race_ethnicity_encoder.joblib"),
    'parental_level_of_education': joblib.load("parental_level_of_education_encoder.joblib"),
    'lunch': joblib.load("lunch_encoder.joblib"),
    'test_preparation_course': joblib.load("test_preparation_course_encoder.joblib")
}

single_course_predict_router = APIRouter(prefix='/ml')

# Crear el endpoint de predicción
@single_course_predict_router.post("/predict")
def predict_math_score(data: StudentModel):
    # Convertir los datos de entrada en un DataFrame
    input_data = pd.DataFrame([data.model_dump()])
    
    # Reemplazar caracteres no válidos en los nombres de las columnas
    input_data.columns = [col.replace('/', '_').replace(' ', '_') for col in input_data.columns]
    
    # Codificar las variables categóricas
    for col, encoder in label_encoders.items():
        safe_col = col.replace('/', '_').replace(' ', '_')
        input_data[safe_col] = encoder.transform(input_data[safe_col])
    
    # Seleccionar las características adecuadas
    X_new = input_data[['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']]
    
    # Hacer la predicción
    prediction = model.predict(X_new)
    
    # Retornar el resultado de la predicción
    return {"predicted_math_score": prediction[0]}

# @router.post("/", response_model=Any)
# def create_item(
#     *,
#     db: Session = Depends(deps.get_db),
#     item_in: schemas.itemCreate,

# ) -> Any:

#     item = crud.item.create_with_owner(db=db, obj=item_in)
#     return item


# @router.get("/{id}", response_model=schemas.item)
# def read_item(
#     *,
#     db: Session = Depends(deps.get_db),
#     id: int,
#     current_user: models.User = Depends(deps.get_current_active_user),
# ) -> Any:
#     """
#     Get item by ID.
#     """
#     item = crud.item.get(db=db, id=id)
#     if not item:
#         raise HTTPException(status_code=404, detail="item not found")

#     return item


