# Dividir el dataset en conjunto de entrenamiento y prueba
from pathlib import Path
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from app.utils.data_preparation import DataPreparationService

class ScorePredictionModelService:
    def __init__(self, data_file):
        self.data_service = DataPreparationService(data_file)
        self.model = LinearRegression()
        self.k = 5
        self.kf = KFold(n_splits=self.k, shuffle=True, random_state=42)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def prepare_data(self):
        self.data_service.load_and_prepare_data()
        X, y = self.data_service.select_features_and_target()
        X_encoded = self.data_service.encode_categorical_variables(X)
        self.data_service.save_encoders()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42
        )

    #Aquí es donde entrenamos el modelo usando self.model.fit(self.X_train, self.y_train).
    def train_model(self):
        # Entrenar el modelo con el conjunto de entrenamiento
        self.model.fit(self.X_train, self.y_train)

    #Después de entrenar el modelo, usamos el conjunto de prueba para hacer predicciones y calcular métricas como el MSE (Error Cuadrático Medio) y el R².
    def evaluate_model(self):
        # Evaluar el modelo usando el conjunto de prueba
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        return mse, r2

    #Realiza la validación cruzada para evaluar el modelo usando diferentes particiones del conjunto de datos.
    def cross_validate_model(self):
        # Validación cruzada
        mse_scores = cross_val_score(
            self.model, self.X_train, self.y_train, cv=self.kf, scoring='neg_mean_squared_error'
        )
        r2_scores = cross_val_score(
            self.model, self.X_train, self.y_train, cv=self.kf, scoring='r2'
        )
        return mse_scores, r2_scores

    def save_model(self, filename="linear_regression_model.joblib"):
        # Guardar el modelo entrenado
        joblib.dump(self.model, filename)

DATA_DIR = Path().resolve().joinpath("app/data") or Path().resolve().resolve().joinpath("app/data")

# Ejemplo de uso
if __name__ == "__main__":
    model_service = ScorePredictionModelService(f"{DATA_DIR}/StudentsPerformanceAA3.csv")
    model_service.prepare_data()
    model_service.train_model()
    mse, r2 = model_service.evaluate_model()
    print(f"MSE en el conjunto de prueba: {mse}")
    print(f"R2 en el conjunto de prueba: {r2}")

    # Validación cruzada
    mse_scores, r2_scores = model_service.cross_validate_model()
    print("MSE Scores en validación cruzada:", mse_scores)
    print("R2 Scores en validación cruzada:", r2_scores)

    # Guardar el modelo entrenado
    model_service.save_model()
    
#python score_prediction_model.py

# service = DataPreparationService("StudentsPerformanceAA3.csv")
# service.load_and_prepare_data()
# X, y = service.select_features_and_target()
# X_encoded = service.encode_categorical_variables(X)
# service.save_encoders()

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Inicializar el modelo de regresión lineal
# model = LinearRegression()

# # Configurar K-Fold Cross-Validation
# k = 5
# kf = KFold(n_splits=k, shuffle=True, random_state=42)

# # Evaluación usando Cross-Validation
# mse_scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
# r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

# # Imprimir los resultados de Cross-Validation
# print("Mean Squared Error (MSE) para cada fold:", -mse_scores)
# print("MSE Promedio:", -mse_scores.mean())
# print("R2 Score para cada fold:", r2_scores)
# print("R2 Promedio:", r2_scores.mean())

# # Entrenar el modelo en el conjunto de entrenamiento completo
# model.fit(X_train, y_train)

# # Evaluar el modelo en el conjunto de prueba
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(f"Mean Squared Error en Test: {mse}")
# print(f"R2 Score en Test: {r2}")

# # Guardar el modelo entrenado
# joblib.dump(model, "linear_regression_model_math_score.joblib")