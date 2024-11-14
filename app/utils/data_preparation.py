import pandas as pd
# from sklearn.model_selection import train_test_split, KFold, cross_val_score
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib

# # Cargar y preparar los datos
# df = pd.read_csv("StudentsPerformanceAA3.csv")
# df['total score'] = df['math score'] + df['reading score'] + df['writing score']
# df['average'] = df['total score'] / 3

# # Seleccionar las características y la variable objetivo
# X = df[['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']]
# y = df['math score']  # Cambiar aquí para predecir una puntuación específica

# # Codificar variables categóricas
# label_encoders = {col: LabelEncoder().fit(X[col]) for col in X.columns}
# for col, encoder in label_encoders.items():
#     X[col] = encoder.transform(X[col])

# # Guardar los codificadores para la API
# for col, encoder in label_encoders.items():
#     joblib.dump(encoder, f"{col}_encoder.joblib")
    

class DataPreparationService:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.label_encoders = {}

    def load_and_prepare_data(self):
        # Cargar y preparar los datos
        self.df = pd.read_csv(self.file_path)
        self.df['total_score'] = self.df['math_score'] + self.df['reading_score'] + self.df['writing_score']
        self.df['average'] = self.df['total_score'] / 3

    def select_features_and_target(self):
        # Seleccionar las características y la variable objetivo
        X = self.df[['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']]
        y = self.df['math_score']  # Cambiar aquí para predecir una puntuación específica
        return X, y

    def encode_categorical_variables(self, X):
        # Codificar variables categóricas
        # self.label_encoders = {col: LabelEncoder().fit(X[col]) for col in X.columns}
        # for col, encoder in self.label_encoders.items():
        #     X[col] = encoder.transform(X[col])
        # return X
        # Codificar variables categóricas sin modificar el DataFrame original
        X_encoded = X.copy() #Para asegurar que no modifiquemos el DataFrame original X.
        self.label_encoders = {col: LabelEncoder().fit(X_encoded[col]) for col in X_encoded.columns}
        for col, encoder in self.label_encoders.items():
            X_encoded[col] = encoder.transform(X_encoded[col])
        return X_encoded

    def save_encoders(self):
        # Guardar los codificadores para la API
        for col, encoder in self.label_encoders.items():
            # Reemplazar caracteres no válidos en el nombre de la columna RACE/ETHNICITY
            safe_col = col.replace('/', '_').replace(' ', '_')
            joblib.dump(encoder, f"{safe_col}_encoder.joblib")

# Ejemplo de uso
if __name__ == "__main__":
    service = DataPreparationService("StudentsPerformanceAA3.csv")
    service.load_and_prepare_data()
    X, y = service.select_features_and_target()
    X_encoded = service.encode_categorical_variables(X)
    service.save_encoders()