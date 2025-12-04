#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#

import gzip
import json
import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def load_data():
    """Load and unzip the train and test datasets."""
    train_data = pd.read_csv(
        "files/input/train_data.csv.zip", compression="zip", index_col=0
    )
    test_data = pd.read_csv(
        "files/input/test_data.csv.zip", compression="zip", index_col=0
    )
    return train_data, test_data


def preprocess_data(train_data, test_data):
    """
    Paso 1: Preprocesar los datos.
    - Crear la columna 'Age' a partir de la columna 'Year' (asumiendo año actual = 2021)
    - Eliminar las columnas 'Year' y 'Car_Name'
    """
    # Create Age column
    train_data["Age"] = 2021 - train_data["Year"]
    test_data["Age"] = 2021 - test_data["Year"]

    # Drop Year column (Car_Name is already the index, so we just reset it to remove it)
    train_data = train_data.drop(columns=["Year"]).reset_index(drop=True)
    test_data = test_data.drop(columns=["Year"]).reset_index(drop=True)

    return train_data, test_data


def split_data(train_data, test_data):
    """
    Paso 2: Dividir los datasets en x_train, y_train, x_test, y_test.
    """
    # Split features and target variable
    x_train = train_data.drop(columns=["Present_Price"])
    y_train = train_data["Present_Price"]

    x_test = test_data.drop(columns=["Present_Price"])
    y_test = test_data["Present_Price"]

    return x_train, y_train, x_test, y_test


def create_pipeline(x_train):
    """
    Paso 3: Crear un pipeline con:
    - One-hot encoding para variables categóricas
    - Escalado de variables numéricas a [0, 1]
    - Selección de K mejores características
    - Modelo de regresión lineal
    """
    # Identify categorical and numerical columns
    categorical_cols = x_train.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = x_train.select_dtypes(include=["number"]).columns.tolist()

    # Create preprocessor with categorical first, then numerical
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols),
            ("num", MinMaxScaler(), numerical_cols),
        ]
    )

    # Create pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("feature_selection", SelectKBest(score_func=f_regression)),
            ("regressor", LinearRegression()),
        ]
    )

    return pipeline


def optimize_hyperparameters(pipeline, x_train, y_train):
    """
    Paso 4: Optimizar los hiperparámetros del pipeline usando validación cruzada.
    - Usar 10 splits para la validación cruzada
    - Usar el error medio absoluto (MAE) para medir el desempeño
    """
    # Define hyperparameter grid
    # We search over a range of k values for feature selection
    param_grid = {
        "feature_selection__k": range(1, 9),
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    grid_search.fit(x_train, y_train)

    return grid_search


def save_model(model):
    """
    Paso 5: Guardar el modelo comprimido con gzip.
    """
    # Create models directory if it doesn't exist
    os.makedirs("files/models", exist_ok=True)

    # Save model with gzip compression
    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(model, f)


def calculate_and_save_metrics(model, x_train, y_train, x_test, y_test):
    """
    Paso 6: Calcular las métricas y guardarlas en un archivo JSON.
    """
    # Create output directory if it doesn't exist
    os.makedirs("files/output", exist_ok=True)

    # Predict on train and test sets
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Calculate metrics for train set
    train_metrics = {
        "type": "metrics",
        "dataset": "train",
        "r2": r2_score(y_train, y_train_pred),
        "mse": mean_squared_error(y_train, y_train_pred),
        "mad": mean_absolute_error(y_train, y_train_pred),
    }

    # Calculate metrics for test set
    test_metrics = {
        "type": "metrics",
        "dataset": "test",
        "r2": r2_score(y_test, y_test_pred),
        "mse": mean_squared_error(y_test, y_test_pred),
        "mad": mean_absolute_error(y_test, y_test_pred),
    }

    # Save metrics to JSON file
    with open("files/output/metrics.json", "w") as f:
        f.write(json.dumps(train_metrics) + "\n")
        f.write(json.dumps(test_metrics) + "\n")


def main():
    """Main function to execute all steps."""
    # Load data
    train_data, test_data = load_data()

    # Preprocess data
    train_data, test_data = preprocess_data(train_data, test_data)

    # Split data
    x_train, y_train, x_test, y_test = split_data(train_data, test_data)

    # Create pipeline
    pipeline = create_pipeline(x_train)

    # Optimize hyperparameters
    model = optimize_hyperparameters(pipeline, x_train, y_train)

    # Save model
    save_model(model)

    # Calculate and save metrics
    calculate_and_save_metrics(model, x_train, y_train, x_test, y_test)

    print("Model training and evaluation completed successfully!")


if __name__ == "__main__":
    main()
