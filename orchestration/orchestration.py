from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, roc_auc_score, roc_curve, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint
from torch.utils.data import DataLoader, TensorDataset
from prefect import task, flow
import numpy as np
import mlflow
import mlflow.pyfunc
import os
import pandas as pd

#Task 1: Load data
@task(
    retries = 3,
    retry_delay_seconds = 2,
    name = 'Load data',
    tags = ['load_data'],
    description = 'Load the processed data'
)
def load_data(data_path = 'data/processed/processed_normalized_data.csv'):
    data = pd.read_csv(data_path)
    return data

#Task 2: Divide the data in features-target
@task(
    retries = 3,
    retry_delay_seconds = 2,
    name = 'Divide data in features-target',
    tags = ['processing_Data'],
    description = 'Divide the pandas dataframe in features dataframe and target dataframe'
)
def processing_data(data):
    features = data.iloc[:, :-1]
    target = data.iloc[:, -1]
    return features, target

#Task 3: Split data
@task(
    retries = 3,
    retry_delay_seconds = 2,
    name = 'Split data',
    tags = ['split_data'],
    description = 'Split the data using sklearn train-test split'
)
def split_data(features, target, test_size_value = 0.2):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = test_size_value, random_state = 42)
    return X_train, X_test, y_train, y_test

#Task 4: Obtain the best model for the mlflow experiments
@task(
    retries = 3,
    retry_delay_seconds = 2,
    name = 'Obtaining the best model',
    tags = ['import_model'],
    description = 'Load the best model from the mlflow traking obtaining using the models notebook'
)
def import_model():
    mlruns_path = "file:///tracking"
    mlflow.set_tracking_uri(f"file:///tracking")
    experiments = mlflow.search_experiments()
    best_model_uri = None
    best_metric = float("-inf")
    for experiment in experiments:
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        for _, run in runs.iterrows():
            if 'metrics.accuracy' in run:
                accuracy = run['metrics.accuracy']
                if accuracy > best_metric:
                    best_metric = accuracy
                    best_model_uri = f"runs:/{run.run_id}/model"
    if best_model_uri:
        best_model = mlflow.pyfunc.load_model(best_model_uri)
        print(f"Modelo con mejor métrica (accuracy: {best_metric}) cargado exitosamente.")
    else:
        print("No se encontró ningún modelo con la métrica especificada.")
    return best_model

#Task 5: Run inference with the best model
@task(
    retries = 3,
    retry_delay_seconds = 2,
    name = 'Inference',
    tags = ['inference'],
    description = 'Inference with the best model obtained from mlflow'
)
def inference(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    return accuracy

#Flow
@flow(retries = 3, retry_delay_seconds = 5, log_prints = True)
def churn_classification():
    data = load_data()
    features, target = processing_data(data)
    X_train, X_test, y_train, y_test = split_data(features, target, test_size_value = 0.2)
    model = import_model()
    inference(model, X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    churn_classification()
