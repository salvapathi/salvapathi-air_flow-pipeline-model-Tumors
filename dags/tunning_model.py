import airflow
import os 
path1="/opt/airflow/data_preprocessed"
path2="/opt/airflow/saved_models"
path3="/opt/airflow/prediction_&_report"
os.chdir(path1)
os.chdir(path2)
os.chdir(path3)
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
# from airflow.providers.common.operators.file_operator import FileOperator
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn import set_config
from ucimlrepo import fetch_ucirepo 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import joblib
from datetime import datetime
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
svc=SVC()
KNN=KNeighborsClassifier()
LR=LogisticRegression()
RF=RandomForestClassifier()
model=RF
param_grid = {
    'parameters__n_estimators': [100,200],
    'parameters__max_depth': [None,20,30,100]
    }
# Define the DAG
dag = DAG(
    "Tunning_model",
    default_args={
        "owner": "salvapathi_naidu",
        "start_date": airflow.utils.dates.days_ago(1),  # Set the start date to one day ago
    },
    schedule_interval="@daily",
)

# Define the tasks
def extraction():
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
    X = breast_cancer_wisconsin_diagnostic.data.features 
    y = breast_cancer_wisconsin_diagnostic.data.targets 
    X["Diagnosis"]=y["Diagnosis"]
    data=X
    data.to_csv("/opt/airflow/data_preprocessed/raw_data.csv",index=False)

def Transforming():
    # Load the processed data
    data = pd.read_csv("/opt/airflow/data_preprocessed/raw_data.csv")
    #changing the formate 
    data["Diagnosis"]=data["Diagnosis"].replace("M","Malignant_tumors").replace("B","Benign_tumors")
    data.to_csv("/opt/airflow/data_preprocessed/cleaned_data.csv",index=False)

def ml_model():
    # load clean data
    data=pd.read_csv("/opt/airflow/data_preprocessed/cleaned_data.csv")
    numeric_columns=data.select_dtypes(include='number').columns.tolist()
    categorical_columns =data.select_dtypes(include=['object']).columns.tolist()
    categorical_columns.remove("Diagnosis") #it is

    x_train,x_test,y_train,y_test=train_test_split(data.drop(["Diagnosis"],axis=1),data["Diagnosis"],test_size=0.2,stratify=data["Diagnosis"])
    print("Train Data Shape:",x_train.shape,"Test Data Shape:",x_test.shape)
    print(x_train)
    print(y_train)
    print(x_test)

    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy='mean')),
                                        ("Standard Scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy='most_frequent')),
        ("onehot", OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    columns_ct = ColumnTransformer([
        ("Standardization", numeric_transformer, numeric_columns),
        ("onehotencoder", categorical_transformer,categorical_columns)
    ])

    #pipe=make_pipeline(columns_ct,model)

    pipe = Pipeline(steps=[
    ("preprocessor", columns_ct),
    ("parameters",model)])

    grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(x_train, y_train)

    print("Best Hyperparameters:", grid_search.best_params_)
    best_model_pipeline = grid_search.best_estimator_
    best_model_pipeline.fit(x_train,y_train)

    model_name = type(model).__name__
    #saving the model to a file 
    model_filename = "/opt/airflow/saved_models/{model_name}_tunned_model.joblib"
    joblib.dump(best_model_pipeline, model_filename)
    print("The model is  saved",model)

    #prediction
    y_pred=best_model_pipeline.predict(x_test)
    x_test["True_values"]=y_test
    x_test["predicted_vaues"]=y_pred.tolist()
    x_test.to_csv(f"/opt/airflow/prediction_&_report/prediction_tunned_{model_name}tunned_model.csv",index=False)
    train_score = best_model_pipeline.score(x_train, y_train)
    test_score = best_model_pipeline.score(x_test,y_test)
    print(f'Training Score: {train_score}')
    print(f'Test Score: {test_score}')

def model_evaluation(filename_prefix="metrics"):
    model_name = type(model).__name__
    test_data=pd.read_csv(f"/opt/airflow/prediction_&_report/prediction_tunned_{model_name}tunned_model.csv")
    accuracy = accuracy_score(test_data["True_values"],test_data["predicted_vaues"])
    precision=precision_score(test_data["True_values"],test_data["predicted_vaues"],pos_label='Malignant_tumors')
    recall=recall_score(test_data["True_values"],test_data["predicted_vaues"],pos_label='Malignant_tumors')
    f1=f1_score(test_data["True_values"],test_data["predicted_vaues"],pos_label='Malignant_tumors')
    cm=confusion_matrix(test_data["True_values"],test_data["predicted_vaues"])
    classification_score=classification_report(test_data["True_values"],test_data["predicted_vaues"])
    # Get the name of the model
    model_name = type(model).__name__
    filename = f"/opt/airflow/prediction_&_report/prediction_{filename_prefix}_tunned_{model_name}.txt"
    # Open the file in write mode and write the metrics
    with open(filename, 'a') as file:
        file.write(f"\n")
        file.write(model_name + '\n')
        file.write(f'Accuracy Score : {accuracy}\n')
        file.write(f'Precision Score : {precision}\n')
        file.write(f'Recall Score : {recall}\n')
        file.write(f'F1 score : {f1}\n')
        file.write('Confusion Matrix :\n')
        file.write(str(cm))
        file.write(f"classification_report :{classification_score}\n")
        file.write(f"\n")


load_data_task = PythonOperator(
    task_id="Extraction_of_data_from_UIC",
    python_callable=extraction,
    dag=dag,
)

Transforming_task = PythonOperator(
    task_id="Transforming_the_data",
    python_callable=Transforming,
    dag=dag,
)

model_building_task = PythonOperator(
    task_id="model_building",
    python_callable=ml_model,
    dag=dag,
 )
model_evaulation_task = PythonOperator(
    task_id="model_evaluation",
    python_callable=model_evaluation,
    dag=dag,
)

# Set the task dependencies
load_data_task >> Transforming_task >> model_building_task >> model_evaulation_task
