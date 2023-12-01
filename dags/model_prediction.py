#model prediction
import airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
# from airflow.providers.common.operators.file_operator import FileOperator
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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
import joblib
RF=RandomForestClassifier()
svc=SVC()
KNN=KNeighborsClassifier()
model=svc

# Define the DAG
dag = DAG(
    "prediction_of_Tummor",
    default_args={
        "owner": "salvapathi_naidu",
        "start_date": airflow.utils.dates.days_ago(1),  # Set the start date to one day ago
    },
    schedule_interval="@daily",
)

# Define the tasks
def test_data():
    data=pd.read_csv(r"./dags/clean_data.csv")
    data.drop(["Diagnosis"],axis=1,inplace=True)
    sampled_data = data.sample(n=110) 
    sampled_data.to_csv(r"./dags/test_data_cancer.csv",index=False) #  x_test.to_csv(r"./dags/test_data.csv",index=False)

def prediction():
    input_data = pd.read_csv(r"./dags/test_data_cancer.csv")
    model_filename = r"./dags/trained_modelsvc.joblib"
    loaded_pipe = joblib.load(model_filename)
    y_pred = loaded_pipe.predict(input_data)
    input_data["Predicted_values"] = y_pred.tolist()  # Corrected column name
    input_data.to_csv(r"./dags/test_data_predicted.csv", index=False)


load_data_task = PythonOperator(
    task_id="load_the_data",
    python_callable=test_data,
    dag=dag,
)

prediction_task = PythonOperator(
    task_id="Prediction_data",
    python_callable=prediction,
    dag=dag,
)
load_data_task >>prediction_task