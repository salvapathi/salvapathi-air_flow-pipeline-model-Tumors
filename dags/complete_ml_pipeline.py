#Training the Model 
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
    "complete_ml_pipeline",
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
    data.to_csv(r"./dags/extracted_data.csv",index=False)

def Transforming():
    # Load the processed data
    data = pd.read_csv(r"./dags/extracted_data.csv")
    #changing the formate 
    data["Diagnosis"]=data["Diagnosis"].replace("M","Malignant_tumors").replace("B","Benign_tumors")
    data.to_csv(r"./dags/clean_data.csv",index=False)

def ml_model():
    # load clean data
    data=pd.read_csv(r"./dags/clean_data.csv")
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

   

    pipe=make_pipeline(columns_ct,model)
    set_config(display="diagram")
    pipe.fit(x_train,y_train)

    #saving the model to a file 
    model_filename = "./dags/trained_modelsvc.joblib"
    joblib.dump(pipe, model_filename)
    print("The model is  saved",model)

    #prediction
    y_pred=pipe.predict(x_test)
    x_test["True_values"]=y_test
    x_test["predicted_vaues"]=y_pred.tolist()
    x_test.to_csv(r"./dags/test_data.csv",index=False)


def model_evaluation():
    test_data=pd.read_csv(r"./dags/test_data.csv")
    accuracy = accuracy_score(test_data["True_values"],test_data["predicted_vaues"])
    print(f"Accuracy OF THE MODEL {model} : {accuracy:.2f}")


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
