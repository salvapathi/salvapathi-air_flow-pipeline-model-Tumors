import os
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
from sklearn.metrics import *
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn import set_config
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
import joblib
from datetime import datetime
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import json 
import seaborn  as sns
import matplotlib.pyplot as plt
from scipy import stats
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet