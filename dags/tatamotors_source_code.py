import os 
dependies_path="/opt/airflow/scripts_python"
os.chdir(dependies_path)
exec(open("/opt/airflow/scripts_python/tatamotors_dependies.py").read())
dag = DAG(
    "Tatamotors_share_source",
    default_args={
        "owner": "salvapathi_Naidu",
        "start_date": airflow.utils.dates.days_ago(1),  # Set the start date to one day ago
    },
    schedule_interval="@daily",
)
path1="/opt/airflow/shares_input"
file_link="/opt/airflow/shares_input/TATAMOTORS.csv"
dropping_columns=["Open","Adj Close","Volume","High","Low","Close"]
window_size=5
span =3
value_replaced=np.nan
model_name=LinearRegression()
#model_name=XGBRegressor()
#model_name=ElasticNet()
model_name1= type(model_name).__name__
Transformed_data_link="/opt/airflow/shares_input/Transformed_data.csv"
columns_dropped_ml_building=["Simple_Moving_Average","EMA","transformed_Average_Price"]

# Define the TASK -1
def load_data(file_link):
    try:
        if file_link[-3:] == "csv":
            data = pd.read_csv(file_link)
            return data
        elif file_link[-3:] == "xls":
            data = pd.read_excel(file_link)
        else:
            raise ValueError("Unsupported file format")
        
        print("File extraction successful.")
        #print(data)
    except FileNotFoundError:
        print(f"Error: File not found at {file_link}")
    except Exception as e:
        print("An undefined error as occqured")
    

def avgshare_price(data):
    data["Average_Price" ] = ((data["High"]+data["Low"])/2)
    print("average _price",data["Average_Price"])
    


def wrangling(data, value_replaced):
    duplicates_before = data.duplicated().sum()
    data.drop_duplicates(inplace=True)
    data["Average_Price"] = data["Average_Price"].replace(0, value_replaced)
    replaced_rows = data["Average_Price"].replace(0, value_replaced).sum()
    print(f"Duplicates before dropping: {duplicates_before}")
    print(f"Rows with 0 replaced: {replaced_rows}")

        
    
def columns_dropping(data,dropping_columns):
    data.drop(dropping_columns,axis=1,inplace=True)
    print(f"The columns are dropped:{dropping_columns}")
  


def calculating_avgs(data,window_size,span):
    data['Simple_Moving_Average'] = data['Average_Price'].rolling(window=window_size).mean()
    data['Simple_Moving_Average'].iloc[0] =data['Average_Price'].iloc[0]
    for i in range(1, data.shape[0]):
        if pd.isnull(data['Simple_Moving_Average'].iloc[i]):
            data['Simple_Moving_Average'].iloc[i] = data['Average_Price'].iloc[i-1:i+1].mean()
    # print(data['Simple_Moving_Average'][0:10])

    data['EMA'] = data['Average_Price'].ewm(span=span, adjust=False).mean()
    data['EMA'].iloc[0] =data['Average_Price'].iloc[0]
    for i in range(1, data.shape[0]):
        if pd.isnull(data['EMA'].iloc[i]):
            data['EMA'].iloc[i] = data['Average_Price'].iloc[i-1:i+1].mean()


    mask = data["Average_Price"].notna()  # Mask for non-null values in 'Average_Price'
    SMA_rmse = np.sqrt(mean_squared_error(data["Average_Price"][mask], data["Simple_Moving_Average"][mask]))
    EMA_rmse = np.sqrt(mean_squared_error(data["Average_Price"][mask], data["EMA"][mask]))

    
        
    if SMA_rmse<EMA_rmse:
        data["Average_Price"]=data["Simple_Moving_Average"].where(data["Average_Price"].isna(),data["Average_Price"])
        print(f"used simple moving average to fill null values :{SMA_rmse} this error")
        used=f"Simple moving average to fill null values :{SMA_rmse} this error"
        
    elif EMA_rmse<SMA_rmse:
        data["Average_Price"]=data["EMA"].where(data["Average_Price"].isna(),data["Average_Price"])
        print(f"used  EMA moving average to fill null values :{EMA_rmse} this error")
        used=f"EMA moving average to fill null values :{EMA_rmse} due to its minimum Error"
        
    else:
        print("THERE ARE NO NULL VALUES TO FILL")
    #REPORT OF THE DATA WHILE CALULATING THE ERRORS 
    Rmse_SSE_EMA = {'RMSE OF  SSE': SMA_rmse, 'Root Mean Squared Error EMA': EMA_rmse,"MOVING AVERAGE USED":used}
    json_data = json.dumps(Rmse_SSE_EMA, indent=3)
    json_file_path = '/opt/airflow/shares_input/RMSE_SSE_EMA.json'
    with open(json_file_path, 'w') as json_file:
        json_file.write(json_data)
 
    data.to_csv("/opt/airflow/shares_input/cleaned_data.csv",index=False)

def extraction(**kwargs):
    ti = kwargs['ti']
    data = load_data(file_link)
    avgshare_price(data)
    wrangling(data, value_replaced)
    columns_dropping(data, dropping_columns)
    calculating_avgs(data, window_size, span)
            
#_____________________________-----------------------------____________________________________---------------
    #functions for data Transformation
cleaned_link="/opt/airflow/shares_input/cleaned_data.csv"
def load_clean_data(cleaned_link):
    data=pd.read_csv(cleaned_link)
    
    #the data is right skeweed means positively skewed so we need to tranform the data to normal distribution
    #data["log_transformed_data"]=np.log1p(data["Average_Price"])
    #data["log_transformed_Average_Price"] = np.log(data['Average_Price'])
    column_to_transform = data['Average_Price']

# Apply Box-Cox transformation
    transformed_column, lambda_value = stats.boxcox(column_to_transform)

    # Add the transformed column to the DataFrame
    data['transformed_Average_Price'] = transformed_column

    sns.kdeplot(data["Average_Price"],color="blue")
    save_path="/opt/airflow/shares_input/kde_averageprice.png"
    plt.savefig(save_path)

    sns.histplot(data["transformed_Average_Price"],bins=80,kde=True,color="red")
    save_path2="/opt/airflow/shares_input/price_normal_distribution.png"
    plt.savefig(save_path2)
    data.to_csv("/opt/airflow/shares_input/Transformed_data.csv",index=False)
    

def Transformation(**kwargs):
     ti = kwargs['ti']
     data=load_clean_data(cleaned_link)

#-----------__________________-----------------_____________________----------------------_________------



def ML_process(Transformed_data_link):
    data=pd.read_csv(Transformed_data_link)
    return data
def columns_ml(data,columns_dropped_ml_building):
    try:
        if data is not None:
            ml_data=data.drop(columns=columns_dropped_ml_building)
            print("Columns dropped successfully.")
        else:
            print("Error: Input DataFrame is None.")
            return None
    except KeyError as e:
        print(f"Error: One or more columns not found in the DataFrame. {e}")
    return ml_data
        
        
def model(ml_data):
    ml_data['Date'] = pd.to_datetime(ml_data['Date'])
    pre_train=ml_data[ml_data["Date"]<='2023-12-01']
    x_train=pre_train.drop(["Average_Price"],axis=1)
    y_train=pre_train["Average_Price"]
    pre_x_test=ml_data[ml_data["Date"]>='2023-12-02']
    x_test=pre_x_test.drop(["Average_Price"],axis=1)
    y_test=pre_x_test["Average_Price"]
    pipe=Pipeline([('scaler', StandardScaler()), ('linear Regressor ',model_name)])
    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)
    train_score = pipe.score(x_train, y_train)
    test_score = pipe.score(x_test,y_test)
    print(f'Training Score : {train_score}')
    
    x_test["True_values"]=y_test
    x_test["predicted_vaues"]=y_pred.tolist()
    x_test.to_csv(f"/opt/airflow/shares_input/prediction{model_name1}.csv",index=False)
    Model_evaluation(y_test, y_pred)
    Trust_Score(train_score, test_score)
    #return y_test,y_pred,train_score,test_score

def Model_evaluation(y_test,y_pred):
    mse=mean_squared_error(y_test,y_pred)
    rmse=np.sqrt(mse)
    mae=mean_absolute_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)
    jason_file_path="/opt/airflow/shares_input/model_evaluation.json"
    Model_evaluation={ "MeanSquared error":mse,"Root mean squared error":rmse,"Mean Absolute Error":mae,"R-Square":r2}
    jason_data=json.dumps(Model_evaluation,indent=4)
    with open(jason_file_path,"w") as json_file:
        json_file.write(jason_data)
def Trust_Score(train_score,test_score):
    json_file_path = '/opt/airflow/shares_input/train_test_score.json'
    Train_Test_Score={"Train_score":train_score,"Test_score":test_score}
    json_score = json.dumps(Train_Test_Score, indent=2)
    with open(json_file_path, 'w') as json_file:
            json_file.write(json_score)
        



def Building_Model(**kwargs):
    ti = kwargs['ti']
    data=ML_process(Transformed_data_link)
    ml_data=columns_ml(data,columns_dropped_ml_building)
    model(ml_data)

load_data_task = PythonOperator(
    task_id="Extraction_of_data",
    python_callable=extraction,
    provide_context=True,
    dag=dag,
)

Transformation_data = PythonOperator(
    task_id="Transformation",
    python_callable=Transformation,
    provide_context=True,
    dag=dag,
)
Machine_Learning_model=PythonOperator(
    task_id="machine_learning_Pipeline",
    python_callable=Building_Model,
    provide_context=True,
    dag=dag,
)


# Set the task dependencies
load_data_task >> Transformation_data >> Machine_Learning_model