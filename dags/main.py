from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
import datetime
from airflow.operators.dummy import DummyOperator
import sys
import os 
import requests
import json
import pandas as pd 
from airflow.utils.task_group import TaskGroup
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from joblib import dump
from airflow.models import Variable

Variable.set(key="cities", value=['paris', 'london', 'washington'] )

with DAG(
    dag_id='dag_eval',
    description='dag weather',
    tags=['eval', 'weather'],
    schedule_interval=None,
    default_args={
        'owner': 'airflow',
        'start_date': days_ago(0),
    }
) as my_dag:
    # Fonctions
    def getWeatherData():

        api = "a3350b6e27ce9140362069a22bec5f0f"
        cities = ['paris', 'london', 'washington']
        data = []
        for citie in cities:
            data.append(requests.get(f"https://api.openweathermap.org/data/2.5/weather?q={citie}&appid={api}").content.decode('utf-8'))
        data_json_to_object = [json.loads(x) for x in data]
        with open(f'/app/raw_files/{datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}.json', 'w') as json_file:
            json.dump(data_json_to_object, json_file, indent=4)

    def transform_data_into_csv(n_files=None, filename='data.csv'):
        parent_folder = '/app/raw_files'
        files = sorted(os.listdir(parent_folder), reverse=True)
        if n_files:
            files = files[:n_files]

        dfs = []

        for f in files:
            with open(os.path.join(parent_folder, f), 'r') as file:
                data_temp = json.load(file)
            for data_city in data_temp:
                dfs.append(
                    {
                        'temperature': data_city['main']['temp'],
                        'city': data_city['name'],
                        'pression': data_city['main']['pressure'],
                        'date': f.split('.')[0]
                    }
                )

        df = pd.DataFrame(dfs)

        #print('\n', df.head(10))

        df.to_csv(os.path.join('/app/clean_data', filename), index=False)

    def compute_model_score(model, X, y,xcom_key, task_instance):
        
        # computing cross val
        cross_validation = cross_val_score(
            model,
            X,
            y,
            cv=3,
            scoring='neg_mean_squared_error')

        model_score = cross_validation.mean()
        task_instance.xcom_push(
            key=xcom_key,
            value=model_score
        )
        

    def train_and_save_model(model, X, y, path_to_model='./app/model.pckl'):
        # training the model
        model.fit(X, y)
        # saving model
        print(str(model), 'saved at ', path_to_model)
        dump(model, path_to_model)

    def prepare_data(path_to_data='/app/clean_data/fulldata.csv'):
        # reading data
        df = pd.read_csv(path_to_data)
        # ordering data according to city and date
        df = df.sort_values(['city', 'date'], ascending=True)

        dfs = []

        for c in df['city'].unique():
            df_temp = df[df['city'] == c]

            # creating target
            df_temp.loc[:, 'target'] = df_temp['temperature'].shift(1)

            # creating features
            for i in range(1, 10):
                df_temp.loc[:, 'temp_m-{}'.format(i)
                            ] = df_temp['temperature'].shift(-i)

            # deleting null values
            df_temp = df_temp.dropna()

            dfs.append(df_temp)

        # concatenating datasets
        df_final = pd.concat(
            dfs,
            axis=0,
            ignore_index=False
        )

        # deleting date variable
        df_final = df_final.drop(['date'], axis=1)

        # creating dummies for city variable
        df_final = pd.get_dummies(df_final)

        features = df_final.drop(['target'], axis=1)
        target = df_final['target']

        return features, target

    X, y = prepare_data('/app/clean_data/fulldata.csv')

    def bestModel(task_instance):
        score_lr = task_instance.xcom_pull(
            key="score_lr",
            task_ids='model_cross_validation.LinearRegression'
        )
        score_dt = task_instance.xcom_pull(
            key="score_dt",
            task_ids='model_cross_validation.DecisionTreeRegressor'
        )
        score_rf = task_instance.xcom_pull(
            key="score_rf",
            task_ids='model_cross_validation.RandomForestRegressor'
        )

    # Vérification des scores
        print(f"Scores: LR: {score_lr}, DT: {score_dt}, RF: {score_rf}")

            # using neg_mean_square_error
        if score_lr < score_dt and score_lr < score_rf:
            train_and_save_model(
                LinearRegression(),
                X,
                y,
                '/app/clean_data/best_model.pickle'
            )
        elif score_dt < score_lr and score_dt < score_rf:
            train_and_save_model(
                DecisionTreeRegressor(),
                X,
                y,
                '/app/clean_data/best_model.pickle'
            )
        else :
            train_and_save_model(
                RandomForestRegressor(),
                X,
                y,
                '/app/clean_data/best_model.pickle'
            )
    

    


    task_1 = PythonOperator(
        task_id='getData',
        python_callable=getWeatherData,
        dag=my_dag
    )
    
    
    task_2 = PythonOperator(
        task_id='20csv',
        python_callable=transform_data_into_csv,
        op_kwargs= {
        'n_files': 20,
        'filename':'data.csv',
        },
        dag=my_dag
    )

    task_3 = PythonOperator(
        task_id='fullcsv',
        python_callable=transform_data_into_csv,
        op_kwargs= {
        'n_files': None,
        'filename':'fulldata.csv',
        },
        dag=my_dag
    )

    with TaskGroup("model_cross_validation") as group_4:
        task_4_1 = PythonOperator(
                task_id='LinearRegression',
                python_callable=compute_model_score,
                op_kwargs= {
                'model': LinearRegression(),
                'X':X,
                'y':y,
                'xcom_key':'score_lr'
                },
                dag=my_dag
            )

        task_4_2 = PythonOperator(
                task_id='DecisionTreeRegressor',
                python_callable=compute_model_score,
                op_kwargs= {
                'model': DecisionTreeRegressor(),
                'X':X,
                'y':y,
                'xcom_key':'score_dt'
                },
                dag=my_dag
            )

        task_4_3 = PythonOperator(
                task_id='RandomForestRegressor',
                python_callable=compute_model_score,
                op_kwargs= {
                'model': RandomForestRegressor(),
                'X':X,
                'y':y,
                'xcom_key':'score_rf'
                },
                dag=my_dag
            )

    task_5 = PythonOperator(
                task_id='BestModel',
                python_callable=bestModel,
                dag=my_dag
            )   

    

    start_task = DummyOperator(task_id='start_task')
    end_task_1 = DummyOperator(task_id='end_task_1', trigger_rule="all_done")
    end_task_2 = DummyOperator(task_id='end_task_2', trigger_rule="all_done")


start_task >> task_1 >> task_3 >> group_4 >> task_5 >> end_task_2
start_task >> task_1 >> task_2 >> end_task_1







    

