import requests
import pandas as pd
import json
import numpy as np
import joblib
import pickle
import pytest

with open('app/columns_name.pickle', 'rb') as f:
    columns_names = pickle.load(f)

with open('app\columns_name_nums.pickle', 'rb') as f:
    nums_columns_name = pickle.load(f)

pipeline = joblib.load('app/pipeline-xgboost-scoring')
pipeline_nums = joblib.load('app/pipeline-nums-col-scoring')

test_data = pd.read_csv('data/X_test.csv').loc[[0]].to_json()

def test_check_columns():
    df = pd.read_json(test_data)
    columns = df.columns
    assert all(item in columns.tolist() for item in columns_names)

def test_check_pipeline_predict():
    df = pd.read_json(test_data)
    y_pred = pipeline.predict(df)
    y_proba = pipeline.predict_proba(df)
    assert y_pred[0] in [0, 1]

    assert len([y_proba[0][1]]) == 1

    assert (y_proba[0][1] >= 0) & (y_proba[0][1] <= 1)

def test_shape_features_prep():
     df = pd.read_json(test_data)
     actual_shape = pipeline[0].transform(df).shape
     expected_shape = (1, 158)
     assert actual_shape == expected_shape

def test_check_trans_nums():
     df = pd.read_json(test_data)
     data = pipeline_nums.transform(df[nums_columns_name])
     assert data.shape == (1, 147)










# def test_can_call_endpoint_predict():
#     response = requests.get(ENDPOINT + '/predict', json = test_data.to_json())
#     assert response.status_code == 200

# def test_can_call_endpoint_shap():
#     response = requests.get(ENDPOINT + '/api/shap', json = test_data.to_json())
#     assert response.status_code == 200

# def test_shape_features_prep():
#     actual_shape = pipeline[0].transform(test_data).shape
#     expected_shape = (1, 158)
#     assert actual_shape == expected_shape

# def test_shap_base_value():
#     response = requests.get(ENDPOINT + '/api/shap', json = test_data.to_json())
#     data = json.loads(response.text)
#     assert np.array(data['shapley_base_values']).shape == (1,)