import requests
import pandas as pd
from app import flask_api
import json
import numpy as np


ENDPOINT = 'https://ocr-scoring-app.azurewebsites.net'
test_data = pd.read_csv('data\X_test.csv').loc[[0]]

def test_can_call_endpoint_predict():
    response = requests.get(ENDPOINT + '/predict', json = test_data.to_json())
    assert response.status_code == 200

def test_can_call_endpoint_shap():
    response = requests.get(ENDPOINT + '/api/shap', json = test_data.to_json())
    assert response.status_code == 200

def test_shape_features_prep():
    actual_shape = flask_api.features_prep(test_data).shape
    expected_shape = (1, 158)
    assert actual_shape == expected_shape

def test_shap_base_value():
    response = requests.get(ENDPOINT + '/api/shap', json = test_data.to_json())
    data = json.loads(response.text)
    assert np.array(data['shapley_base_values']).shape == (1,)