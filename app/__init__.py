import pandas as pd
import joblib
from flask import Flask, jsonify, request, make_response, send_file
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import shap
import pickle

#X_train = pd.read_csv('data/X_train.csv')
test = 'https://ocrscoringapp.blob.core.windows.net/containerocr/y_test.csv?sp=r&st=2023-02-15T14:05:58Z&se=2023-02-15T22:05:58Z&spr=https&sv=2021-06-08&sr=b&sig=tsLlGwruDdn7TkG8OMWXWl0Im8PlfUCtOQIM7MyPRE4%3D'
y_train = pd.read_csv(test)

app  = Flask(__name__)

@app.route('/')
def test():
    return {"test": y_train.columns.tolist()}
@app.route('/test1')
def tt():
    return {'testf':y_train.TARGET[0]}

@app.route('/test2')
def test2():
    return {"test2":'Test!!'}
# pipeline = joblib.load('app/pipeline-xgboost-scoring')
# pipeline_nums = joblib.load('app/pipeline-nums-col-scoring')

# pipeline.fit(X_train, y_train)

# with open('app/columns_name_nums.pickle', 'rb') as f:
#     nums_columns_name = pickle.load(f)

# pipeline_nums.fit(X_train[nums_columns_name])
#  

# @app.route('/predict', methods = ['GET', 'POST']) #:y_pred[0], 'Class probabilities': y_proba[0][0]
# def predict():
#      data = request.get_json()
#      df = pd.read_json(data)
#      y_pred = pipeline.predict(df)
#      y_proba = pipeline.predict_proba(df)

#      return jsonify({"Class":y_pred[0].tolist(), 'Class probabilities': y_proba[0][1].tolist()})

# def features_prep(df): 
#      data = pipeline[0].transform(df)
#      return data

# @app.route('/api/shap', methods = ['GET', 'POST'])
# def shap_values():
#      data = request.get_json()
#      df = pd.read_json(data)
#      df = features_prep(df)
#      explainer = shap.TreeExplainer(pipeline[1])
#      shapley = explainer(df)
#      shapley_values = shapley.values.tolist()
#      shapley_base_values = shapley.base_values.tolist()
#      shapley_data = shapley.data.tolist()
#      return jsonify({"shapley_values":shapley_values, 'shapley_base_values': shapley_base_values, 'shapley_data': shapley_data})

# @app.route('/transform_nums', methods = ['GET', 'POST'])
# def transform_data():
#      data = request.get_json()
#      df = pd.read_json(data)
#      data = pipeline_nums.transform(df[nums_columns_name])
#      data = pd.DataFrame(data, columns = nums_columns_name)
#      return jsonify({'data': data.to_json()})