import pandas as pd
# import joblib
from flask import Flask, jsonify, request, make_response, send_file
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# import shap
# import pickle
# pipeline = joblib.load('app/pipeline-nums-col-scoring')
# pipeline_nums = joblib.load('app/pipeline-nums-col-scoring')

# X_train = pd.read_csv('data/X_train.csv')
# y_train = pd.read_csv('data/y_train.csv')
# pipeline = joblib.load('app/pipeline-xgboost-scoring')
# pipeline_nums = joblib.load('app/pipeline-nums-col-scoring')

# pipeline.fit(X_train, y_train)

# with open('app/columns_name_nums.pickle', 'rb') as f:
#     nums_columns_name = pickle.load(f)

# pipeline_nums.fit(X_train[nums_columns_name])
iris = datasets.load_iris()

data=pd.DataFrame({
    'sepal length':iris.data[:,0],
    'sepal width':iris.data[:,1],
    'petal length':iris.data[:,2],
    'petal width':iris.data[:,3],
    'species':iris.target
})

X=data[['sepal length', 'sepal width', 'petal length', 'petal width']]
y=data['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf=RandomForestClassifier(n_estimators=100)

clf.fit(X_train,y_train)
app = Flask(__name__)

@app.route('/')
def test1():
     return 'Test'

@app.route('/test')
def test2():
     return {'test predict': clf.predict([[3, 5, 4, 2]]).tolist()}

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