import streamlit as st 
import pandas as pd
import requests
import json
import plotly.graph_objects as go
import shap 
import numpy as np
import matplotlib.pyplot as plt 
test = pd.read_csv('data\X_test.csv')

# Partie proba
response = requests.get('http://127.0.0.1:5000/predict', json = test.loc[[0]].to_json())
response_dict = json.loads(response.text)
class_prediction = response_dict['Class']
class_proba = round(response_dict['Class probabilities'],2)

st.title('Prediction')

if class_prediction == 0:
    st.markdown('### Prêt refuser')
else: st.markdown('### Prêt accepter')

fig = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = class_proba,
    mode = "gauge+number",
    title = {'text': ""},
    gauge = {'axis': {'range': [0, 1]},
             'bar': {'color':'darkblue'},
             'steps' : [
                 {'range': [0, 0.5], 'color': "green", 'name':'Prêt accepter'},
                 {'range': [0.5, 1], 'color': "red", 'name': 'Prêt refuser'}],
             'bordercolor': "black",
             'borderwidth': 1}))
fig.update_layout(paper_bgcolor = "lavender", font = {'color': "darkblue", 'family': "Arial"}, legend_title = 'test', showlegend = True)

st.plotly_chart(fig)

# Partie shapley
response_shapley = requests.get('http://127.0.0.1:5000/api/shap', json = test.loc[[0]].to_json())
response_dict_shapley = json.loads(response_shapley.text)
shapley_values = np.array(response_dict_shapley['shapley_values'])
shapley_base_values = np.array(response_dict_shapley['shapley_base_values'])
shapley_data = np.array(response_dict_shapley['shapley_data'])

explainer = shap.Explanation(shapley_values, shapley_base_values, shapley_data)
st.pyplot(shap.waterfall_plot(explainer[0][:, 1], show = False))


#X_train = pd.read_csv('data/X_train.csv')
app  = Flask(__name__)
#test = 'https://ocrscoringapp.blob.core.windows.net/containerocr/X_train.csv?sp=r&st=2023-02-15T15:36:12Z&se=2023-02-15T23:36:12Z&spr=https&sv=2021-06-08&sr=b&sig=ESWBGCG5wLQvo1rj1sElWeKWSL6ql5fa3ScBUumrFF4%3D'
#y_train = pd.read_csv(test)
# x_train = pd.read_csv(test)
# locations = x_train.loc[[0]]
with open('app/columns_name_nums.pickle', 'rb') as f:
    nums_columns_name = pickle.load(f)

pipeline_nums = joblib.load('app/pipeline-nums-col-scoring')
pipeline = joblib.load('app/pipeline-xgboost-scoring')

#x_scaled = pipeline_nums.transform(locations[nums_columns_name])

# @app.route('/')
# def test():
#     return {"test:":x_scaled.tolist()}

# @app.route('/test2')
# def test2():
#     return {"test2:!!!":'Test!!'}

# with open('app/columns_name_nums.pickle', 'rb') as f:
#     nums_columns_name = pickle.load(f)

# pipeline_nums.fit(X_train[nums_columns_name])