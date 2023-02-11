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