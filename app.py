import streamlit as st 
import pandas as pd
import requests
import json
import plotly.graph_objects as go
import shap 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import pickle

data_vis = pd.read_csv('data/data.csv')
data = pd.read_csv('data/data_prod.csv')
with open('columns_name.pickle', 'rb') as f:
    columns_name = pickle.load(f)


with open('columns_name_nums.pickle', 'rb') as f:
    nums_columns_name = pickle.load(f)


st.sidebar.image('image\pret_a_depenser.png')

st.sidebar.markdown("<h6 style='text-align: center;'>Dashboard - Aide à la décision</h6>", unsafe_allow_html=True)
inf = st.sidebar.radio("Informations sur: ", ('Le modèle', 'La prédiction', 'Le client')) 

if inf == "Le modèle":
    st.title("Compréhension global du modèle")
    st.markdown("<h3 style='text-align: center;'>Importance de chaque indicateur</h3>", unsafe_allow_html=True)
    st.image('image\shap_overall.png')

    st.markdown("<h3 style='text-align: center;'>Relations positives et négatives de chaque indicateur</h3>", unsafe_allow_html=True)
    st.image('image\shap_beeswarm.png')
elif inf == "La prédiction":
    st.markdown("<h1 style='text-align: center;'>Prédiction</h1>", unsafe_allow_html=True)

    CUSTOMER_ID = st.selectbox(
    "Choisissez l'identifiant d'un client",
    data.SK_ID_CURR)
    
    col1, col2 = st.columns([1, 4])
    # Partie proba
    response = requests.get('http://127.0.0.1:5000/predict', json = data.query(f'SK_ID_CURR == {CUSTOMER_ID}').to_json())
    response_dict = json.loads(response.text)
    class_prediction = response_dict['Class']
    class_proba = round(response_dict['Class probabilities'],2)
    with col1:
        st.metric(label="Probabilité", value= 1 - class_proba)
        with open('style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    with col2:
        if class_proba >= 0.271906: 
            st.error('Attention ! Le demandeur a un risque élevé de ne pas rembourser le prêt !') 
        else: 
            st.success('Le demandeur a une forte probabilité de rembourser le prêt !')
        # Partie shapley

        response_shapley = requests.get('http://127.0.0.1:5000/api/shap', json = data.query(f'SK_ID_CURR == {CUSTOMER_ID}').to_json())
        response_dict_shapley = json.loads(response_shapley.text)
        shapley_values = np.array(response_dict_shapley['shapley_values'])
        shapley_base_values = np.array(response_dict_shapley['shapley_base_values'])
        shapley_data = np.array(response_dict_shapley['shapley_data'])
        st.subheader('Interprétabilité des résultats - Pour le demandeur')
        number_feature = st.slider("Nombre de caractéristique à afficher", 0, 158, 10)
        explainer = shap.Explanation(shapley_values, shapley_base_values, shapley_data, feature_names= columns_name)
        st.pyplot(shap.waterfall_plot(explainer[0], show = False, max_display = number_feature))
else: 
    st.title("Comparative d'un client à l'ensemble des clients")
    CUSTOMER_ID = st.selectbox(
    "Choisissez l'identifiant d'un client",
    data.SK_ID_CURR)

    options = st.multiselect(
    'Selectionner un ou plusieurs indicateurs',
    nums_columns_name.tolist(),
    ['AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE'])
    sns.set(style="darkgrid")

    fig, ax = plt.subplots()
    sns.boxplot(data_vis[options], ax = ax, flierprops={"marker": "x"}, color='skyblue', showcaps=True)
    client_data = requests.get('http://127.0.0.1:5000/transform_nums', json = data.query(f'SK_ID_CURR == {CUSTOMER_ID}').to_json())
    client_data = json.loads(client_data.text)
    client_data = pd.read_json(client_data['data'])
    for k, i in enumerate(options):
        ax.scatter(k, client_data[i].values, marker='X', s=100, color = 'black', label = 'Client selectionné')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    st.pyplot(fig)

    st.title("Descriptives relatives à un client")
    
    options2 = st.multiselect(
    'Selectionner un ou plusieurs indicateurs',
    data.columns,
    ['AMT_INCOME_TOTAL'])

    st.dataframe(data.query(f'SK_ID_CURR == {CUSTOMER_ID}')[options2])


