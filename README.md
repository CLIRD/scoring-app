# OpenClassRooms-Projet7-Scoring-App

L'entreprise "Prêt à dépenser" propose des crédits à la consommation à des personnes ayant peu ou pas d'historique de prêt. Pour évaluer la probabilité de remboursement de chaque client et ainsi classer les demandes en crédits accordés ou refusés, l'entreprise souhaite développer un algorithme de classification.
Pour cela, elle se basera sur des sources de données variées, telles que les données comportementales et celles provenant d'autres institutions financières.
Cependant, les clients sont de plus en plus demandeurs de transparence en ce qui concerne les décisions d'octroi de crédit. Pour répondre à cette demande, l'entreprise a décidé de développer un dashboard interactif. Ce tableau de bord offrira une transparence maximale aux clients et permettra aux chargés de relation client d'expliquer facilement les décisions d'octroi de crédit.
Le jeu de données fourni comprend 300 000 clients avec plus de 300 variables, mais il présente un déséquilibre important de classes, avec 90 % de clients qui remboursent leur prêt et 10 % de clients qui ne le remboursent pas.

Le dashboard est disponible à l'adresse suivante: https://clird-scoring-app-app-oupinc.streamlit.app/

Le repository contient les éléments suivants:

- github/workflows: Déploiement en continu
- app/columns: Contient les noms des colonnes
- app/models: Contient les modèles entrainés et pipeline
- data: Contient les données nécessaires pour le fonctionnement du dashboard
- image: Contient les images pour le dashboard
- model: Contient les notebooks allant du prétraitement des données, à la création du modèle final
- tests: Contient les tests unitaires
- app.py: Dashoard
- data_dashboard.ipynb: Notebook pour la normalisation des données nécessaires pour le dashboard
- style.css: Feuille de style pour le dashboard
- requirements.txt : liste des librairies python requises pour le fonctionnement des programmes
