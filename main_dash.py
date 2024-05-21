"""Création d'un script python permettant de déployer un modèle de machine learning via une API Flask. 
L'API permet de prédire la probabilité de défaut de paiement d'un client, puis en fonction d'un seuil établi lors de la modélisation,
de déterminer si le client est à risque ou non. 
L'API fournit également des informations (âge, sexe, statut marital, etc.) sur le client et des graphiques de feature importance globale et locale. 
L'API est déployée localement et peut être accédée via un navigateur web. 
Le modèle de machine learning utilisé est un modèle LightGBM calibré."""


# import des packages
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from io import BytesIO
from flask import Flask, request, jsonify
import re
import shap
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

### Préparation des données :

# Fonction qui charge des données :
def load_data(data_path):
    data = pd.read_csv(data_path)
    # On doit changer le nom des colonnes car LightLGBM ne supporte pas certains caractères :
    new_names = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in data.columns}
    new_n_list = list(new_names.values())
    new_names = {col: f'{new_col}_{i}' if new_col in new_n_list[:i] else new_col for i, (col, new_col) in enumerate(new_names.items())}
    data = data.rename(columns=new_names)
    data.set_index('SK_ID_CURR', inplace=True)
    if 'TARGET' in data.columns:
        data.drop('TARGET', axis = 1, inplace=True)
    return data

# Fonction qui charge le scaler, le modèle et l'explainer SHAP créés lors de la modélisation:
def load_scaler_model_explainer():
    model_path = "./data/calibrated_lgbm_v2.pkl"
    with open(model_path, 'rb') as f_in:
        model = pickle.load(f_in)

    scaler_path = "./data/scaler_v2.pkl"
    with open(scaler_path, 'rb') as f_in:
        scaler = pickle.load(f_in)

    explainer_path = "./data/shap_explainer_v2.pkl"
    with open(explainer_path, 'rb') as f_in:
        explainer = pickle.load(f_in)

    return scaler, model, explainer

# Fonction qui scale les données :
def prepare_data(data, scaler):
    features = scaler.transform(data)
    features = pd.DataFrame(features, columns=data.columns, index=data.index)
    return features

# Fonction qui renvoie une liste des identifiants clients :
def get_clients_ids(features):
    clients_ids = features.index.to_list()
    return clients_ids

# Fonction qui crée le dataframe des données brutes (nécessaire pour obtenir les informations clients) :
def load_brut_data(path):
    df_brut = pd.read_csv(path)
    return df_brut

# Fonction qui renvoie les informations d'un client :
def get_client_infos(client_id, path):
    df_brut = load_brut_data(path)
    client = df_brut[df_brut['SK_ID_CURR'] == client_id].drop(['SK_ID_CURR'], axis=1)
    gender = client['CODE_GENDER'].values[0]
    age = client['DAYS_BIRTH'].values[0]
    age = int(np.abs(age) // 365)
    revenu = float(client['AMT_INCOME_TOTAL'].values[0])
    source_revenu = client['NAME_INCOME_TYPE'].values[0]
    montant_credit = float(client['AMT_CREDIT'].values[0])
    statut_famille = client['NAME_FAMILY_STATUS'].values[0]
    education = client['NAME_EDUCATION_TYPE'].values[0]
    ratio_revenu_credit = round((revenu / montant_credit) * 100, 2)
    dict_infos = {
        'sexe' : gender,
        'âge' : age,
        'revenu' : revenu,
        'source_revenu' : source_revenu,
        'montant_credit' : montant_credit,
        'ratio_revenu_credit' : ratio_revenu_credit,
        'statut_famille' : statut_famille,
        'education' : education
    }
    return dict_infos

# Chargement des données, du scaler, du modèle, de la liste des IDs clients valides et de l'explainer SHAP :

df = load_data("./data/subset_train.csv")
scaler, model, explainer = load_scaler_model_explainer()
features = prepare_data(df, scaler)
clients_ids = get_clients_ids(df)
threshold = 0.377 # Déterminé lors de la modélisation
# shap :
shap_values = explainer.shap_values(features)

### Prédiction :

# On instancie l'API Flask :
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = True

# On configure différentes routes pour l'API :

# Route d'"accueil" :

@app.route("/")
def welcome():
    return ("Bienvenue sur l'API de prédiction de défaut de paiement !\n\n")

# Route qui affiche la liste des IDs clients valides :

@app.route('/list_ids', methods=['GET'])
def print_id_list():
    return f'Liste des id clients valides :\n\n{(clients_ids)}'

# Route qui affiche les informations d'un client et sa probabilité de défaut de paiement :
@app.route('/prediction/<int:client_id>', methods=['GET'])
def prediction(client_id):
    if client_id in clients_ids:
        client_data = features.loc[client_id].values.reshape(1, -1)
        proba = model.predict_proba(client_data)[0, 1]

        client_infos = get_client_infos(client_id, "./data/subset_train_brut.csv")

        customer_pred = {
            'id': client_id,
            'probabilité_défaut': proba.round(2),
            'statut': 'Crédit accepté' if proba <= threshold else 'Crédit refusé',
            'client_infos' : client_infos
        }

        return jsonify(customer_pred)
    else:
        return 'Client_id non valide.'

# Route qui affiche la feature importance globale via un summary plot shap :
@app.route('/global_shap', methods=['GET'])
def global_shap():
    shap.summary_plot(shap_values[1], 
                      features=features.values, 
                      feature_names=features.columns,
                      plot_type='violin',
                      max_display=10, 
                      show=False)
    
    # Création d'un objet BytesIO pour stocker l'image
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    plt.close()
    
    # Conversion de l'image en base64
    encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Affichage de l'image directement dans le navigateur
    return f'<img src="data:image/png;base64,{encoded_string}">'

# Route qui affiche la feature importance locale pour le client sélectionné :
@app.route('/local_shap/<int:client_id>', methods=['GET'])
def local_shap(client_id):
    if client_id in clients_ids:  # On s'assure ici que client_id est valide
        client_index = features.index.get_loc(client_id)  # On récupère l'index du client dans le DataFrame features
        
        # On récupère les valeurs SHAP spécifiques au client :
        client_shap_values = shap_values[1][client_index]
        
        # On crée une explication SHAP pour le client :
        exp = shap.Explanation(values = client_shap_values, 
                               base_values = explainer.expected_value[1], 
                               data = features.iloc[client_index], 
                               feature_names=features.columns)
        plt.figure()
        # Création du waterfall plot
        shap.plots.waterfall(exp, show=False)

        # Ajustement de la taille et des marges de la figure:
        plt.gcf().set_size_inches(10, 6)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # Ajustez les marges en fonction de vos besoins
        plt.tight_layout()

        # Création d'un objet BytesIO pour stocker l'image
        buffer_1 = BytesIO()
        plt.savefig(buffer_1, format='png')
        buffer_1.seek(0)

        plt.close()
        
        # Conversion de l'image en base64 :
        encoded_string = base64.b64encode(buffer_1.getvalue()).decode('utf-8')
        
        return f'<img src="data:image/png;base64,{encoded_string}">'
    else:
        return 'ID Client non valide.'

# Route qui permet d'accéder aux données clients transformées :
@app.route('/client_data', methods=['GET'])
def get_client_data():
    df_reset = df.reset_index()
    return df_reset.to_json(orient='records')

# Route qui permet d'accéder aux données brutes :
@app.route('/client_raw_data', methods=['GET'])
def get_client_raw_data():
    df_brut = load_brut_data("./data/subset_train_brut.csv")
    df_brut['DAYS_BIRTH'] = df_brut['DAYS_BIRTH'].apply(lambda x: int(np.abs(x) // 365))
    return df_brut.to_json(orient='records')

@app.route('/scaled_data', methods=['GET'])
def get_scaled_data():
    return features.to_json(orient='records')
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7676)