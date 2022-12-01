import altair as alt
import pandas as pd
import streamlit as st

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wget
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Classification titanic", page_icon="⬇", layout="centered"
)

# $ Titre de l'app
st.title("Classification binaire du titanic")

# Texte
st.write("Quelques visualisations des données")


#### IMPORTATION DES DONNÉES #####
wget.download(
    "https://raw.githubusercontent.com/iid-ulaval/EEAA-datasets/master/titanic_train.csv",
    "./titanic_train.csv",
)
wget.download(
    "https://raw.githubusercontent.com/iid-ulaval/EEAA-datasets/master/titanic_test.csv",
    "./titanic_test.csv",
)
train_data = pd.read_csv("titanic_train.csv")
test_data = pd.read_csv("titanic_test.csv")

#### TEST VIZ ######
# st.dataframe(train_data.head(20))

# fig = plt.figure(figsize=(10, 4))
# sns.barplot(x="Pclass", y="Survived", data=train_data)
# st.pyplot(fig)
#### TEST VIZ ######


# Traitement valeur manquantes
train_data = train_data.dropna()

# Traitement de la variable Sexe
train_data["Sex"] = train_data["Sex"].replace("male", 1)
train_data["Sex"] = train_data["Sex"].replace("female", 0)

# Ici on sépare nos données X (variables prédictives) et y (variables à prédire)
X = train_data[
    [
        "Sex",
        "Age",
    ]
]  # variables prédictives (indépendantes)
y = train_data["Survived"]  # Variable à prédire (dépendantes)

model = LogisticRegression()  # Importe l'algorithme
model.fit(X, y)

AGE = st.slider("Age de la personne?", 0, 2, 65)
SEX = st.radio("Sexe de la personne", ("Homme", "Femme"))

st.write("Cette personne avait ", AGE, "ans et", "était un/une", SEX)

if SEX == "Homme":
    SEX = 1
else:
    SEX = 0


pred = model.predict(
    [[SEX, AGE]]
)  # On prédit les données de validation (20%) pour tester le modèle

if pred == 0:
    pred = "mort"
else:
    pred = "survie"

st.metric("prediction", pred)

st.write(
    "See more in our public [GitHub"
    " repository](https://github.com/streamlit/example-app-time-series-annotation)"
)
