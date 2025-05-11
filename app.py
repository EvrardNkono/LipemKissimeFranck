import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Charger le modèle et le scaler
model = joblib.load("modele.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Prédiction du prix d'un logement à NYC 🗽")
st.write("Entrez les caractéristiques ci-dessous pour prédire le prix.")

# Interface utilisateur
col1, col2 = st.columns(2)

with col1:
    latitude = st.number_input("Latitude", value=40.7)
    minimum_nights = st.number_input("Nombre de nuits minimum", value=1)
    number_of_reviews = st.number_input("Nombre de commentaires", value=10)
    host_id = st.number_input("ID de l'hôte", value=100)  # Valeur par défaut
    calculated_host_listings_count = st.number_input("Nombre d'annonces de l'hôte", value=5)  # Valeur par défaut

with col2:
    longitude = st.number_input("Longitude", value=-73.9)
    reviews_per_month = st.number_input("Commentaires par mois", value=1.2)
    availability_365 = st.number_input("Disponibilité (jours/an)", value=180)
    id_value = st.number_input("ID du logement", value=1)  # Valeur par défaut

# Mettre les données dans un tableau
input_data = pd.DataFrame([[
    id_value,  # ID du logement
    host_id,  # ID de l'hôte
    latitude,
    longitude,
    minimum_nights,
    number_of_reviews,
    reviews_per_month,
    calculated_host_listings_count,  # Nombre d'annonces de l'hôte
    availability_365
]], columns=[
    'id',  # ID du logement
    'host_id',  # ID de l'hôte
    'latitude',
    'longitude',
    'minimum_nights',
    'number_of_reviews',
    'reviews_per_month',
    'calculated_host_listings_count',  # Nombre d'annonces de l'hôte
    'availability_365'
])

# Mise à l’échelle
input_scaled = scaler.transform(input_data)

# Prédiction
prediction = model.predict(input_scaled)[0]

# Affichage
st.subheader("Prix estimé :")
st.success(f"{prediction:.2f} $")
