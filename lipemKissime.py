import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Charger les données
df = pd.read_csv(r'C:\Users\WORKSTATION\Downloads\Telegram Desktop\nyc_air_bnb.csv')

# 2. Nettoyage simple
# On enlève les colonnes non numériques ou inutiles
colonnes_a_supprimer = ['name', 'host_name', 'last_review', 'neighbourhood_group', 'neighbourhood', 'room_type']
df = df.drop(columns=colonnes_a_supprimer)

# 3. Remplacer les valeurs manquantes
df = df.dropna()

X = df.drop('price', axis=1)
y = df['price']

print("Colonnes utilisées pour l'entraînement :", X.columns)
# 4. Définir les features (X) et la target (y)
X = df.drop('price', axis=1)
y = df['price']

# 5. Diviser en jeu d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Appliquer le scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Entraîner un modèle simple (régression linéaire)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 8. Évaluer le modèle
score = model.score(X_test_scaled, y_test)
print(f'R² du modèle : {score:.2f}')

# 9. Sauvegarder le modèle et le scaler
joblib.dump(model, 'modele.pkl')
joblib.dump(scaler, 'scaler.pkl')
