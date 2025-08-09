import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Données
data = pd.DataFrame({
    "gagner": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    "maj": [10, 1, 8, 0, 15, 2, 12, 1, 9, 0],
    "ponctuation": [35, 5, 40, 10, 50, 15, 45, 8, 38, 6],
    "longueur": [120, 300, 90, 280, 70, 200, 85, 220, 110, 250],
    "liens": [3, 0, 2, 0, 4, 1, 3, 0, 2, 0],
    "spam": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
})

# Séparation des données
X = data[["gagner", "maj", "ponctuation", "longueur", "liens"]]
y = data["spam"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Modèle
model = LogisticRegression()
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)
print("Précision du modèle :", accuracy_score(y_test, y_pred))

# Prédire un nouvel e-mail
email = [[1, 2, 55, 80, 1]]  # Format correct
prediction = model.predict(email)
print("Email détecté comme :", "Spam" if prediction[0] == 1 else "Non Spam")
