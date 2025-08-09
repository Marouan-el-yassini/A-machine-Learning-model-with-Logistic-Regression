# Logistic Regression Examples with Scikit-learn

This repository contains two simple machine learning examples using logistic regression with Python and scikit-learn. Both examples rely on small, hardcoded datasets and illustrate basic usage of logistic regression for binary classification.

---

## Example 1: Email Spam Detection

This script demonstrates how to use logistic regression to classify emails as "Spam" or "Non Spam" based on several features:

- **gagner**: Indicator variable (possibly for "win" in the email)
- **maj**: Number of uppercase words
- **ponctuation**: Number of punctuation marks
- **longueur**: Email length
- **liens**: Number of links
- **spam**: Target variable (1 = spam, 0 = not spam)

### Code

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Data
data = pd.DataFrame({
    "gagner": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    "maj": [10, 1, 8, 0, 15, 2, 12, 1, 9, 0],
    "ponctuation": [35, 5, 40, 10, 50, 15, 45, 8, 38, 6],
    "longueur": [120, 300, 90, 280, 70, 200, 85, 220, 110, 250],
    "liens": [3, 0, 2, 0, 4, 1, 3, 0, 2, 0],
    "spam": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
})

# Split data
X = data[["gagner", "maj", "ponctuation", "longueur", "liens"]]
y = data["spam"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
print("Précision du modèle :", accuracy_score(y_test, y_pred))

# Predict a new email
email = [[1, 2, 55, 80, 1]]  # Correct format
prediction = model.predict(email)
print("Email détecté comme :", "Spam" if prediction[0] == 1 else "Non Spam")
```

---

## Example 2: Student Success Prediction

This script predicts whether a student will succeed (pass/fail) based on study hours and previous grades.

- **heures_revision**: Study hours
- **note_precedente**: Previous grade
- **reussi**: Target variable (1 = success, 0 = failure)

### Code

```python
import pandas as pd
data = pd.DataFrame({
    "heures_revision": [2, 1, 3, 4, 0, 1, 5, 3, 2, 0],
    "note_precedente": [8, 5, 7, 6, 3, 6, 9, 6, 5, 2],
    "reussi": [1, 0, 1, 1, 0, 0, 1, 1, 0, 0]
})

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = data[["heures_revision", "note_precedente"]]
y = data["reussi"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, shuffle=True)
model = LogisticRegression()
model.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix

y_pred = model.predict(X_test)
print("presiscion de model ", accuracy_score(y_test, y_pred))

netudiant = [[2, 10]]
prediction = model.predict(netudiant)
proba = model.predict_proba(netudiant)

print("Réussite prédite :", "Oui" if prediction[0] == 1 else "Non")
print("Probabilité de réussite :", round(proba[0][1]*100, 2), "%")
```

---

## Requirements

- Python 3.x
- pandas
- scikit-learn

Install dependencies with:

```bash
pip install pandas scikit-learn
```

---

## Usage

1. Copy either code block into a `.py` file.
2. Run with Python:

```bash
python your_script.py
```

---

## License

This project is provided for educational purposes.
