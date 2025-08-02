# 📊 Student Success Prediction with Logistic Regression

This project demonstrates a simple machine learning approach to predict whether a student will succeed ("réussi") based on their study hours and previous grades. Using Python, pandas, and scikit-learn, the code builds, trains, and evaluates a logistic regression model on a small sample dataset.

---

## 🚀 Overview

The purpose of this project is to provide a clear and accessible example of binary classification using logistic regression. It can serve as an educational resource for beginners in data science and machine learning.

---

## 🛠️ How It Works

### 1. **Dataset Creation**
A small dataset of 10 students is created, capturing:
- `heures_revision`: Number of hours of study
- `note_precedente`: Previous grade
- `reussi`: Whether the student succeeded (1) or not (0)

### 2. **Model Building**
- **Features**: `heures_revision` and `note_precedente`
- **Target**: `reussi`
- The dataset is split into training and test sets (50% each).
- A logistic regression classifier is trained on the training set.

### 3. **Evaluation**
- Model accuracy is computed on the test set.
- A confusion matrix can be generated for deeper evaluation.

### 4. **Prediction**
- The model predicts the outcome for a new student with 2 hours of study and a previous grade of 10.
- The probability of success is also provided.

---

## 📄 Example Code

```python
import pandas as pd
data = pd.DataFrame({
    "heures_revision": [2, 1, 3, 4, 0, 1, 5, 3, 2, 0],
    "note_precedente": [8, 5, 7, 6, 3, 6, 9, 6, 5, 2],
    "reussi": [1, 0, 1, 1, 0, 0, 1, 1, 0, 0]
})

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

X = data[["heures_revision", "note_precedente"]]
y = data["reussi"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, shuffle=True)
model = LogisticRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Model accuracy: ", accuracy_score(y_test, y_pred))

# Predict for a new student
netudiant = [[2, 10]]
prediction = model.predict(netudiant)
proba = model.predict_proba(netudiant)

print("Predicted success:", "Yes" if prediction[0] == 1 else "No")
print("Success probability:", round(proba[0][1]*100, 2), "%")
```

---

## 📈 Results

- The model computes and displays its accuracy on the test set.
- It outputs whether a student with 2 hours of revision and a previous grade of 10 is predicted to succeed, along with the probability of success.

---

## 📝 Conclusion

The project demonstrated that machine learning models can effectively predict students' academic performance based on features such as study hours and previous grades. The results highlight the importance of data-driven approaches in education, enabling early interventions and personalized learning strategies to enhance student outcomes.

---

## 👤 Author

- **Marouan El Yassini**

---

## 📃 License

This project is licensed under the MIT License.
