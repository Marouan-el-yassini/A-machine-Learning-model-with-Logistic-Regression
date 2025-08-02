import pandas as pd
data=pd.DataFrame({
    "heures_revision": [2, 1, 3, 4, 0, 1, 5, 3, 2, 0],
    "note_precedente": [8, 5, 7, 6, 3, 6, 9, 6, 5, 2],
    "reussi": [1, 0, 1, 1, 0, 0, 1, 1, 0, 0]
})
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X=data[["heures_revision","note_precedente"]]
y=data["reussi"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5,random_state=1,shuffle=True)
model=LogisticRegression()

model.fit(X_train,y_train)

from sklearn.metrics import confusion_matrix

y_pred=model.predict(X_test)



print("presiscion de model ",accuracy_score(y_test,y_pred))

netudiant=[[2,10]]
prediction=model.predict(netudiant)
proba=model.predict_proba(netudiant)

print("Réussite prédite :", "Oui" if prediction[0] == 1 else "Non")
print("Probabilité de réussite :", round(proba[0][1]*100, 2), "%")
