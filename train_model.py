import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import joblib


DATASET_FILE = "data/gestures_v3.csv"
MODEL_FILE = "models/gesture_classifier_v4.pkl"


# Chargement du dataset
dataset = pd.read_csv(DATASET_FILE)


X = dataset.drop("label", axis=1)
y = dataset["label"]


print(f"Nombre d'exemples : {len(dataset)}")
print(f"Nombre de features : {X.shape[1]}")

print()
print("Répartition des classes :")
print(y.value_counts())


# Séparation entraînement / test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print()
print(f"Exemples entraînement : {len(X_train)}")
print(f"Exemples test : {len(X_test)}")


# Création du modèle
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


print()
print("Entraînement du modèle...")


# Entraînement
model.fit(
    X_train,
    y_train
)


print("Entraînement terminé.")


# Sauvegarde du modèle
joblib.dump(
    model,
    MODEL_FILE
)


print(
    f"Modèle sauvegardé dans {MODEL_FILE}"
)


# Prédictions sur le jeu de test
predictions = model.predict(X_test)


# Accuracy
accuracy = accuracy_score(
    y_test,
    predictions
)


print()
print(f"Accuracy : {accuracy:.2%}")


# Rapport de classification
print()
print("Rapport de classification:")

print(
    classification_report(
        y_test,
        predictions
    )
)


# Matrice de confusion
print()
print("Matrice de confusion :")

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=model.classes_
)

print(matrix)


print()
print("Ordre des classes :")
print(model.classes_)