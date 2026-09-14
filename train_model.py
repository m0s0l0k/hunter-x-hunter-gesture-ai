import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ---------------------------------
# Chargement du dataset
# ---------------------------------

dataset = pd.read_csv("data/gestures.csv")


# ---------------------------------
# Séparation features / labels
# ---------------------------------

X = dataset.drop("label", axis=1)
y = dataset["label"]


print(f"Nombre d'exemples : {len(dataset)}")
print(f"Nombre de features : {X.shape[1]}")
print()
print("Répartition des classes :")
print(y.value_counts())


# ---------------------------------
# Séparation entraînement / test
# ---------------------------------

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


# ---------------------------------
# Création du modèle
# ---------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ---------------------------------
# Entraînement
# ---------------------------------

print()
print("Entraînement du modèle...")

model.fit(X_train, y_train)

print("Entraînement terminé.")

joblib.dump(
    model,
    "models/gesture_classifier_v2.pkl"
)

print(
    "Modèle sauvegardé dans "
    "models/gesture_classifier_v2.pkl"
)


# ---------------------------------
# Évaluation
# ---------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)


print()
print(f"Accuracy : {accuracy:.2%}")

print()
print("Rapport de classification :")

print(
    classification_report(
        y_test,
        predictions
    )
)

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