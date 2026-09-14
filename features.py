import math


def extract_landmarks(hand_landmarks):
    """
    Transforme les 21 landmarks MediaPipe
    en 63 features normalisées.

    Le poignet (landmark 0) devient notre origine.
    """

    wrist = hand_landmarks[0]

    features = []

    # Coordonnées relatives au poignet
    for landmark in hand_landmarks:
        x = landmark.x - wrist.x
        y = landmark.y - wrist.y
        z = landmark.z - wrist.z

        features.extend([x, y, z])

    # Recherche de la plus grande distance par rapport au poignet
    max_distance = 0

    for i in range(0, len(features), 3):
        x = features[i]
        y = features[i + 1]
        z = features[i + 2]

        distance = math.sqrt(x**2 + y**2 + z**2)

        if distance > max_distance:
            max_distance = distance

    # Évite une division par zéro
    if max_distance == 0:
        return features

    # Normalisation de l'échelle
    features = [
        value / max_distance
        for value in features
    ]

    return features