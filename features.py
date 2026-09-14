import math


def normalize_landmarks(hand_landmarks):
    """
    Transforme les 21 landmarks d'une main
    en 63 features normalisées.
    """

    wrist = hand_landmarks[0]

    features = []

    for landmark in hand_landmarks:
        x = landmark.x - wrist.x
        y = landmark.y - wrist.y
        z = landmark.z - wrist.z

        features.extend([x, y, z])

    max_distance = 0

    for i in range(0, len(features), 3):

        x = features[i]
        y = features[i + 1]
        z = features[i + 2]

        distance = math.sqrt(
            x**2 + y**2 + z**2
        )

        if distance > max_distance:
            max_distance = distance

    if max_distance == 0:
        return features

    features = [
        value / max_distance
        for value in features
    ]

    return features


def extract_landmarks(hand_landmarks_list):
    """
    Transforme une ou deux mains MediaPipe
    en 126 features.

    - 1 main  -> 63 features + 63 zéros
    - 2 mains -> 63 + 63 = 126 features
    """

    features = []

    # Première main
    if len(hand_landmarks_list) >= 1:
        features.extend(
            normalize_landmarks(
                hand_landmarks_list[0]
            )
        )
    else:
        features.extend([0.0] * 63)

    # Deuxième main
    if len(hand_landmarks_list) >= 2:
        features.extend(
            normalize_landmarks(
                hand_landmarks_list[1]
            )
        )
    else:
        features.extend([0.0] * 63)

    return features