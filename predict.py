import cv2
import joblib
import mediapipe as mp
import pandas as pd
import os

from features import extract_landmarks
from character_mapping import GESTURE_TO_CHARACTER


MODEL_FILE = "models/gesture_classifier_v4.pkl"
CHARACTER_DIR = "assets/characters"


# Chargement du modèle
model = joblib.load(MODEL_FILE)

print("Modèle chargé.")
print("Classes :", model.classes_)


# Chargement des images des personnages
character_images = {}

for gesture, character in GESTURE_TO_CHARACTER.items():

    image_path = os.path.join(
        CHARACTER_DIR,
        f"{character}.png"
    )

    image = cv2.imread(image_path)

    if image is None:
        print(f"Impossible de charger : {image_path}")
    else:
        character_images[gesture] = image
        print(f"Image chargée : {character}.png")


# Configuration MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
)


# Lancement du détecteur
with HandLandmarker.create_from_options(options) as landmarker:

    camera = cv2.VideoCapture(0)

    timestamp_ms = 0

    while True:

        success, frame = camera.read()

        if not success:
            print("Impossible de lire la webcam.")
            break


        # OpenCV utilise BGR, MediaPipe attend RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        timestamp_ms += 1


        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        # Dimensions de l'image
        height, width, _ = frame.shape


        # Dessin des articulations et connexions
        for hand_landmarks in result.hand_landmarks:

            # Points des articulations
            for landmark in hand_landmarks:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )


            # Connexions entre les articulations
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20),
                (5, 9), (9, 13), (13, 17)
            ]


            for start, end in connections:

                x1 = int(
                    hand_landmarks[start].x * width
                )

                y1 = int(
                    hand_landmarks[start].y * height
                )

                x2 = int(
                    hand_landmarks[end].x * width
                )

                y2 = int(
                    hand_landmarks[end].y * height
                )

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )


        # Reconnaissance du geste
        if result.hand_landmarks:

            features = extract_landmarks(
                result.hand_landmarks
            )


            features_df = pd.DataFrame(
                [features],
                columns=model.feature_names_in_
            )


            probabilities = model.predict_proba(
                features_df
            )[0]


            prediction_index = probabilities.argmax()


            prediction = model.classes_[
                prediction_index
            ]


            confidence = probabilities[
                prediction_index
            ]


            print(
                f"Geste détecté : {prediction} "
                f"({confidence:.2%})"
            )


            # Affichage du personnage correspondant
            if prediction in character_images:

                cv2.imshow(
                    "Hunter x Hunter - Character",
                    character_images[prediction]
                )


        else:

            # Aucune main détectée
            prediction = "no_hand"


            if prediction in character_images:

                cv2.imshow(
                    "Hunter x Hunter - Character",
                    character_images[prediction]
                )


        # Affichage de la webcam
        cv2.imshow(
            "Hunter x Hunter - Gesture Prediction",
            frame
        )


        # Q pour quitter
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    camera.release()
    cv2.destroyAllWindows()