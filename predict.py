import cv2
import joblib
import mediapipe as mp
import pandas as pd

from features import extract_landmarks


MODEL_FILE = "models/gesture_classifier_v2.pkl"


# Chargement de notre modèle de classification
model = joblib.load(MODEL_FILE)

print("Modèle chargé.")
print("Classes :", model.classes_)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
)

with HandLandmarker.create_from_options(options) as landmarker:

    camera = cv2.VideoCapture(0)

    timestamp_ms = 0

    while True:
        success, frame = camera.read()

        if not success:
            print("Impossible de lire la webcam.")
            break

        # OpenCV utilise BGR, MediaPipe attend RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms += 1

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]

            features = extract_landmarks(hand_landmarks)

            features_df = pd.DataFrame(
                [features],
                columns=model.feature_names_in_
            )

            probabilities = model.predict_proba(features_df)[0]

            prediction_index = probabilities.argmax()

            prediction = model.classes_[prediction_index]

            confidence = probabilities[prediction_index]

            print(
                f"Geste détecté : {prediction} "
                f"({confidence:.2%})"
            )

            cv2.putText(
                frame,
                f"Geste : {prediction}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Confiance : {confidence:.1%}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        # Nombre de mains détectées
        number_of_hands = len(result.hand_landmarks)

        cv2.putText(
            frame,
            f"Mains detectees : {number_of_hands}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "Hunter x Hunter - Gesture Prediction",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()