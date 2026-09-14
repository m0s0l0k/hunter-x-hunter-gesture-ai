import cv2
import csv
import time
import mediapipe as mp

from features import extract_landmarks


# -----------------------------
# Configuration
# -----------------------------

DATASET_FILE = "data/gestures.csv"

LABELS = {
    ord("1"): "fist",
    ord("2"): "open_hand",
    ord("3"): "v_sign",
    ord("4"): "unknown",
}

# -----------------------------
# Initialisation MediaPipe
# -----------------------------

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


# -----------------------------
# Création du fichier CSV
# -----------------------------

import os

file_exists = os.path.exists(DATASET_FILE)

with open(DATASET_FILE, "a", newline="") as file:
    writer = csv.writer(file)

    if not file_exists:
        header = ["label"]

        for i in range(63):
            header.append(f"feature_{i}")

        writer.writerow(header)


# -----------------------------
# Capture des données
# -----------------------------

with HandLandmarker.create_from_options(options) as landmarker:

    camera = cv2.VideoCapture(0)

    timestamp_ms = 0
    current_label = None
    samples_collected = 0

    while True:

        success, frame = camera.read()

        if not success:
            print("Impossible de lire la webcam.")
            break

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

        # -----------------------------
        # Récupération des landmarks
        # -----------------------------

        if result.hand_landmarks:

            hand_landmarks = result.hand_landmarks[0]

            features = extract_landmarks(
                hand_landmarks
            )

            # Dessin des landmarks

            height, width, _ = frame.shape

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

            # -----------------------------
            # Enregistrement
            # -----------------------------

            if current_label is not None:

                with open(
                    DATASET_FILE,
                    "a",
                    newline=""
                ) as file:

                    writer = csv.writer(file)

                    writer.writerow(
                        [current_label] + features
                    )

                samples_collected += 1

        # -----------------------------
        # Interface
        # -----------------------------

        cv2.putText(
            frame,
            "1: Poing | 2: Main ouverte | 3: V",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Geste: {current_label}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Echantillons: {samples_collected}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "Hunter x Hunter - Dataset Collector",
            frame
        )

        # -----------------------------
        # Gestion clavier
        # -----------------------------

        key = cv2.waitKey(1) & 0xFF

        if key in LABELS:

            current_label = LABELS[key]

            print(
                f"Collecte activee : {current_label}"
            )

        elif key == ord("q"):

            break

    camera.release()
    cv2.destroyAllWindows()


print()
print("Collecte terminee.")
print(f"Dataset : {DATASET_FILE}")
print(f"Echantillons : {samples_collected}")