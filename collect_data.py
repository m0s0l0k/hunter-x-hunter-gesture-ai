import cv2
import csv
import os
import mediapipe as mp

from features import extract_landmarks


DATASET_FILE = "data/gestures_v3.csv"


LABELS = {
    ord("1"): "fist",
    ord("2"): "open_hand",
    ord("3"): "v_sign",
    ord("4"): "pray_one_hand",
    ord("5"): "heart_two_hand",
    ord("6"): "open_hand_pointing",
    ord("7"): "killua_hand",
    ord("8"): "kuroro_bouche",
}


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


# Vérifie si le dataset existe déjà
file_exists = os.path.exists(DATASET_FILE)


# Création du fichier et de son en-tête
with open(DATASET_FILE, "a", newline="") as file:

    writer = csv.writer(file)

    if not file_exists:

        header = ["label"]

        for i in range(126):
            header.append(f"feature_{i}")

        writer.writerow(header)


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


        # Nombre de mains détectées
        number_of_hands = len(
            result.hand_landmarks
        )


        # Dessine les landmarks
        for hand_landmarks in result.hand_landmarks:

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


        # Collecte des données
        if result.hand_landmarks:

            features = extract_landmarks(
                result.hand_landmarks
            )


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


        # Interface
        cv2.putText(
            frame,
            "1:Fist  2:Open  3:V  4:Pray  5:Heart  6:Point  7:Killua  8:Kuroro",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Geste : {current_label}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Mains detectees : {number_of_hands}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Echantillons : {samples_collected}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        cv2.imshow(
            "Hunter x Hunter - Dataset Collector",
            frame
        )


        key = cv2.waitKey(1) & 0xFF


        # Sélection du geste
        if key in LABELS:

            current_label = LABELS[key]

            print(
                f"Collecte activee : {current_label}"
            )


        # Q pour quitter
        elif key == ord("q"):

            break


    camera.release()
    cv2.destroyAllWindows()


print()
print("Collecte terminee.")
print(f"Dataset : {DATASET_FILE}")
print(f"Echantillons : {samples_collected}")