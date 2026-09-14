import cv2
import mediapipe as mp
from features import extract_landmarks


# -----------------------------
# Initialisation de MediaPipe
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
    num_hands=2,
)


# -----------------------------
# Détection des mains
# -----------------------------

with HandLandmarker.create_from_options(options) as landmarker:

    camera = cv2.VideoCapture(0)

    timestamp_ms = 0

    while True:
        success, frame = camera.read()

        if not success:
            print("Impossible de lire la webcam.")
            break

        # OpenCV utilise BGR.
        # MediaPipe attend une image RGB.
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

        # -----------------------------
        # Affichage des landmarks
        # -----------------------------

        for hand_landmarks in result.hand_landmarks:

            features = extract_landmarks(hand_landmarks)

            print(f"Nombre de features : {len(features)}")
            print(features)

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
            "Hunter x Hunter - Hand Tracking",
            frame
        )

        # Q pour quitter
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()