import cv2
import mediapipe as mp
import numpy as np
import pickle
from tensorflow.keras.models import load_model

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

# ==========================
# Load model & label map A-Z
# ==========================
MODEL_PATH = 'gesture_model_az.h5'      # model hasil training neutral + A-Z
LABEL_PATH = 'gesture_labels_az.pkl'    # file label_map terbaru

model = load_model(MODEL_PATH)
with open(LABEL_PATH, 'rb') as f:
    # label_map: {0:'neutral', 1:'A', ..., 26:'Z'}
    labels = pickle.load(f)

no_of_timesteps = 20
seq_buffer = []

CONF_THRESHOLD = 0.50   # threshold 0.5

def extract_63(res):
    c = []
    if res.multi_hand_landmarks:
        hlm = res.multi_hand_landmarks[0]
        for lm in hlm.landmark:
            c.extend([lm.x, lm.y, lm.z])
    return c + [0] * (63 - len(c))

CAMERA_INDEX = 0
cap = cv2.VideoCapture(CAMERA_INDEX)

print("="*60)
print("🧪 TESTING GESTURE MODEL (neutral + A-Z)")
print("="*60)
print("Lakukan gesture sesuai dataset (neutral, A, B, ..., Z)")
print(f"Prediction dianggap AKURAT jika confidence > {CONF_THRESHOLD:.2f}")
print("Q = quit")
print("="*60)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = hands.process(rgb)

        # -------- ekstrak fitur 63 dimensi --------
        feat = extract_63(res)
        if sum(feat) != 0:
            seq_buffer.append(feat)
            if len(seq_buffer) > no_of_timesteps:
                seq_buffer.pop(0)

        gesture_name = "None"
        conf = 0.0

        # -------- prediksi jika buffer penuh --------
        if len(seq_buffer) == no_of_timesteps:
            X = np.array(seq_buffer).reshape(1, no_of_timesteps, -1)
            prob = model.predict(X, verbose=0)[0]
            cls  = int(np.argmax(prob))
            conf = float(np.max(prob))

            # labels = dict {id: label}
            if isinstance(labels, dict):
                gesture_name = labels.get(cls, str(cls))
            else:
                gesture_name = labels[cls]

        # -------- gambar landmark --------
        if res.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                res.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS
            )

        # -------- tampilan teks --------
        color = (0, 255, 0) if conf > CONF_THRESHOLD else (0, 165, 255)
        cv2.putText(
            frame,
            f'Gesture: {gesture_name} ({conf:.2f})',
            (20, 50),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0,
            color,
            2
        )

        if conf > CONF_THRESHOLD:
            cv2.putText(
                frame,
                "AKURAT!",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        cv2.imshow('Gesture Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("\n✅ Test selesai (model neutral + A-Z)")
