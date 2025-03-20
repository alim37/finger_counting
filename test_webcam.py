import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

model = tf.keras.models.load_model("finger_counter_model_improved.h5")

# https://mediapipe.readthedocs.io/en/latest/solutions/hands.html
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def preprocess_hand_image(hand_img):
    if hand_img.size == 0:
        return None
    gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    resized = cv2.resize(thresh, (64, 64))
    processed = resized / 255.0
    processed = np.expand_dims(processed, axis=-1)  
    processed = np.expand_dims(processed, axis=0)     
    return processed

cap = cv2.VideoCapture(0)
prediction_history = []

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                h, w, _ = frame.shape
                landmarks_array = np.array([[lm.x * w, lm.y * h] for lm in hand_landmarks.landmark])
                x_min, y_min = np.min(landmarks_array, axis=0).astype(int)
                x_max, y_max = np.max(landmarks_array, axis=0).astype(int)
                padding_x = int((x_max - x_min) * 0.25)
                padding_y = int((y_max - y_min) * 0.25)
                x_min = max(0, x_min - padding_x)
                y_min = max(0, y_min - padding_y)
                x_max = min(w, x_max + padding_x)
                y_max = min(h, y_max + padding_y)

                hand_img = frame[y_min:y_max, x_min:x_max]
                processed = preprocess_hand_image(hand_img)
                if processed is not None:

                    prediction = model.predict(processed, verbose=0)
                    predicted_class = np.argmax(prediction)
                    confidence = prediction[0][predicted_class]

                    prediction_history.append(predicted_class)
                    if len(prediction_history) > 5:
                        prediction_history.pop(0)
                    smoothed_prediction = max(set(prediction_history), key=prediction_history.count)

                    # 0-indexing, add 1.
                    model_finger_count = smoothed_prediction + 1

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(frame, f"Model: {model_finger_count}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Finger Counter', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
