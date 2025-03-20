import cv2
import os
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

mp_hands = mp.solutions.hands

def process_video_with_mediapipe(video_path, label):
    cap = cv2.VideoCapture(video_path)
    frames = []
    labels = []
    
    with mp_hands.Hands(static_image_mode=True,  max_num_hands=1, min_detection_confidence=0.5) as hands:
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
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
                    
                    if hand_img.size > 0:
                        gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

                        resized = cv2.resize(thresh, (64, 64))
                        
                        normalized = resized / 255.0
                        
                        frames.append(normalized)
                        labels.append(label)
                        
                        if frame_count % 10 == 0:
                            debug_dir = f"debug_frames/class_{label}"
                            os.makedirs(debug_dir, exist_ok=True)
                            cv2.imwrite(f"{debug_dir}/frame_{os.path.basename(video_path)}_{frame_count}.jpg", resized)
            
            frame_count += 1
    
    cap.release()
    
    if not frames:  
        print(f"Warning: No hands detected in {video_path}")
        return [], []
        
    return np.array(frames), np.array(labels)

def prepare_dataset(folder, save_debug=True):
    
    X, y = [], []
    
    if save_debug:
        os.makedirs("debug_frames", exist_ok=True)
    
    class_counts = {}
    
    for file in os.listdir(folder):
        if file.endswith(".mp4"):
            label = int(file.split("_")[0]) - 1  # 0 indexed
            class_counts[label] = class_counts.get(label, 0) + 1
            
            video_path = os.path.join(folder, file)
            print(f"Processing {video_path}, label: {label+1}")
            
            frames, labels = process_video_with_mediapipe(video_path, label)
            
            if len(frames) > 0:
                X.extend(frames)
                y.extend(labels)
    
    print("Class distribution in dataset:")
    for label in sorted(class_counts.keys()):
        print(f"  Class {label+1}: {class_counts[label]} videos")
    
    print(f"Total extracted frames: {len(X)}")
    
    if not X:  
        raise Exception("No valid frames extracted from any videos!")
    
    X = np.array(X).reshape(-1, 64, 64, 1)  # Reshape for CNN input
    y = to_categorical(y, num_classes=5)  # One-hot encoding
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# https://www.tensorflow.org/api_docs/python/tf/keras/Model
def train_model(train_X, train_y, test_X, test_y, epochs=20, batch_size=32):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(5, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    history = model.fit(
        train_X, train_y, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data=(test_X, test_y),
        verbose=1
    )
    
    test_loss, test_acc = model.evaluate(test_X, test_y)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    model.save("finger_counter_model_improved.h5")
    print("Model saved as 'finger_counter_model_improved.h5'")
    
    return model, history

if __name__ == "__main__":
    train_X, test_X, train_y, test_y = prepare_dataset("dataset", save_debug=True)
    model, history = train_model(train_X, train_y, test_X, test_y, epochs=20)
    
    np.save("train_X_improved.npy", train_X)
    np.save("test_X_improved.npy", test_X)
    np.save("train_y_improved.npy", train_y)
    np.save("test_y_improved.npy", test_y)