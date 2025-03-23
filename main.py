import cv2
import os
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

mp_hands = mp.solutions.hands

def process_video_with_mediapipe(video_path, label):
    cap = cv2.VideoCapture(video_path)
    frames = []
    labels = []
    
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
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
        
    return np.array(frames), np.array(labels)

def prepare_dataset(folder, save_debug=True):
    X, y = [], []
    
    if save_debug:
        os.makedirs("debug_frames", exist_ok=True)
    
    class_counts = {}
    
    for file in os.listdir(folder):
        if file.endswith(".mp4"):
            label = int(file.split("_")[0]) - 1  # 0-indexed
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
    
    X = np.array(X).reshape(-1, 64, 64, 1)  # CNN input as (64,64,1)
    y = to_categorical(y, num_classes=5)     
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# reduce amount of time spent running since before I had to reprocess the data everytime.
def load_or_process_dataset(dataset_folder="dataset", save_debug=True):
    cache_files = {
        "train_X": "train_X_improved.npy",
        "test_X": "test_X_improved.npy",
        "train_y": "train_y_improved.npy",
        "test_y": "test_y_improved.npy",
    }
    
    if all(os.path.exists(f) for f in cache_files.values()):
        print("Loading cached data.")
        train_X = np.load(cache_files["train_X"])
        test_X = np.load(cache_files["test_X"])
        train_y = np.load(cache_files["train_y"])
        test_y = np.load(cache_files["test_y"])
    else:
        print("Processing data")
        train_X, test_X, train_y, test_y = prepare_dataset(dataset_folder, save_debug)
        np.save(cache_files["train_X"], train_X)
        np.save(cache_files["test_X"], test_X)
        np.save(cache_files["train_y"], train_y)
        np.save(cache_files["test_y"], test_y)
    
    return train_X, test_X, train_y, test_y

train_X, test_X, train_y, test_y = load_or_process_dataset("dataset", save_debug=True)

model = Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))  # Output layer for 5 classes
model.summary()

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_X, train_y, epochs=20, batch_size=32, validation_data=(test_X, test_y), verbose=1)

model.save("finger_counter_model_improved.h5")
print("Model saved as 'finger_counter_model_improved.h5'")

np.save("train_X_improved.npy", train_X)
np.save("test_X_improved.npy", test_X)
np.save("train_y_improved.npy", train_y)
np.save("test_y_improved.npy", test_y)
