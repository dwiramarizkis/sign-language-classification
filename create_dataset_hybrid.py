import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands

# Use multiple detection attempts with different settings
hands_configs = [
    mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=1, model_complexity=1),
    mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.1, max_num_hands=1, model_complexity=1),
    mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.05, max_num_hands=1, model_complexity=0),
]

DATA_DIR = './asl_dataset'

# Two types of data: landmarks and raw images
landmark_data = []
landmark_labels = []
image_data = []
image_labels = []

skipped_files = 0
processed_with_landmarks = 0
processed_with_images = 0
no_detection = 0

# Gestures that are typically hard to detect (closed fist)
HARD_GESTURES = ['a', 'm', 'n', 's', 't', 'e']

print('Creating hybrid dataset (landmarks + raw images)...\n')

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue

    print(f'Processing class: {dir_}')
    landmark_count = 0
    image_count = 0
    is_hard_gesture = dir_.lower() in HARD_GESTURES

    for img_path in os.listdir(dir_path):
        if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_full_path = os.path.join(DATA_DIR, dir_, img_path)
        img = cv2.imread(img_full_path)

        if img is None:
            skipped_files += 1
            continue

        # Resize to standard size for raw image processing
        img_resized = cv2.resize(img, (64, 64))
        img_normalized = img_resized.astype('float32') / 255.0

        # Try to detect landmarks with multiple configurations
        detected = False
        for hands in hands_configs:
            h, w = img.shape[:2]
            if h < 100 or w < 100:
                scale = max(100 / h, 100 / w)
                img_test = cv2.resize(img, None, fx=scale, fy=scale)
            else:
                img_test = img

            img_rgb = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                x_ = []
                y_ = []
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Normalize coordinates
                data_aux = []
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))
                
                # Add extra features for better discrimination
                # 1. Palm orientation (angle between wrist and middle finger)
                wrist = hand_landmarks.landmark[0]
                middle_mcp = hand_landmarks.landmark[9]
                palm_angle = np.arctan2(middle_mcp.y - wrist.y, middle_mcp.x - wrist.x)
                data_aux.append(palm_angle)
                
                # 2. Hand openness (distance between thumb and pinky tips)
                thumb_tip = hand_landmarks.landmark[4]
                pinky_tip = hand_landmarks.landmark[20]
                openness = np.sqrt((thumb_tip.x - pinky_tip.x)**2 + (thumb_tip.y - pinky_tip.y)**2)
                data_aux.append(openness)
                
                # 3. Finger curl (average distance from fingertips to palm)
                palm_center_x = np.mean([hand_landmarks.landmark[i].x for i in [0, 5, 9, 13, 17]])
                palm_center_y = np.mean([hand_landmarks.landmark[i].y for i in [0, 5, 9, 13, 17]])
                fingertips = [4, 8, 12, 16, 20]
                avg_curl = np.mean([
                    np.sqrt((hand_landmarks.landmark[i].x - palm_center_x)**2 + 
                           (hand_landmarks.landmark[i].y - palm_center_y)**2)
                    for i in fingertips
                ])
                data_aux.append(avg_curl)

                landmark_data.append(data_aux)
                landmark_labels.append(dir_)
                landmark_count += 1
                processed_with_landmarks += 1
                detected = True
                break

        # For hard gestures or failed detection, use raw image
        if not detected or is_hard_gesture:
            image_data.append(img_normalized.flatten())
            image_labels.append(dir_)
            image_count += 1
            processed_with_images += 1
            if not detected:
                no_detection += 1

    print(f'  Landmarks: {landmark_count}, Raw images: {image_count}')

# Close all hands instances
for hands in hands_configs:
    hands.close()

print(f'\n=== Summary ===')
print(f'Processed with landmarks: {processed_with_landmarks}')
print(f'Processed with raw images: {processed_with_images}')
print(f'Failed detection (using raw): {no_detection}')
print(f'Skipped (corrupt): {skipped_files}')

# Save both datasets
with open('data_landmarks.pickle', 'wb') as f:
    pickle.dump({'data': landmark_data, 'labels': landmark_labels}, f)
print('\nLandmark dataset saved as data_landmarks.pickle')

with open('data_images.pickle', 'wb') as f:
    pickle.dump({'data': image_data, 'labels': image_labels}, f)
print('Image dataset saved as data_images.pickle')
