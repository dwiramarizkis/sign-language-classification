import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

print('Loading datasets...')

# Load landmark data
try:
    landmark_dict = pickle.load(open('./data_landmarks.pickle', 'rb'))
    has_landmarks = True
except:
    has_landmarks = False
    print('No landmark data found')

# Load image data
try:
    image_dict = pickle.load(open('./data_images.pickle', 'rb'))
    has_images = True
except:
    has_images = False
    print('No image data found')

if not has_landmarks and not has_images:
    print('Error: No data found. Run create_dataset_hybrid.py first')
    exit(1)

# Prepare data
all_data = []
all_labels = []
data_types = []  # 0 for landmarks, 1 for images

if has_landmarks:
    landmark_data = landmark_dict['data']
    landmark_labels = landmark_dict['labels']
    
    # Filter consistent length
    lengths = [len(d) for d in landmark_data]
    common_length = max(set(lengths), key=lengths.count)
    
    for i, d in enumerate(landmark_data):
        if len(d) == common_length:
            all_data.append(d)
            all_labels.append(landmark_labels[i])
            data_types.append(0)
    
    print(f'Landmark samples: {len([t for t in data_types if t == 0])}')

if has_images:
    image_data = image_dict['data']
    image_labels = image_dict['labels']
    
    for i, d in enumerate(image_data):
        all_data.append(d)
        all_labels.append(image_labels[i])
        data_types.append(1)
    
    print(f'Image samples: {len([t for t in data_types if t == 1])}')

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(all_labels)
num_classes = len(label_encoder.classes_)

print(f'\nTotal samples: {len(all_data)}')
print(f'Number of classes: {num_classes}')
print(f'Classes: {label_encoder.classes_}')

# Find max length and pad all data to same size
max_length = max(len(d) for d in all_data)
print(f'Max feature length: {max_length}')

# Pad all data to max length
padded_data = []
for d in all_data:
    d_array = np.array(d, dtype=np.float32)
    if len(d_array) < max_length:
        # Pad with zeros
        padded = np.pad(d_array, (0, max_length - len(d_array)), mode='constant')
    else:
        padded = d_array
    padded_data.append(padded)

all_data = np.array(padded_data, dtype=np.float32)
labels_encoded = np.array(labels_encoded)
data_types = np.array(data_types)

print(f'Data shape after padding: {all_data.shape}')

# Split data
x_train, x_test, y_train, y_test, type_train, type_test = train_test_split(
    all_data, labels_encoded, data_types, 
    test_size=0.2, shuffle=True, stratify=labels_encoded, random_state=42
)

# Determine input shape
input_shape = x_train.shape[1]
print(f'Input shape: {input_shape}')

# Build deeper model for better discrimination
model = keras.Sequential([
    layers.Input(shape=(input_shape,)),
    
    # Deep feature extraction
    layers.Dense(1024, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    
    layers.Dense(num_classes, activation='softmax')
])

# Compile
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'model_hybrid_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Train with class weights for imbalanced data
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

print('\nTraining hybrid model...')
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=150,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

# Evaluate
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f'\nTest Accuracy: {test_accuracy * 100:.2f}%')

# Save model
best_model = keras.models.load_model('model_hybrid_best.h5')
best_model.save('model_hybrid.h5')
print('Model saved as model_hybrid.h5')

with open('label_encoder_hybrid.pickle', 'wb') as f:
    pickle.dump(label_encoder, f)
print('Label encoder saved')

# Plot
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history_hybrid.png')
print('Training history saved')
plt.show()
