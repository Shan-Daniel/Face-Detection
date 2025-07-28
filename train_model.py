import os
import numpy as np
from PIL import Image
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Directory containing images
UPLOAD_DIR = 'uploads'

# Prepare data
X = []
y = []

for filename in os.listdir(UPLOAD_DIR):
    if filename.endswith('.jpg'):
        label = 0 if 'with_mask' in filename else 1
        img_path = os.path.join(UPLOAD_DIR, filename)
        img = Image.open(img_path).convert('RGB').resize((128, 128))
        X.append(np.array(img))
        y.append(label)

X = np.array(X)
y = np.array(y)
print(f'Loaded {len(X)} images. X shape: {X.shape}, y shape: {y.shape}')
y = to_categorical(y, 2)

# Use all data for training since dataset is very small
X_train, y_train = X, y

# Data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Simple CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model with augmentation
batch_size = len(X_train) if len(X_train) > 0 else 1
model.fit(aug.flow(X_train, y_train, batch_size=batch_size), epochs=3, steps_per_epoch=3)

# Save model with error handling
print('Attempting to save model...')
try:
    model.save('model.h5')
    print('Model saved as model.h5 successfully')
except Exception as e:
    print(f'Error saving model: {e}')

print('Training complete.')
