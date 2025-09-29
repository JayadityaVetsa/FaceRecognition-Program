# train_emotion_cnn.py
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

DATA_DIR = Path('data/emotions')
MODEL_PATH = Path('models')
MODEL_PATH.mkdir(parents=True, exist_ok=True)

img_size = (64,64)
batch_size = 32
epochs = 25

# Data augmentation + generator
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2,
                                   rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
                                   shear_range=0.1, zoom_range=0.1, fill_mode='nearest')

train_gen = train_datagen.flow_from_directory(str(DATA_DIR), target_size=img_size, color_mode='grayscale',
                                              batch_size=batch_size, class_mode='categorical', subset='training')
val_gen = train_datagen.flow_from_directory(str(DATA_DIR), target_size=img_size, color_mode='grayscale',
                                            batch_size=batch_size, class_mode='categorical', subset='validation')

num_classes = train_gen.num_classes
print("Classes:", train_gen.class_indices)

# Model
model = models.Sequential([
    layers.Input(shape=(img_size[0], img_size[1], 1)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(str(MODEL_PATH/'emotion_cnn.h5'), save_best_only=True),
    tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)
]

history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)
model.save(str(MODEL_PATH/'emotion_cnn_final.h5'))
print("Saved emotion_cnn model to models/")
