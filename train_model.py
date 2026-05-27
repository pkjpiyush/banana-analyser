import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json, os

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
DATASET_DIR = "dataset"
os.makedirs("model", exist_ok=True)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    f"{DATASET_DIR}/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
val_data = val_datagen.flow_from_directory(
    f"{DATASET_DIR}/val",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                          include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

with open("model/class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)

print("🍌 Starting training...")
history = model.fit(train_data, epochs=EPOCHS, validation_data=val_data)

model.save("model/banana_model.h5")
print("✅ Model saved to model/banana_model.h5")