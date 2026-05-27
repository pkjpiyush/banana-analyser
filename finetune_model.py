import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
DATASET_DIR = "dataset"

# Load the model
model = load_model("model/banana_model.h5")

# Print layers so we can see the structure
print("Model layers:")
for i, layer in enumerate(model.layers):
    print(f"{i}: {layer.name} - trainable: {layer.trainable}")

# Unfreeze last 30 layers of the whole model
for layer in model.layers[-30:]:
    layer.trainable = True

# Recompile with very low learning rate
model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Data generators
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

print("🍌 Starting fine-tuning...")
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data
)

model.save("model/banana_model_finetuned.h5")
print("✅ Fine-tuned model saved!")
print(f"Final val accuracy: {max(history.history['val_accuracy'])*100:.2f}%")