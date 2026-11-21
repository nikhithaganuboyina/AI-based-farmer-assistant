import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# 1. Configuration
DATASET_DIR = "dataset" # Create this folder and put your crop folders inside
BATCH_SIZE = 32
EPOCHS = 10
IMG_SIZE = (224, 224)

print("🚀 Starting Multi-Crop Model Training...")

if not os.path.exists(DATASET_DIR):
    print(f"❌ Error: Dataset directory '{DATASET_DIR}' not found!")
    print("Please create a 'dataset' folder and organize your images like:")
    print("dataset/")
    print("  ├── Apple___Apple_scab/")
    print("  ├── Apple___healthy/")
    print("  ├── Potato___Early_blight/")
    print("  └── ...")
    exit(1)

# 2. Data Preparation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2, # 80% training, 20% validation
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

num_classes = train_generator.num_classes

print(f"✅ Found {num_classes} crop classes.")

# 3. Save class indices (CRITICAL for predict_disease.py)
class_indices = train_generator.class_indices
with open("class_indices.json", "w") as f:
    json.dump(class_indices, f, indent=4)
print("✅ Saved class_indices.json")

# 4. Build Model (Must match predict_disease.py architecture)
base_model = MobileNetV2(
    weights="imagenet", # Use pre-trained weights for faster learning
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model layers so we don't destroy pre-trained features
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 5. Train Model
print("🔄 Training model...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# 6. Save Weights (predict_disease.py uses load_weights)
model.save_weights("repaired_model.h5")
print("✅ Saved repaired_model.h5")

print("\n🎉 Training Complete!")
print("Next Steps:")
print("1. Move 'class_indices.json' and 'repaired_model.h5' into your 'models/' folder.")
print("2. Restart your Flask app.")
