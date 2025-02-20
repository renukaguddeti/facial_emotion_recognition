import os
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
from dataset_loader import load_dataset
from model import build_model  

# ✅ Define dataset paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
VAL_DIR = os.path.join(BASE_DIR, "dataset", "validation")

# ✅ Load Dataset (Pass both train & validation directories)
train_data, val_data = load_dataset(TRAIN_DIR, VAL_DIR)

# ✅ Build Model
model = build_model()

# ✅ Save Best Model During Training
model_save_path = os.path.join(BASE_DIR, "models", "emotion_model.h5")
checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, monitor="val_accuracy", mode="max")

# ✅ Train Model
model.fit(
    train_data,
    validation_data=val_data,
    epochs=25,
    callbacks=[checkpoint]
)

print(f"✅ Model saved at: {model_save_path}")
