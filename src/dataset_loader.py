import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

def load_dataset(train_dir, val_dir, batch_size=32):
    """Loads the dataset and applies augmentation for better accuracy."""

    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1.0/255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical"
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical"
    )

    return train_generator, val_generator

# ✅ Test dataset loading
if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
    VAL_DIR = os.path.join(BASE_DIR, "dataset", "validation")

    print("🔄 Loading dataset...")
    train_data, val_data = load_dataset(TRAIN_DIR, VAL_DIR)
    print(f"✅ Train Data Loaded: {train_data.samples} images")
    print(f"✅ Validation Data Loaded: {val_data.samples} images")
