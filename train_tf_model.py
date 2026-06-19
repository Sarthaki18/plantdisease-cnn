
"""
Train/fine-tune a TensorFlow MobileNetV2 classifier on the extracted augmented plant leaf dataset.
The saved model and metadata are consumed directly by src/app.py.
"""

import json
from pathlib import Path

import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = Path(r"D:\Plant_leave_diseases_dataset_with_augmentation")
OUTPUT_DIR = PROJECT_ROOT / "src" / "Classification-based Anomaly Detection"
MODEL_SAVE_PATH = OUTPUT_DIR / "plant_disease_model_tf.h5"
KERAS_SAVE_PATH = OUTPUT_DIR / "plant_disease_model_tf.keras"
METADATA_PATH = OUTPUT_DIR / "model_metadata.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 5
EPOCHS_FINE_TUNE = 5
SEED = 42
VALIDATION_SPLIT = 0.2
EXCLUDED_CLASSES = {"Background_without_leaves"}


def configure_runtime():
    tf.keras.utils.set_random_seed(SEED)
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def load_datasets():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_PATH}")

    class_names = sorted(
        path.name
        for path in DATASET_PATH.iterdir()
        if path.is_dir() and path.name not in EXCLUDED_CLASSES
    )
    if not class_names:
        raise RuntimeError(f"No trainable disease classes found in {DATASET_PATH}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        class_names=class_names,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_PATH,
        class_names=class_names,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(1000, seed=SEED).prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    return train_ds, val_ds, class_names


def build_model(num_classes):
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = tf.keras.layers.RandomFlip("horizontal")(x)
    x = tf.keras.layers.RandomRotation(0.04)(x)
    x = tf.keras.layers.RandomZoom(0.08)(x)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    x = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base_model


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )


def main():
    configure_runtime()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("TensorFlow Plant Disease Fine-tuning")
    print("=" * 72)
    print(f"Dataset: {DATASET_PATH}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPUs: {tf.config.list_physical_devices('GPU')}")

    train_ds, val_ds, class_names = load_datasets()
    num_classes = len(class_names)
    print(f"Classes: {num_classes}")

    model, base_model = build_model(num_classes)
    compile_model(model, 1e-3)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(KERAS_SAVE_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
        ),
    ]

    print("\nTraining classifier head...")
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        callbacks=callbacks,
        verbose=1,
    )

    print("\nFine-tuning top MobileNetV2 layers...")
    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False
    compile_model(model, 1e-5)

    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE_TUNE,
        callbacks=callbacks,
        verbose=1,
    )

    val_loss, val_accuracy = model.evaluate(val_ds, verbose=1)
    print(f"Final validation accuracy: {val_accuracy:.4f}")

    model.save(MODEL_SAVE_PATH)
    model.save(KERAS_SAVE_PATH)

    metadata = {
        "classes": class_names,
        "num_classes": num_classes,
        "image_size": list(IMAGE_SIZE),
        "architecture": "MobileNetV2 transfer learning",
        "framework": "TensorFlow/Keras",
        "dataset_path": str(DATASET_PATH),
        "model_path": str(MODEL_SAVE_PATH),
        "keras_model_path": str(KERAS_SAVE_PATH),
        "epochs_head": EPOCHS_HEAD,
        "epochs_fine_tune": EPOCHS_FINE_TUNE,
        "val_loss": float(val_loss),
        "val_accuracy": float(val_accuracy),
        "history": {
            "head": {k: [float(v) for v in vals] for k, vals in history_head.history.items()},
            "fine_tune": {k: [float(v) for v in vals] for k, vals in history_fine.history.items()},
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved H5 model: {MODEL_SAVE_PATH}")
    print(f"Saved Keras model: {KERAS_SAVE_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
