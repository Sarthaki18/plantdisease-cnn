
import tensorflow as tf
import os
import argparse
import json
from pathlib import Path

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

print("========================================================================")
print("TensorFlow Plant Disease Fine-tuning")
print("========================================================================")

# Argument parsing
parser = argparse.ArgumentParser(description='TensorFlow Plant Disease Fine-tuning')
parser.add_argument('--dataset_path', type=str, default='/content/Plant_leave_diseases_dataset_with_augmentation',
                    help='Path to the dataset directory')
parser.add_argument('--image_size', type=int, default=96, # Further reduced image size
                    help='Image size for training (e.g., 224 for MobileNetV2)')
parser.add_argument('--batch_size', type=int, default=4, # Already reduced batch size
                    help='Batch size for training')
parser.add_argument('--epochs', type=int, default=5,
                    help='Number of epochs for fine-tuning')
parser.add_argument('--learning_rate', type=float, default=0.0001,
                    help='Learning rate for the optimizer')
parser.add_argument('--output_dir', type=str,
                    default=r'D:\Plant-Disease-Classification-using-CNN-main\Plant-Disease-Classification-using-CNN-main\src\Classification-based Anomaly Detection',
                    help='Directory where the model and metadata will be saved')
parser.add_argument('--model_name', type=str, default='plant_disease_model_tf.keras',
                    help='Keras model filename')
parser.add_argument('--exclude_background', action='store_true', default=True,
                    help='Exclude Background_without_leaves so the model only predicts plant classes')
args = parser.parse_args()

# Print system information
print(f"Dataset: {args.dataset_path}")
print(f"TensorFlow: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs: {gpus}")
else:
    print("No GPU devices found.")

# Data loading and preprocessing
IMG_HEIGHT = args.image_size
IMG_WIDTH = args.image_size
BATCH_SIZE = args.batch_size

# Create a function to parse images
def parse_image(filepath):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.cast(img, tf.float32) / 255.0 # Normalize to [0,1]
    return img

# Load datasets
excluded_classes = {'Background_without_leaves'} if args.exclude_background else set()
class_names_for_training = sorted(
    entry.name for entry in Path(args.dataset_path).iterdir()
    if entry.is_dir() and entry.name not in excluded_classes
)
if not class_names_for_training:
    raise RuntimeError(f"No trainable classes found in {args.dataset_path}")

train_ds = tf.keras.utils.image_dataset_from_directory(
    args.dataset_path,
    class_names=class_names_for_training,
    labels='inferred',
    label_mode='int',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset='training'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    args.dataset_path,
    class_names=class_names_for_training,
    labels='inferred',
    label_mode='int',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=42,
    validation_split=0.2,
    subset='validation'
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print(f"Classes: {NUM_CLASSES}")

# Configure dataset for performance (removed .cache())
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Model definition (MobileNetV2 base model)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False # Freeze the base model

# Create a new classifier head
inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x) # Output layer should be float32
model = tf.keras.Model(inputs, x)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.summary()

print("Training classifier head...")
# Train the classifier head
history = model.fit(
    train_ds,
    epochs=args.epochs,
    validation_data=val_ds
)

# Fine-tuning: Unfreeze the base model and re-train with a lower learning rate
print("Fine-tuning full model...")
base_model.trainable = True

# Recompile the model with a lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate / 10),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.summary()

# Continue training
history_fine_tune = model.fit(
    train_ds,
    epochs=args.epochs * 2, # Continue for another args.epochs during fine-tuning
    initial_epoch=history.epoch[-1], # Start from where classifier head training left off
    validation_data=val_ds
)

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
model_path = output_dir / args.model_name
metadata_path = output_dir / 'model_metadata.json'
root_model_path = Path(r'D:\Plant-Disease-Classification-using-CNN-main\Plant-Disease-Classification-using-CNN-main\plant_disease_model.keras')

model.save(model_path)
model.save(root_model_path)

metadata = {
    'classes': class_names,
    'num_classes': NUM_CLASSES,
    'image_size': [IMG_HEIGHT, IMG_WIDTH],
    'architecture': 'MobileNetV2',
    'framework': 'TensorFlow/Keras',
    'dataset_path': args.dataset_path,
    'model_path': str(model_path),
    'root_model_path': str(root_model_path),
    'excluded_classes': sorted(excluded_classes),
    'preprocessing': 'raw RGB 0..255 input; tf.keras.applications.mobilenet_v2.preprocess_input inside model',
    'head_epochs': args.epochs,
    'fine_tune_epochs': args.epochs,
}
metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

print(f"Saved model: {model_path}")
print(f"Saved app model: {root_model_path}")
print(f"Saved metadata: {metadata_path}")
print("Training complete.")
