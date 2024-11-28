import tensorflow as tf
import kagglehub
from keras.src.layers import BatchNormalization
from tensorflow.keras import layers, models, mixed_precision
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define image size and batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Create data augmentation and rescaling layers
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.2, 0.2),
])

rescale = layers.Rescaling(1./255)

# Download latest version
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

# Load training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    path + '/chest_xray/train',
    labels='inferred',
    label_mode='binary',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True
)

# Apply data augmentation and rescaling to training dataset
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
train_dataset = train_dataset.map(lambda x, y: (rescale(x), y))
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Load validation dataset
val_dataset = tf.keras.utils.image_dataset_from_directory(
    path + '/chest_xray/val',
    labels='inferred',
    label_mode='binary',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False
)

# Apply rescaling to validation dataset
val_dataset = val_dataset.map(lambda x, y: (rescale(x), y))
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Build the model
model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=(224,224,3)),
    MaxPooling2D(pool_size=(2,2)),
    BatchNormalization(),
    Conv2D(64, kernel_size=(3,3), activation="relu"),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.1),
    BatchNormalization(),
    Conv2D(128, kernel_size=(3,3), activation="relu"),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.1),
    Flatten(),
    Dense(32, activation="relu"),
    Dropout(0.1),
    Dense(1, activation="sigmoid", dtype=tf.float32)
])

# Modify optimizer for mixed precision
base_optimizer = tf.keras.optimizers.Adam()
optimizer = mixed_precision.LossScaleOptimizer(base_optimizer)

# Compile the model with the loss scale optimizer
model.compile(
    optimizer=base_optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Fit the model
hist = model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset,
    validation_steps=2
)