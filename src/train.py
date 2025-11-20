import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# Parámetros
DATA_DIR = "dataset/processed"
IMAGE_SIZE = (256,256)
BATCH_SIZE = 10
NUM_CLASSES = 2
EPOCHS = 60
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Carga de datos
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR,"train"),
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR,"val"),
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR,"test"),
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False
)

# Precarga de datos para rendimiento
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# Aumento durante ejecución
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.12),
    layers.RandomTranslation(0.08,0.08)
])

# Construcción de modelo
base = tf.keras.applications.EfficientNetV2S(
    input_shape=(*IMAGE_SIZE,3),
    include_top=False,
    weights='imagenet'
)
base.trainable = False  

inputs = keras.Input(shape=(*IMAGE_SIZE,3))
x = data_augmentation(inputs)
x = tf.keras.applications.efficientnet.preprocess_input(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = keras.Model(inputs, outputs)

# Compilación
optimizer = tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5) if 'tfa' in globals() else keras.optimizers.Adam(1e-3)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall'), keras.metrics.AUC(name='auc')]
)

# Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR,"best_model.h5"), save_best_only=True, monitor='val_loss'),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

# Entrenamiento (Fase 1)
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# Ajustes en capas superiores
base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall'), keras.metrics.AUC(name='auc')]
)
history_ft = model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks)

# Guardado de modelo final
model.save(os.path.join(MODEL_DIR,"final_model.h5"))
print("Model saved.")
