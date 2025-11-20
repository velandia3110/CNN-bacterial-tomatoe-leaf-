import tensorflow as tf
from tensorflow import keras
import numpy as np
import sklearn.metrics as skm
import os

MODEL_PATH = "models/final_model.h5"
DATA_DIR = "dataset/processed/test"
IMAGE_SIZE=(224,224)
batch_size=32

model = keras.models.load_model(MODEL_PATH)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(DATA_DIR, image_size=IMAGE_SIZE, batch_size=batch_size, label_mode='categorical', shuffle=False)
y_true = []
y_pred = []
filenames = []
for batch_imgs, batch_labels in test_ds:
    preds = model.predict(batch_imgs)
    y_true.extend(np.argmax(batch_labels.numpy(),axis=1).tolist())
    y_pred.extend(np.argmax(preds,axis=1).tolist())

labels = test_ds.class_names
print("Etiquetas:", labels)

# Metricas
acc = skm.accuracy_score(y_true, y_pred)
prec = skm.precision_score(y_true, y_pred, average='weighted', zero_division=0)
rec = skm.recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = skm.f1_score(y_true, y_pred, average='weighted', zero_division=0)
cm = skm.confusion_matrix(y_true, y_pred)
print(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1: {f1:.4f}")
print("Confusion matrix:\n", cm)
