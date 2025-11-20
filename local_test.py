import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os

# Configuración
MODEL_PATH = "models/final_model.h5"   
CLASSES = ['bacteria', 'saludable', 'no_reconocido']  
IMAGE_SIZE = (256, 256)

# Cargar modelo
print("🔹 Cargando modelo...")
model = keras.models.load_model(MODEL_PATH)
print("Modelo cargado con éxito.")

# Función para prueba con imagen local
def predict_image(image_path):
    img = keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    confidence = preds[0][class_idx]
    print(f"Predicción: {CLASSES[class_idx]} (confianza: {confidence*100:.2f}%)")

# Función para prueba con cámara
def predict_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return
    print("Cámara iniciada — presiona 'q' para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.resize(frame, IMAGE_SIZE)
        img_array = np.expand_dims(img, axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        preds = model.predict(img_array)
        class_idx = np.argmax(preds)
        confidence = preds[0][class_idx]
        label = f"{CLASSES[class_idx]} ({confidence*100:.1f}%)"
        cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("CNN Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\nSelecciona modo:")
    print("1. Probar imagen local")
    print("2. Usar cámara en vivo")
    option = input("Opción [1/2]: ")

    if option == "1":
        path = input("Ruta de la imagen: ").strip()
        if os.path.exists(path):
            predict_image(path)
        else:
            print("No se encontró la imagen.")
    elif option == "2":
        predict_camera()
    else:
        print("Opción inválida.")
