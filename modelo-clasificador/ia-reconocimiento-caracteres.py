import tensorflow as tf

tf.random.set_seed(54321)

# =====================================================================
# Etapa 1: Datos originales
# =====================================================================

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"Muestras de entrenamiento: {len(x_train)}")
print(f"Muestras de prueba: {len(x_test)}")

# =====================================================================
# Etapa 2: Preprocesamiento de datos
# =====================================================================

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype("float32") / 255.0

y_trainOneHot = tf.one_hot(y_train, 10)
y_testOneHot = tf.one_hot(y_test, 10)

print(f"x_train: {x_train.shape}")
print(f"y_trainOneHot: {y_trainOneHot.shape}")

# =====================================================================
# Etapa 3: Creación y preparación de tensores
# =====================================================================
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    validation_split=0.1,
)

datagen.fit(x_train)

# =====================================================================
# Etapa 4: Modelo y entrenamiento
# =====================================================================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from google.colab import drive
import os

def cnn_classifier():
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax"),
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=1e-3),
        metrics=["accuracy"],
    )

    return model


model = cnn_classifier()
model.summary()

early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6
)

history = model.fit(
    datagen.flow(x_train, y_trainOneHot, batch_size=128, subset="training"),
    validation_data=datagen.flow(x_train, y_trainOneHot, batch_size=128, subset="validation"),
    epochs=50,
    callbacks=[early_stopping, reduce_lr],
)

loss, acc = model.evaluate(x_test, y_testOneHot, verbose=0)
print(f"\nPrecisión del modelo: {acc*100:.2f}% - Pérdida: {loss:.4f}")

drive.mount("/content/drive")
base_dir = "/content/drive/MyDrive/datasets/reconocimiento-caracteres"
os.makedirs(base_dir, exist_ok=True)
model_path = os.path.join(base_dir, "reconocimiento-caracteres.model.keras")
model.save(model_path)
print(f"Modelo guardado en: {model_path}")

# =====================================================================
# Etapa 5: Evaluación y predicción
# =====================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_score, recall_score, f1_score

np.random.seed(54321)

y_pred = np.argmax(model.predict(x_test), axis=1)
print(classification_report(y_test, y_pred, digits=4))

precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
accuracy = np.mean(y_pred == y_test)

print(f'Exactitud (Accuracy): {accuracy*100:.2f}%')
print(f'Precisión (Precision): {precision*100:.2f}%')
print(f'Recall: {recall*100:.2f}%')
print(f'F1-Score: {f1*100:.2f}%')

conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=range(10)).plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusión - MNIST")
plt.show()

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Entrenamiento")
plt.plot(history.history["val_accuracy"], label="Validación")
plt.title("Evolución de la Precisión")
plt.xlabel("Épocas")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Entrenamiento")
plt.plot(history.history["val_loss"], label="Validación")
plt.title("Evolución de la Pérdida")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.legend()
plt.show()

idx = np.random.randint(0, len(x_test))
sample = x_test[idx]
true_label = y_test[idx]

pred = model.predict(sample.reshape(1, 28, 28, 1))
pred_label = np.argmax(pred)
conf = np.max(pred) * 100

print(f"\nEjemplo [{idx}]")
print(f"Etiqueta verdadera: {true_label}, predicción: {pred_label}, confianza: {conf:.2f}%")
cv2_imshow(cv2.resize((sample * 255).astype(np.uint8).squeeze(), (200, 200), interpolation=cv2.INTER_NEAREST))
