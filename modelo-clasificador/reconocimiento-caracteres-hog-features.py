# **1. DATASET DE IMAGENES MNIST**

import tensorflow as tf
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

print("Cargando datos MNIST...")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from google.colab.patches import cv2_imshow
cv2_imshow(cv2.resize(x_train[1],(200,100)))

y_train[0]

x_test.shape

"""# **2. DATASET DE CARACTERISTICAS CON HOG FEATURES**"""

#Función para extraer características HOG
def get_hog():
    winSize = (28, 28)
    blockSize = (8, 8)
    blockStride = (2, 2)
    cellSize = (4, 4)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    return hog

def get_feature_dataset(x):
    hog = get_hog()
    features = []
    for img in x:
        feature = hog.compute(img)
        # hog.compute() devuelve un array 2D (n, 1), aplanarlo a 1D
        features.append(feature.flatten())
    features = np.array(features)
    return features

print("Extrayendo características HOG...")
features_train = get_feature_dataset(x_train)
features_test = get_feature_dataset(x_test)

#Convertir etiquetas a one-hot encoding
y_train_onehot = tf.one_hot(y_train, np.max(y_train) + 1)
y_test_onehot = tf.one_hot(y_test, np.max(y_train) + 1)

print(f"Shape features_train: {features_train.shape}")
print(f"Shape features_test: {features_test.shape}")

img=x_train[1]
def get_hog():
  winSize=(28,28)
  blockSize=(8,8)
  blockStride=(2,2)
  cellSize=(4,4)
  nbins=9
  hog=cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
  return hog
hog=get_hog()
hog.compute(img).shape

"""**CREACIÓN DEL MODELO**"""

model = tf.keras.Sequential([
    tf.keras.layers.Dense(200, input_dim=features_train.shape[1],
                        activation='relu', name='capa1'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(180, activation='relu', name='capa2'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(150, activation='relu', name='capa3'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax', name='salida')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

print("Modelo creado")
model.summary()

model.save("modelos/modelo_ocr.h5")
print("Archivo modelo_ocr.h5 generado correctamente")

"""**CONFIGURACIÓN DE CALLBACKS**"""

#Crear carpeta para modelos si no existe
if not os.path.exists('modelos'):
    os.makedirs('modelos')
    print("✓ Carpeta 'modelos' creada")

#Nombre con fecha y hora
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
nombre_modelo_final = f'modelos/mnist_model_{timestamp}.h5'
nombre_mejor_modelo = 'modelos/best_model.h5'

callbacks = [
    #Guardar el mejor modelo durante entrenamiento
    tf.keras.callbacks.ModelCheckpoint(
        nombre_mejor_modelo,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    #Early stopping
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    #Reducir learning rate
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=0.00001
    )
]

print(f"Modelos se guardarán como:")
print(f"Durante entrenamiento: {nombre_mejor_modelo}")
print(f"Al finalizar: {nombre_modelo_final}")

"""**GUARDAR MODELO**"""

#Guardar modelo completo en formato H5
model.save(nombre_modelo_final)
print(f"✓ Modelo guardado: {nombre_modelo_final}")

#Obtener tamaño del archivo
size_mb = os.path.getsize(nombre_modelo_final) / (1024 * 1024)
print(f"  Tamaño: {size_mb:.2f} MB")

"""**DEFINICIÓN DEL MODELO MEJORADO**"""

def create_classifier():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(200, input_dim=features_train.shape[1],
                            activation='relu', name='capa1'),
        tf.keras.layers.Dropout(0.3),  # Regularización
        tf.keras.layers.Dense(180, activation='relu', name='capa2'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(150, activation='relu', name='capa3'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax', name='salida')
    ])

    # Usar Adam optimizer (más eficiente que SGD)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    return model

model = create_classifier()
model.summary()

"""**ENTRENAMIENTO POR PARTES**"""

#Callbacks para control del entrenamiento
callbacks = [
    # Guardar el mejor modelo
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    # Early stopping para evitar sobreajuste
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    # Reducir learning rate si no mejora
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        verbose=1
    )
]

#Entrenamiento completo con validación
print("\nEntrenando modelo completo...")
history = model.fit(
    features_train,
    y_train_onehot,
    epochs=20,  # Más epochs con early stopping
    batch_size=128,  # Batch size mayor para eficiencia
    validation_split=0.2,  # 20% de datos para validación
    callbacks=callbacks,
    verbose=1
)

#Entrenamiento por lotes manualmente
def entrenar_por_lotes(model, X, y, lotes=5, epochs_por_lote=5):
    """
    Entrena el modelo dividiendo los datos en lotes
    """
    n_samples = len(X)
    tamano_lote = n_samples // lotes

    for i in range(lotes):
        inicio = i * tamano_lote
        fin = (i + 1) * tamano_lote if i < lotes - 1 else n_samples

        print(f"\n{'='*50}")
        print(f"Entrenando lote {i+1}/{lotes}")
        print(f"Muestras: {inicio} a {fin}")
        print(f"{'='*50}")

        X_lote = X[inicio:fin]
        y_lote = y[inicio:fin]

        history_lote = model.fit(
            X_lote,
            y_lote,
            epochs=epochs_por_lote,
            batch_size=100,
            validation_split=0.2,
            verbose=1
        )

    return model

# Descomentar para usar entrenamiento por lotes
# model_lotes = create_classifier()
# model_lotes = entrenar_por_lotes(model_lotes, features_train, y_train_onehot)



"""**EVALUACIÓN DEL MODELO**"""

#Predicciones
prediction_train = model.predict(features_train)
prediction_test = model.predict(features_test)

y_pred_train = np.argmax(prediction_train, 1)
y_pred_test = np.argmax(prediction_test, 1)

#Calcular errores
error_train = 100 * np.sum(y_pred_train != y_train) / len(y_train)
error_test = 100 * np.sum(y_pred_test != y_test) / len(y_test)

print(f"\nError entrenamiento: {error_train:.2f}%")
print(f"Error prueba: {error_test:.2f}%")
print(f"Precisión entrenamiento: {100-error_train:.2f}%")
print(f"Precisión prueba: {100-error_test:.2f}%")

"""**VISUALIZACIÓN DE RESULTADOS**"""

#Gráfica de entrenamiento
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['loss'], label='Pérdida entrenamiento')
ax1.plot(history.history['val_loss'], label='Pérdida validación')
ax1.set_xlabel('Época')
ax1.set_ylabel('Pérdida')
ax1.set_title('Evolución de la Pérdida')
ax1.legend()

ax2.plot(history.history['accuracy'], label='Precisión entrenamiento')
ax2.plot(history.history['val_accuracy'], label='Precisión validación')
ax2.set_xlabel('Época')
ax2.set_ylabel('Precisión')
ax2.set_title('Evolución de la Precisión')
ax2.legend()

plt.tight_layout()
plt.show()

#Matriz de confusión
conf_mat = confusion_matrix(y_test, y_pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
disp.plot(cmap='Blues')
plt.title('Matriz de Confusión - Conjunto de Prueba')
plt.show()

#Matriz normalizada
conf_mat_norm = np.round(100 * conf_mat / np.sum(conf_mat, 1, keepdims=True), 1)
disp2 = ConfusionMatrixDisplay(confusion_matrix=conf_mat_norm)
disp2.plot(cmap='Blues')
plt.title('Matriz de Confusión Normalizada (%)')
plt.show()

"""**VERIFICACIÓN DE MODELO GUARDADO**"""

#Cargar modelo H5
modelo_cargado = tf.keras.models.load_model(nombre_modelo_final)
print(f"Modelo cargado correctamente desde: {nombre_modelo_final}")

#Probar con una predicción
test_sample = features_test[0:1]
prediccion_original = model.predict(test_sample, verbose=0)
prediccion_cargada = modelo_cargado.predict(test_sample, verbose=0)

digito_original = np.argmax(prediccion_original)
digito_cargado = np.argmax(prediccion_cargada)
digito_real = y_test[0]

print(f"\nPrueba de predicción:")
print(f"  Dígito real:              {digito_real}")
print(f"  Predicción modelo actual: {digito_original}")
print(f"  Predicción modelo cargado: {digito_cargado}")

if np.allclose(prediccion_original, prediccion_cargada):
    print("Verificación exitosa: Las predicciones coinciden perfectamente")
else:
    print("Advertencia: Hay pequeñas diferencias en las predicciones")

import h5py

with h5py.File("modelo_ocr.h5", "r") as f:
    print("Estructura:")
    f.visit(print)
