# 🔢 Reconocimiento de Caracteres - Clasificador de Dígitos

Un proyecto de inteligencia artificial para clasificar dígitos manuscritos (0-9) utilizando una red neuronal convolucional (CNN) entrenada con TensorFlow. Incluye una interfaz interactiva desarrollada con Streamlit para pruebas en tiempo real.

## Descripción

Este proyecto implementa un sistema completo de reconocimiento de caracteres que:

- **Entrena un modelo CNN** sobre el dataset MNIST (70,000 imágenes de dígitos)
- **Aplica técnicas de aumentación de datos** (rotación, zoom, desplazamiento) para mejorar la generalización
- **Proporciona una interfaz web** interactiva para clasificar dígitos
- **Soporta múltiples modos de entrada**: carga de imágenes, dibujo en canvas o entrada manual

## Características

- Modelo pre-entrenado con alta precisión
- Interfaz web interactiva con Streamlit
- Tres modos de entrada flexible:
  - Cargar imagen desde archivo
  - Dibujar dígito en canvas
  - Entrada manual de píxeles
- Predicciones en tiempo real con niveles de confianza
- Historial de predicciones
- Preprocesamiento automático de imágenes

## Estructura del Proyecto

```
reconocimiento-caracteres/
├── README.md                                    # Este archivo
├── requirements.txt                             # Dependencias del proyecto
├── requirements-lock.txt                        # Dependencias bloqueadas (reproducibilidad)
├── model/
│   └── reconocimiento-caracteres.model.keras   # Modelo pre-entrenado
├── modelo-clasificador/
│   └── ia-reconocimiento-caracteres.py         # Script de entrenamiento del modelo
└── src/
    ├── main.py                                  # Aplicación principal Streamlit
    ├── model_handler.py                         # Gestión del modelo y predicciones
    ├── image_processor.py                       # Procesamiento de imágenes
    ├── session_state.py                         # Gestión del estado de sesión
    └── ui_components.py                         # Componentes de la interfaz
```

## Instalación

### Requisitos previos
- Python 3.8 o superior
- pip o conda

### Pasos de instalación

1. **Clonar o descargar el repositorio**
   ```bash
   cd reconocimiento-caracteres
   ```

2. **Crear un entorno virtual (recomendado)**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

## Uso

### Ejecutar la aplicación

```bash
streamlit run src/main.py
```

La aplicación se abrirá en tu navegador por defecto en `http://localhost:8501`

### Modos de entrada disponibles

1. **Cargar Imagen**: Sube una imagen PNG, JPG o BMP de un dígito
2. **Dibujar**: Dibuja un dígito en el canvas interactivo
3. **Entrada Manual**: Ingresa manualmente píxeles para el dígito

### Ejemplo de uso

1. Selecciona un modo de entrada
2. Proporciona tu dígito (carga, dibuja o ingresa datos)
3. Haz clic en "Predecir"
4. Visualiza el resultado con el nivel de confianza

## Modelo

### Arquitectura CNN

El modelo utiliza una arquitectura convolucional con:

- **Capas Convolucionales**: Extracción de características
- **Max Pooling**: Reducción de dimensionalidad
- **Dropout**: Regularización para prevenir overfitting
- **Capas Densas**: Clasificación final

### Datos de Entrenamiento

- **Dataset**: MNIST (70,000 imágenes de dígitos 28×28)
- **División**: 90% entrenamiento, 10% validación
- **Aumentación de datos**: Rotación, zoom, desplazamiento y shear

### Optimización

- **Optimizador**: Adam
- **Loss**: Categorical Crossentropy
- **Early Stopping**: Previene overfitting
- **Learning Rate Reduction**: Ajuste dinámico del aprendizaje

## Dependencias principales

```
streamlit >= 1.28.0           # Framework web interactivo
streamlit-drawable-canvas >= 0.2.0  # Canvas para dibujar
tensorflow >= 2.13.0          # Framework de deep learning
opencv-python >= 4.8.0        # Procesamiento de imágenes
pillow >= 10.0.0              # Manipulación de imágenes
numpy >= 1.24.0               # Computación numérica
```

Para ver todas las dependencias, consulta `requirements.txt` y `requirements-lock.txt`.
