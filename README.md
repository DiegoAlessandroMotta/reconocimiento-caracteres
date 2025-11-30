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
- Dos modos de entrada flexible:
  - Cargar imagen desde archivo
  - Dibujar dígito en canvas
- Predicciones en tiempo real con niveles de confianza
- Historial de predicciones
- Preprocesamiento automático de imágenes
- Umbral de certeza configurable
- Preprocesamiento inteligente (mantiene relación de aspecto y recorta automáticamenet el contenido)
- Visualización de imagen procesada

## Estructura del Proyecto

```
reconocimiento-caracteres/
├── README.md                                    # Este archivo
├── requirements.txt                             # Dependencias para desarrollo local
├── requirements-cloud.txt                       # Dependencias para despliegue en Streamlit Cloud
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
   
   **Para desarrollo local:**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Para despliegue en Streamlit Cloud:**
   - Usa `requirements-cloud.txt` en lugar de `requirements.txt`
   - Este archivo usa `opencv-python-headless` que es compatible con Streamlit Cloud

## 💻 Uso

### Ejecutar la aplicación localmente

```bash
source venv/bin/activate

# Ejecutar la aplicación
streamlit run src/main.py
```

La aplicación se abrirá en tu navegador por defecto en `http://localhost:8501`

## Despliegue en Streamlit Cloud

Para desplegar en Streamlit Community Cloud:

1. Sube tu repositorio a GitHub
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu repositorio
4. En "Main file path", especifica: `src/main.py`
5. En "Requirements file", especifica: `requirements-cloud.txt`

**Nota importante:** Streamlit Cloud no soporta `opencv-python` porque requiere librerías gráficas del sistema. Usa `requirements-cloud.txt` que incluye `opencv-python-headless` en su lugar.

### Modos de entrada disponibles

1. **Cargar Imagen**: Sube una imagen PNG, JPG o BMP de un dígito
2. **Dibujar**: Dibuja un dígito en el canvas interactivo
3. **Entrada Manual**: Ingresa manualmente píxeles para el dígito

### Ejemplo de uso

1. Selecciona un modo de entrada
2. Proporciona tu dígito (carga, dibuja o ingresa datos)
3. Haz clic en "Predecir"
4. Visualiza el resultado con el nivel de confianza

**Nota sobre el umbral de certeza:** Si la confianza del modelo es menor al 70%, la aplicación indicará que no se pudo clasificar el dígito. Esto ayuda a evitar clasificaciones incorrectas cuando la imagen no es clara o no contiene un dígito reconocible.

**Nota sobre el preprocesamiento:** El sistema mantiene automáticamente la relación de aspecto de las imágenes originales, evitando deformaciones que podrían impedir el reconocimiento correcto de dígitos en imágenes rectangulares. Además, recorta automáticamente el contenido relevante eliminando espacio vacío, lo que permite que los dígitos ocupen el máximo espacio posible en la imagen procesada.

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

## 📦 Dependencias principales

```
streamlit >= 1.28.0           # Framework web interactivo
streamlit-drawable-canvas >= 0.2.0  # Canvas para dibujar
tensorflow >= 2.13.0          # Framework de deep learning
numpy >= 1.24.0               # Computación numérica
pillow >= 10.0.0              # Manipulación de imágenes
opencv-python >= 4.8.0        # Procesamiento de imágenes (desarrollo local)
# opencv-python-headless >= 4.8.0  # Para despliegue en Streamlit Cloud
```

Para ver todas las dependencias, consulta `requirements.txt` (desarrollo local) o `requirements-cloud.txt` (despliegue en la nube).
