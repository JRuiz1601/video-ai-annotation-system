## 📑 Tabla de Contenidos

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Arquitectura del Sistema](#2-arquitectura-del-sistema)
3. [Requisitos y Dependencias](#3-requisitos-y-dependencias)
4. [Procedimiento de Despliegue](#4-procedimiento-de-despliegue)
5. [Configuración del Sistema](#5-configuración-del-sistema)
6. [Pruebas y Validación](#6-pruebas-y-validación)

---

## 1. Resumen Ejecutivo

### 1.1 Objetivo del Despliegue

Implementar una **aplicación web interactiva** que permita la clasificación en tiempo real de actividades humanas mediante análisis de video por webcam, utilizando técnicas de visión por computadora y aprendizaje automático.

### 1.2 Características Principales

- ✅ **Clasificación en Tiempo Real:** Procesamiento de video streaming desde webcam
- ✅ **5 Actividades Detectables:** Caminar hacia/de regreso, girar, sentarse, ponerse de pie
- ✅ **Alto Rendimiento:** 98.55% accuracy con inferencia rápida (~50-60 FPS)
- ✅ **Interfaz Intuitiva:** UI basada en Gradio con visualización de landmarks
- ✅ **Acceso Remoto:** URL pública compartible vía túnel ngrok
- ✅ **Sin Instalación Cliente:** Acceso desde cualquier navegador moderno

### 1.3 Especificaciones Técnicas

| Componente | Tecnología | Versión |
|------------|------------|---------|
| Modelo ML | Random Forest | scikit-learn 1.5.2 |
| Detección Pose | MediaPipe Pose | 0.10.21 |
| Framework UI | Gradio | 5.8.0 |
| Procesamiento Video | OpenCV | 4.10.0.84 |
| Plataforma | Google Colab / Local | Python 3.10+ |

---

## 2. Arquitectura del Sistema

### 2.1 Diagrama de Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                         USUARIO (Navegador)                      │
│                                                                  │
│  Webcam → Gradio Interface (HTML/JS) → Video Stream            │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTPS (Túnel ngrok)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SERVIDOR (Google Colab / Local)               │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Gradio Application                     │  │
│  │                   (process_frame function)                │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         │                                        │
│  ┌──────────────────────▼───────────────────────────────────┐  │
│  │               MediaPipe Pose Detection                    │  │
│  │       (Extracción de 33 landmarks × 4 coords)            │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         │                                        │
│  ┌──────────────────────▼───────────────────────────────────┐  │
│  │            Feature Engineering Pipeline                   │  │
│  │   • 132 features raw (landmarks)                         │  │
│  │   • 19 distancias euclidianas                            │  │
│  │   • 15 ángulos articulares                               │  │
│  │   • 15 ratios y características adicionales              │  │
│  │   → Total: 83 features geométricas                       │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         │                                        │
│  ┌──────────────────────▼───────────────────────────────────┐  │
│  │         Preprocessing & Transformation Pipeline           │  │
│  │   1. StandardScaler (normalización)                      │  │
│  │   2. PCA (reducción a 16 componentes)                    │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         │                                        │
│  ┌──────────────────────▼───────────────────────────────────┐  │
│  │              Random Forest Classifier                     │  │
│  │   • 200 árboles de decisión                              │  │
│  │   • max_depth=20                                         │  │
│  │   • 98.55% accuracy                                      │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         │                                        │
│  ┌──────────────────────▼───────────────────────────────────┐  │
│  │                  Output Generation                        │  │
│  │   • Clase predicha (1 de 5 actividades)                 │  │
│  │   • Confianza (max probability)                          │  │
│  │   • Distribución de probabilidades                       │  │
│  │   • Frame anotado con landmarks                          │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         │                                        │
└─────────────────────────┼────────────────────────────────────────┘
                          │ JSON + Image
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    USUARIO (Visualización)                       │
│                                                                  │
│  • Video con skeleton overlay                                   │
│  • Actividad detectada + confianza                             │
│  • Distribución de probabilidades                              │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Pipeline de Procesamiento

```python
Frame (webcam) 
  → BGR to RGB conversion
  → MediaPipe Pose detection
  → Extract 33 landmarks × (x, y, z, visibility)
  → Compute geometric features (83 total)
  → StandardScaler normalization
  → PCA dimensionality reduction (83 → 16)
  → Random Forest prediction
  → Output: {class, confidence, probabilities}
  → Annotate frame with landmarks
  → Display results
```

---

## 3. Requisitos y Dependencias

### 3.1 Requisitos de Hardware

#### Servidor (Google Colab recomendado)

| Componente | Mínimo | Recomendado |
|------------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 4 GB | 8 GB |
| GPU | No requerida | No requerida |
| Almacenamiento | 500 MB | 1 GB |
| Red | 5 Mbps upload | 10+ Mbps upload |

#### Cliente (Usuario)

| Componente | Requisito |
|------------|-----------|
| Navegador | Chrome 90+, Firefox 88+, Edge 90+ |
| Webcam | 720p (1280×720) mínimo |
| Red | 3 Mbps download |
| JavaScript | Habilitado |

### 3.2 Dependencias de Software

#### Dependencias Principales

```txt
# Core ML & Vision
mediapipe==0.10.21          # Detección de pose
numpy==1.26.4               # Operaciones numéricas
opencv-python==4.10.0.84    # Procesamiento de video
scikit-learn==1.5.2         # Modelo Random Forest
joblib==1.4.2               # Serialización de modelos

# UI & Deployment
gradio==5.8.0               # Interfaz web interactiva

# Utilities
pandas==2.2.3               # Manipulación de datos
matplotlib==3.9.2           # Visualización (opcional)
protobuf==4.25.8            # Serialización MediaPipe
```

#### Versiones Críticas

⚠️ **IMPORTANTE:** Se debe usar **NumPy 1.26.4** (NO 2.x) para compatibilidad con MediaPipe 0.10.21. Los modelos fueron guardados con NumPy 1.x.

### 3.3 Archivos del Modelo

Los siguientes archivos **deben** estar disponibles para el despliegue:

| Archivo | Descripción | Tamaño Aprox. | Ubicación Original |
|---------|-------------|---------------|-------------------|
| `randomforest_model.pkl` | Modelo Random Forest entrenado | ~50 MB | `Entrega2/data/trained_models/` |
| `scaler.pkl` | StandardScaler para normalización | ~20 KB | `Entrega2/data/models/transformers/` |
| `pca.pkl` | PCA transformer (16 componentes) | ~15 KB | `Entrega2/data/models/transformers/` |
| `label_encoder.pkl` | Codificador de clases | ~2 KB | `Entrega2/data/models/transformers/` |

---

## 4. Procedimiento de Despliegue

### 4.1 Opción A: Despliegue en Google Colab (Recomendado)

#### Paso 1: Preparar el Entorno

```python
# Abrir Google Colab
# Archivo → Abrir cuaderno → Subir notebook
# Seleccionar: 07_gradio_webcam_demo.ipynb
```

#### Paso 2: Instalar Dependencias

```bash
# Ejecutar celda 1
# Instala MediaPipe, NumPy, OpenCV, Gradio
# Tiempo estimado: 60-90 segundos

!pip install mediapipe==0.10.21 numpy==1.26.4 protobuf==4.25.8 --upgrade --force-reinstall
!pip install opencv-python gradio matplotlib pandas scikit-learn -q
```

**Salida esperada:**
```
📦 INSTALANDO DEPENDENCIAS...
============================================================
Successfully installed mediapipe-0.10.21 numpy-1.26.4 ...
✅ Dependencias instaladas
⚠️  Ignorar warnings de compatibilidad NumPy/MediaPipe
```

#### Paso 3: Verificar Imports

```python
# Ejecutar celda 2
import gradio as gr
import cv2
import numpy as np
import mediapipe as mp
```

**Salida esperada:**
```
✅ MediaPipe: 0.10.21
✅ NumPy: 1.26.4
✅ OpenCV: 4.10.0.84
✅ Gradio: 5.8.0
```

#### Paso 4: Cargar Modelos

```python
# Ejecutar celda 3
# Se abrirá un diálogo para subir archivos
# Subir los 4 archivos en este orden:
```

1. `randomforest_model.pkl`
2. `scaler.pkl`
3. `pca.pkl`
4. `label_encoder.pkl`

**Proceso:**
- Navegar a `Entrega2/data/trained_models/` y `Entrega2/data/models/transformers/`
- Seleccionar archivos
- Esperar carga completa (barra de progreso)

**Salida esperada:**
```
📤 SUBIR ARCHIVOS DE MODELO
============================================================
✅ Archivos subidos: 4
   📦 randomforest_model.pkl (48.32 MB)
   📦 scaler.pkl (18.45 KB)
   📦 pca.pkl (12.78 KB)
   📦 label_encoder.pkl (1.23 KB)

🔍 VERIFICANDO ARCHIVOS REQUERIDOS:
   ✅ Modelo Random Forest: randomforest_model.pkl (49482.24 KB)
   ✅ Scaler (normalización): scaler.pkl (18.45 KB)
   ✅ PCA (reducción dimensionalidad): pca.pkl (12.78 KB)
   ✅ Label Encoder (clases): label_encoder.pkl (1.23 KB)

🤖 CARGANDO MODELO Y TRANSFORMADORES...
============================================================
   ✅ Random Forest cargado (98.55% accuracy)
   ✅ Scaler cargado
   ✅ PCA cargado (16 componentes)
   ✅ Label Encoder cargado (5 clases)

🏷️  ACTIVIDADES DETECTABLES:
   1. Caminar Hacia
   2. Caminar Regreso
   3. Girar
   4. Ponerse De Pie
   5. Sentarse

✅ MODELO LISTO PARA INFERENCIA
```

#### Paso 5: Configurar MediaPipe

```python
# Ejecutar celda 4
# Configura MediaPipe Pose con parámetros optimizados
```

**Configuración aplicada:**
```python
pose = mp_pose.Pose(
    static_image_mode=False,      # Video streaming
    model_complexity=1,            # Balance velocidad/precisión
    smooth_landmarks=True,         # Suavizado temporal
    min_detection_confidence=0.5,  # Umbral detección
    min_tracking_confidence=0.5    # Umbral tracking
)
```

#### Paso 6: Definir Funciones de Procesamiento

```python
# Ejecutar celda 5
# Define extract_landmarks, compute_geometric_features, predict_activity
```

**Funciones cargadas:**
- ✅ `extract_landmarks()`: MediaPipe → 132 coords
- ✅ `compute_geometric_features()`: 132 → 83 features
- ✅ `predict_activity()`: Pipeline completo de predicción

#### Paso 7: Crear Función Principal

```python
# Ejecutar celda 6
# Define process_frame() para Gradio
```

#### Paso 8: Crear Interfaz Gradio

```python
# Ejecutar celda 7
# Crea la interfaz con configuración de inputs/outputs
```

#### Paso 9: Lanzar Aplicación

```python
# Ejecutar celda 8
demo.launch(share=True, debug=False, show_error=True)
```

**Salida esperada:**
```
============================================================
🚀 LANZANDO APLICACIÓN GRADIO
============================================================

📹 Accede a tu cámara cuando el navegador lo solicite
🌐 Se generará una URL pública para compartir

Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://abc123xyz.gradio.live

This share link expires in 72 hours. For free permanent hosting...
```

#### Paso 10: Acceder a la Aplicación

1. **Copiar URL pública:** `https://abc123xyz.gradio.live`
2. **Abrir en navegador moderno** (Chrome/Firefox/Edge)
3. **Permitir acceso a webcam** cuando se solicite
4. **Compartir URL** con otros usuarios (válida 72 horas)

## 5. Configuración del Sistema

### 5.1 Parámetros de MediaPipe

```python
# Archivo: 07_gradio_webcam_demo.ipynb (Celda 4)

mp_pose.Pose(
    static_image_mode=False,        # Optimizado para video
    model_complexity=1,              # 0=lite, 1=full, 2=heavy
    smooth_landmarks=True,           # Suavizado Kalman filter
    min_detection_confidence=0.5,    # Umbral inicial (ajustar 0.3-0.7)
    min_tracking_confidence=0.5      # Umbral seguimiento (ajustar 0.3-0.7)
)
```

**Ajustes recomendados según escenario:**

| Escenario | `model_complexity` | `min_detection_confidence` |
|-----------|-------------------|---------------------------|
| Lighting bajo | 1 | 0.3 |
| Movimiento rápido | 0 (lite) | 0.4 |
| Alta precisión | 2 (heavy) | 0.7 |
| **Balanceado (default)** | **1** | **0.5** |

### 5.2 Parámetros de Gradio

```python
# Archivo: 07_gradio_webcam_demo.ipynb (Celda 7)

gr.Interface(
    fn=process_frame,
    inputs=gr.Image(
        sources=["webcam"],  # Solo webcam (no upload)
        type="numpy",
        streaming=True       # Modo tiempo real
    ),
    live=True,              # Actualización continua
    cache_examples=False,   # Sin cache (tiempo real)
    allow_flagging="never"  # Desactivar feedback
)
```

### 5.3 Configuración de Visualización

```python
# Archivo: 07_gradio_webcam_demo.ipynb (Celda 6)

# Colores según confianza
if confidence_pct >= 90:
    color = (0, 255, 0)   # Verde - Alta confianza
elif confidence_pct >= 75:
    color = (0, 255, 255) # Amarillo - Media confianza
else:
    color = (0, 0, 255)   # Rojo - Baja confianza
```

---

## 6. Pruebas y Validación

### 6.1 Checklist Pre-Despliegue

- [ ] ✅ Todas las dependencias instaladas correctamente
- [ ] ✅ 4 archivos de modelo cargados sin errores
- [ ] ✅ MediaPipe detecta landmarks en frame de prueba
- [ ] ✅ Pipeline de features produce 83 features
- [ ] ✅ Modelo predice clase válida (1-5)
- [ ] ✅ Gradio genera URL pública sin errores
- [ ] ✅ Webcam accesible desde navegador
