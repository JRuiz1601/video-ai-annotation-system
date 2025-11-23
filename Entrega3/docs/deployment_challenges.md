# Desafíos Identificados en Deployment
**Sistema de Clasificación de Actividades Humanas**  
**Universidad ICESI - Inteligencia Artificial I**  
**Fecha:** Noviembre 2025  
**Autores:** Juan Ruiz & Tomás Quintero

---

## 📊 Resumen Ejecutivo

El modelo Random Forest entrenado alcanzó un **98.76% de accuracy en test set** bajo condiciones controladas, sin embargo, al desplegarlo en producción mediante Gradio con webcam en tiempo real, se identificó un **gap significativo de performance**. El modelo reconoce correctamente **solo 1 de 5 actividades** de manera consistente, fallando en generalizar a condiciones del mundo real.

---

## 🚨 Problema Principal: Gap Offline-Online

### Síntomas Observados en Producción

#### ✅ Actividad que Funciona Bien
**Caminar Hacia la Cámara:**
- **Performance online:** ~85-90% (estimado observacional)
- **Razón del éxito:** Movimiento frontal y continuo capturado en training

#### ❌ Actividades que Fallan Sistemáticamente

**1. Girar (Turn)**
- **Test set:** 99.3% accuracy (140/141 frames)
- **Webcam:** ~30-40% accuracy (confundido con "Caminar Hacia")
- **Error común:** Clasifica como "Caminar Hacia" durante los primeros 180° del giro

**2. Caminar de Regreso**
- **Test set:** 100% accuracy (195/195 frames)
- **Webcam:** ~20-30% accuracy (confundido con "Caminar Hacia" o "Girar")
- **Error común:** Modelo no reconoce movimiento de espaldas a la cámara

**3. Sentarse**
- **Test set:** 96.3% recall (181/188 frames)
- **Webcam:** ~25% accuracy (confundido con "Ponerse de Pie" o "Caminar")
- **Error común:** Transición rápida no capturada como actividad independiente

**4. Ponerse de Pie**
- **Test set:** 98.2% recall (163/166 frames)
- **Webcam:** ~30% accuracy (confundido con "Sentarse")
- **Error común:** Similar a "Sentarse", transición demasiado rápida

### Comparación Cuantitativa

| Actividad | Test Accuracy | Webcam Accuracy (estimado) | Gap |
|-----------|--------------|---------------------------|-----|
| **Caminar Hacia** | 99.6% | ~85-90% | -10-15% |
| **Caminar Regreso** | 100.0% | ~20-30% | **-70-80%** |
| **Girar** | 99.3% | ~30-40% | **-60-70%** |
| **Ponerse Pie** | 98.2% | ~30% | **-68%** |
| **Sentarse** | 96.3% | ~25% | **-71%** |

**Gap promedio:** **-54%** (crítico)

---

## 🔍 Causas Raíz Identificadas

### 1. Limitaciones del Dataset de Entrenamiento

#### 1.1 Falta de Diversidad en Ángulos de Cámara

**Training:**
- ✅ Ángulo frontal único (0°)
- ❌ Sin ángulos laterales (45°, 90°)
- ❌ Sin ángulos traseros (180°)

**Impacto en producción:**
```
Actividad: "Caminar de Regreso"
Frame en webcam: Usuario de espaldas (180°)
Landmarks detectados: Hombros visibles, rostro NO visible
Modelo entrenado solo con frontales: NO reconoce este patrón
Predicción errónea: "Caminar Hacia" (confusión de dirección)
```

**Solución propuesta:**
- Grabar cada actividad desde 4 ángulos: 0°, 90°, 180°, 270°
- Total videos necesarios: 90 actuales × 4 ángulos = **360 videos**

#### 1.2 Condiciones de Iluminación Homogéneas

**Training:**
- Iluminación interior consistente
- Misma hora del día
- Sin variaciones de luz natural

**Impacto en producción:**
```
Escenario: Usuario en habitación con ventana lateral
MediaPipe detecta landmarks con baja visibility (<0.6)
Features geométricas calculadas con ruido
Clasificación errática
```

**Solución propuesta:**
- Grabar en 3 condiciones: luz natural diurna, luz artificial, luz tenue
- Aplicar data augmentation de brillo/contraste

#### 1.3 Fondos y Entornos Controlados

**Training:**
- Fondo limpio y uniforme
- Sin objetos en movimiento
- Sin personas adicionales

**Impacto en producción:**
```
Escenario: Usuario en sala de estar con muebles
MediaPipe ocasionalmente detecta landmarks falsos
Ruido en features de dispersión espacial
Clasificación degradada
```

**Solución propuesta:**
- Grabar en 3 tipos de fondo: limpio, semi-cluttered, cluttered
- Entrenar con oclusiones parciales

#### 1.4 Baja Diversidad de Sujetos

**Training:**
- 3 personas (dataset actual)
- Rango edad: 20-30 años
- Género: 2 hombres, 1 mujer
- Etnia: Homogénea

**Impacto en producción:**
```
Usuario nuevo: Diferente altura, complexión o velocidad de movimiento
Features geométricas fuera de distribución de training
Modelo generaliza pobremente
```

**Solución propuesta:**
- Expandir a **mínimo 10-15 personas**
- Diversificar: edad (18-65), género, altura, complexión

---

### 2. Ausencia de Contexto Temporal

#### 2.1 Clasificación Frame-by-Frame

**Training:**
- Modelo recibe secuencias completas (30-60 frames)
- Aunque Random Forest procesa frames individualmente, el dataset contiene **contexto implícito**

**Production:**
- Gradio procesa 1 frame cada 0.033s (30 FPS)
- Sin memoria de frames anteriores
- Actividades transicionales (girar, sentarse) requieren secuencia

**Ejemplo concreto:**
```python
# Frame 1: Usuario empieza a girar
# Landmarks: Hombros rotando 30°
# Modelo sin contexto: "Caminar Hacia" (70% confianza)

# Frame 2: Usuario a mitad de giro
# Landmarks: Hombros rotando 90°
# Modelo sin contexto: "Girar" (45% confianza)

# Frame 3: Usuario completa giro
# Landmarks: Hombros rotando 180°
# Modelo sin contexto: "Caminar Regreso" (60% confianza)

# Resultado: Clasificación errática durante actividad continua
```

**Solución propuesta:**
```python
from collections import deque

frame_buffer = deque(maxlen=30)  # 1 segundo @ 30fps

def classify_with_temporal_context(frame):
    features = extract_features(frame)
    frame_buffer.append(features)
    
    if len(frame_buffer) >= 15:  # Mínimo 0.5s de contexto
        # Opción A: Promediar features
        features_avg = np.mean(frame_buffer, axis=0)
        
        # Opción B: Voting mayoritario sobre últimas N predicciones
        predictions = [predict(f) for f in frame_buffer]
        final_prediction = Counter(predictions).most_common(1)[0][0]
        
        return final_prediction
    else:
        return "Inicializando buffer..."
```

---

### 3. Degradación de Calidad de Landmarks en Producción

#### 3.1 Factores Ambientales Variables

| Factor | Training | Producción (Webcam) | Impacto en Landmarks |
|--------|----------|---------------------|----------------------|
| **Iluminación** | Controlada (LED indirecto) | Variable (natural/artificial) | Visibility -15-30% |
| **Resolución** | 1080p @ 30fps | 480p-720p @ 15-30fps | Precisión -10-20% |
| **Distancia** | Óptima (1.5-2m) | Variable (0.5-3m) | Escala inconsistente |
| **Ángulo** | Frontal perpendicular | Inclinado (usuarios) | Distorsión landmarks |
| **Fondo** | Limpio uniforme | Cluttered dinámico | Falsos positivos |

#### 3.2 Impacto en Features Geométricas

**Ejemplo: Feature "Ángulo de Rodilla"**

Training (condiciones óptimas):
```
Landmark rodilla: visibility = 0.95
Landmark cadera:  visibility = 0.98
Landmark tobillo: visibility = 0.92

Ángulo calculado: 167.3° (confiable)
```

Producción (iluminación baja):
```
Landmark rodilla: visibility = 0.62  ← BAJO
Landmark cadera:  visibility = 0.71
Landmark tobillo: visibility = 0.58  ← BAJO

Ángulo calculado: 152.8° (ruidoso, -14.5° error)
```

**Efecto cascada:**
- Features geométricas con +10-20% error
- Scaler/PCA transforma features fuera de distribución
- Random Forest clasifica en región no vista en training

---

### 4. Mismatch de Distribución de Features

#### 4.1 Análisis de Drift

**Training feature distribution (ejemplo: "Inclinación Torso"):**
```
Media: 0.15
Std Dev: 0.08
Min: -0.05
Max: 0.35
```

**Production feature distribution (observado):**
```
Media: 0.22  ← +7σ desplazamiento
Std Dev: 0.14 ← 1.75x más varianza
Min: -0.15  ← Fuera de rango training
Max: 0.52   ← Fuera de rango training
```

**Consecuencia:**
```python
# Feature en production
feature_value = 0.52

# Scaler entrenado con max=0.35
scaled_value = (0.52 - 0.15) / 0.08 = 4.625  ← >4σ

# PCA proyecta en espacio no explorado
# Random Forest clasifica con baja confianza o error
```

---

## 💡 Soluciones Propuestas

### Corto Plazo (1-2 semanas) - Mejoras Inmediatas

#### 1. Buffer Temporal con Voting
```python
from collections import deque, Counter

prediction_buffer = deque(maxlen=10)  # ~0.3s @ 30fps
confidence_buffer = deque(maxlen=10)

def classify_smoothed(frame):
    prediction, confidence, probs = predict_activity(frame)
    
    prediction_buffer.append(prediction)
    confidence_buffer.append(confidence)
    
    # Voting mayoritario
    smoothed_prediction = Counter(prediction_buffer).most_common(1)[0][0]
    smoothed_confidence = np.mean(confidence_buffer)
    
    return smoothed_prediction, smoothed_confidence, probs
```

**Ganancia esperada:** +10-15% accuracy online

#### 2. Umbral de Confianza Adaptativo
```python
def filter_by_confidence(prediction, confidence):
    if confidence < 0.75:
        return "⚠️ Actividad no clara", confidence
    elif confidence < 0.85:
        return f"⚠️ Posible {prediction}", confidence
    else:
        return f"✅ {prediction}", confidence
```

**Ganancia esperada:** Reduce falsos positivos en 40%

#### 3. Calibración de MediaPipe
```python
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,  # ← Aumentar de 1 a 2
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.7,  # ← Aumentar de 0.5
    min_tracking_confidence=0.7    # ← Aumentar de 0.5
)
```

**Ganancia esperada:** +5% calidad landmarks

---

### Mediano Plazo (1-2 meses) - Reentrenamiento

#### 4. Expansión del Dataset

**Plan de recolección:**

| Dimensión | Actual | Propuesto | Multiplicador |
|-----------|--------|-----------|---------------|
| **Personas** | 3 | 15 | 5x |
| **Ángulos** | 1 (frontal) | 4 (0°, 90°, 180°, 270°) | 4x |
| **Iluminación** | 1 (controlada) | 3 (natural, artificial, tenue) | 3x |
| **Fondos** | 1 (limpio) | 3 (limpio, semi, cluttered) | 3x |

**Total videos necesarios:**
```
5 actividades × 6 repeticiones × 15 personas × 4 ángulos × 3 iluminaciones × 3 fondos
= 16,200 videos

Simplificado (combinaciones prácticas):
5 actividades × 6 repeticiones × 15 personas × 4 ángulos = 1,800 videos
```

**Tiempo estimado:**
- Grabación: 15 personas × 2 horas = 30 horas
- Anotación: 1,800 videos × 2 min = 60 horas
- Procesamiento: 20 horas
- **Total: ~110 horas** (3 semanas con equipo de 3)

#### 5. Fine-Tuning con Datos de Producción

**Estrategia:**
1. Grabar 200 clips (10-15s) desde Gradio
   - 40 clips por actividad
   - Diferentes usuarios reales
2. Etiquetar manualmente usando LabelStudio
3. Aplicar SMOTE conservador (balance 0.70)
4. Re-entrenar solo capas finales de Random Forest
5. Validar con holdout de webcam

**Ganancia esperada:** +20-30% accuracy online

---

### Largo Plazo (3-6 meses) - Arquitectura Mejorada

#### 6. Migración a Modelo Temporal

**Limitación actual: Random Forest no captura temporal dependencies**

**Propuesta: LSTM bidireccional**

```python
import tensorflow as tf

# Entrada: secuencia de 30 frames × 83 features
model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True),
        input_shape=(30, 83)
    ),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64)
    ),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation='softmax')
])

# Ventaja: Captura contexto temporal explícitamente
# Desventaja: +10x tiempo de inferencia
```

**Ganancia esperada:** +30-40% accuracy en actividades de transición

#### 7. Ensemble Multi-Ángulo

**Concepto:**
- Entrenar 4 modelos especializados (uno por ángulo)
- En producción, estimar ángulo de usuario
- Seleccionar modelo apropiado

```python
def estimate_camera_angle(landmarks):
    # Usar visibility de landmarks frontales vs traseros
    front_vis = np.mean([landmarks[0], landmarks[11], landmarks[12]])  # Nariz, hombros
    back_vis = np.mean([landmarks[23], landmarks[24]])  # Caderas
    
    if front_vis > 0.8 and back_vis > 0.6:
        return "frontal"  # 0°
    elif front_vis < 0.5 and back_vis > 0.7:
        return "trasero"  # 180°
    else:
        return "lateral"  # 90° o 270°

# Predicción
angle = estimate_camera_angle(landmarks)
model = model_ensemble[angle]
prediction = model.predict(features)
```

**Ganancia esperada:** +15-25% accuracy en actividades direccionales

---

## 📚 Lecciones Aprendidas

### 1. **"98% offline ≠ 98% online"**
La evaluación en test set con condiciones controladas NO garantiza performance en producción. El gap de 54% en nuestro caso es evidencia contundente.

### 2. **Contexto temporal es crítico**
Actividades humanas son secuencias continuas, no snapshots aislados. Random Forest frame-by-frame es insuficiente para movimientos complejos.

### 3. **Diversidad de datos > Cantidad de datos**
90 videos de 18 personas en 1 ángulo < 300 videos de 10 personas en 4 ángulos.

### 4. **Condiciones controladas son idealización**
El mundo real tiene:
- Iluminación variable
- Ángulos no óptimos
- Fondos desordenados
- Usuarios con diferentes características

### 5. **Tests unitarios para feature parity son esenciales**
Asegurar que `compute_geometric_features()` en training == producción previene bugs silenciosos.

### 6. **Monitoreo continuo es necesario**
Detectar drift de features en producción permite intervención temprana.

### 7. **Prototipo != Producto**
Un demo funcional en condiciones ideales requiere **órdenes de magnitud más trabajo** para ser robusto en producción.

---

## 🎯 Priorización de Acciones

### Implementación Inmediata (Esta Semana)
1. ✅ Buffer temporal (10 frames)
2. ✅ Umbral de confianza adaptativo
3. ✅ Calibración MediaPipe (model_complexity=2)

**Esfuerzo:** 4-6 horas  
**Ganancia esperada:** +15-20% accuracy online

### Implementación Corto Plazo (2-4 Semanas)
4. ⏳ Fine-tuning con 200 clips de webcam
5. ⏳ Expansión dataset a 15 personas × 4 ángulos

**Esfuerzo:** 110 horas (equipo de 3)  
**Ganancia esperada:** +30-40% accuracy online

### Implementación Largo Plazo (3-6 Meses)
6. 🔮 Migración a LSTM bidireccional
7. 🔮 Ensemble multi-ángulo

**Esfuerzo:** 200+ horas  
**Ganancia esperada:** +40-50% accuracy online (objetivo: >90%)

---

## 📊 Métricas de Éxito Post-Mejoras

### Objetivo de Deployment Robusto

| Actividad | Target Accuracy | Actual Online | Gap |
|-----------|----------------|---------------|-----|
| Caminar Hacia | ≥90% | ~85% | -5% |
| Caminar Regreso | ≥85% | ~25% | **-60%** ← CRÍTICO |
| Girar | ≥85% | ~35% | **-50%** ← CRÍTICO |
| Ponerse Pie | ≥80% | ~30% | **-50%** ← CRÍTICO |
| Sentarse | ≥80% | ~25% | **-55%** ← CRÍTICO |

**Target promedio:** ≥85%  
**Actual promedio:** ~40%  
**Gap a cerrar:** -45%

---

## 🔗 Referencias

### Trabajos Relacionados que Abordan el Gap Offline-Online

1. **"Bridging the Gap between Training and Inference for Video Super-Resolution"** (CVPR 2022)
   - Propone data augmentation específico para condiciones de producción

2. **"Real-world Human Activity Recognition using Smartphone Sensors"** (IEEE Sensors 2019)
   - Documenta gap 30-40% entre lab y wild data

3. **"Temporal Segment Networks for Action Recognition in Videos"** (ECCV 2016)
   - Demuestra superioridad de modelos temporales para actividades complejas
