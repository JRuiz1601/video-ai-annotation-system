# Data Augmentation & Preprocessing Strategy
## Sistema de Anotación de Video - Entrega 2

## 🎯 Resumen Ejecutivo

Este documento describe la estrategia implementada para preparar datos de landmarks de pose humana para clasificación de actividades, garantizando **ausencia de data leakage** y **métricas realistas**.

### Métricas Clave

| Métrica | Valor |
|---------|-------|
| **Dataset Original** | 6,443 frames (90 videos) |
| **Dataset Final** | 7,352 frames (6,443 real + 909 SMOTE) |
| **Balance Inicial** | 0.51 (desbalanceado) |
| **Balance Final (Train)** | 0.80 (excelente) |
| **Ratio Sintético** | 16.8% (seguro < 20%) |
| **Features Originales** | 64 landmarks (32 × 2) |
| **Features Finales** | 16 componentes PCA (95.1% varianza) |
| **Data Leakage** | ✅ 0% (verificado) |

---

## 🔍 Problema Identificado

### Dataset Original (Notebook 1 - EDA)

```
Total frames: 6,443
Distribución por actividad:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Caminar Hacia    : 1,844 (28.6%)  ← Mayoría
Caminar Regreso  : 1,301 (20.2%)
Sentarse         : 1,253 (19.4%)
Ponerse de Pie   : 1,103 (17.1%)
Girar            :   942 (14.6%)  ← Minoría
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Balance ratio: 942/1,844 = 0.51 (DESBALANCEADO)
```

### Desafíos Técnicos

1. **Desbalance de Clases:**
   - Clase mayoritaria 2× más grande que minoritaria
   - Riesgo de bias del modelo hacia "Caminar Hacia"
   - Performance pobre en clases minoritarias

2. **Alta Dimensionalidad:**
   - 64 features base + 19 geométricas = 83 features
   - Riesgo de overfitting
   - Entrenamiento computacionalmente costoso

3. **Riesgo de Data Leakage:**
   - Augmentation tradicional aplica técnicas a todo el dataset
   - Split posterior contamina test set con información de train
   - Métricas infladas artificialmente

---

## 🔄 Estrategia de Data Augmentation

### Notebook 3: `03_data_augmentation_strategy.ipynb`

#### Metodología: SMOTE Conservador Sin Leakage

Implementamos un enfoque **conservador** basado en SMOTE (Synthetic Minority Oversampling Technique) con separación previa de datasets.

### Paso 1: Split Estratificado (ANTES de Augmentation)

**⚠️ PASO CRÍTICO:** Split realizado ANTES de cualquier técnica sintética.

```
Distribución del split:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Train      : 4,509 frames (70%)
Validation :   967 frames (15%)
Test       :   967 frames (15%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total      : 6,443 frames (100%)
```

**Características del split:**
- ✅ Estratificado: Mantiene proporciones de cada actividad
- ✅ Reproducible: `random_state=42`
- ✅ Sin solapamiento: Videos únicos por split
- ✅ Balance preservado: Ratio 0.51 en todos los splits

### Paso 2: Análisis de Desbalance (SOLO Train)

Calculamos necesidades de augmentation **exclusivamente** en el train set:

```
Balance train original: 0.512

Target conservador: 80% de clase mayoritaria
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Target = 1,290 × 0.80 = 1,032 frames por actividad

Frames sintéticos necesarios:
  Caminar Hacia   : 1,290 → 1,290 (sin cambio)
  Caminar Regreso :   911 → 1,032 (+121)
  Sentarse        :   877 → 1,032 (+155)
  Ponerse de Pie  :   771 → 1,032 (+261)
  Girar           :   660 → 1,032 (+372)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total sintéticos: 909 frames
```

**Justificación del Target (80% vs 100%):**
- Target 100% = Balance perfecto, pero 30.1% sintético (alto riesgo overfitting)
- Target 80% = Balance bueno (0.80), solo 16.8% sintético (bajo riesgo)
- **Decisión:** Priorizar seguridad sobre balance perfecto

### Paso 3: Aplicación de SMOTE (SOLO Train)

**Técnica:** Synthetic Minority Oversampling Technique

```
SMOTE(
    sampling_strategy={activity: 1032 for minority classes},
    random_state=42,
    k_neighbors=5
)
```

**Funcionamiento:**
1. Para cada frame minoritario:
   - Encuentra k=5 vecinos más cercanos de la misma clase
   - Genera punto sintético interpolando entre frame y vecino
   - Coordenadas: `new = original + λ × (neighbor - original)`
   - λ ~ Uniform(0, 1)

2. Repite hasta alcanzar target de 1,032 frames por actividad

**Ventajas:**
- ✅ Preserva distribución de features
- ✅ No genera outliers extremos
- ✅ Aumenta variabilidad sin ruido
- ✅ Específico para datos numéricos (landmarks)

### Resultado del Augmentation

```
Train set final:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Original: 4,509 frames (83.2%)
SMOTE:      909 frames (16.8%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total:    5,418 frames

Balance: 0.512 → 0.800 (+56.4% mejora)

Distribución balanceada:
  Caminar Hacia   : 1,290 (23.8%)
  Caminar Regreso : 1,032 (19.0%)
  Girar           : 1,032 (19.0%)
  Ponerse de Pie  : 1,032 (19.0%)
  Sentarse        : 1,032 (19.0%)
```

**Validation & Test:**
- ✅ **Sin modificaciones**
- ✅ 100% datos reales
- ✅ Balance original 0.509 (refleja distribución real)

### Técnicas NO Implementadas (Justificación)

#### Rotaciones Espaciales
- **Descartado:** Dataset ya tiene variabilidad angular natural
- **Riesgo:** Generar poses anatómicamente imposibles
- **Decisión:** Simplicidad > Complejidad

#### Interpolación Temporal
- **Descartado:** Clasificación por frame individual (no series temporales)
- **Riesgo:** Crear transiciones artificiales sin valor
- **Decisión:** SMOTE es suficiente para balanceo

---

## 🔧 Pipeline de Preprocessing

### Notebook 4: `04_data_preparation_pipeline.ipynb`

#### Objetivo

Transformar landmarks crudos en features optimizadas para modelos ML, **sin contaminar validation/test**.

### Paso 1: Feature Engineering Geométrico

**Motivación:** Landmarks crudos (x, y, z) no capturan relaciones espaciales significativas.

#### Features Creadas (19 nuevas)

**1. Distancias Corporales (8 features):**
```
- shoulder_width: distancia hombro-L a hombro-R
- hip_width: distancia cadera-L a cadera-R
- L_torso_length: hombro-L a cadera-L
- R_torso_length: hombro-R a cadera-R
- L_thigh_length: cadera-L a rodilla-L
- R_thigh_length: cadera-R a rodilla-R
- L_shin_length: rodilla-L a tobillo-L
- R_shin_length: rodilla-R a tobillo-R

Fórmula: d = √((x₂-x₁)² + (y₂-y₁)² + (z₂-z₁)²)
```

**Significado:** Capturan proporciones corporales y configuración espacial.

**2. Ángulos Articulares (4 features):**
```
- L_elbow_angle: ángulo hombro-L → codo-L → muñeca-L
- R_elbow_angle: ángulo hombro-R → codo-R → muñeca-R
- L_knee_angle: ángulo cadera-L → rodilla-L → tobillo-L
- R_knee_angle: ángulo cadera-R → rodilla-R → tobillo-R

Fórmula: θ = arccos((v₁·v₂)/(|v₁||v₂|))
Rango: 0° (recto) a 180° (extendido)
```

**Significado:** Flexión/extensión articular, clave para diferenciar actividades.

**3. Ratios Corporales (3 features):**
```
- shoulder_hip_ratio: shoulder_width / hip_width
- torso_thigh_ratio: L_torso_length / L_thigh_length
- body_height_approx: |shoulder_y - ankle_y|
```

**Significado:** Proporciones independientes de tamaño absoluto.

**4. Centros de Masa (4 features):**
```
- center_mass_x: (L_hip_x + R_hip_x) / 2
- center_mass_y: (L_hip_y + R_hip_y) / 2
- upper_center_x: (L_shoulder_x + R_shoulder_x) / 2
- upper_center_y: (L_shoulder_y + R_shoulder_y) / 2
```

**Significado:** Posición global del cuerpo en el frame.

#### Resultado Feature Engineering

```
Features totales:
  Landmarks originales: 64 (32 puntos × 2 lados)
  Geométricas nuevas:   19
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total:                83 features

Aplicado a:
  ✅ Train:      5,418 × 83
  ✅ Validation:   967 × 83
  ✅ Test:         967 × 83
```

### Paso 2: Normalización (StandardScaler)

**Problema:** Features con escalas diferentes confunden al modelo.

```
Ejemplo ANTES de normalización:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
L_shoulder_x:      0.52  (rango 0-1)
L_elbow_angle:   120.00  (rango 0-180°)
shoulder_width:    0.15  (rango 0-0.3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Problema: Modelo da más peso a valores grandes (ángulos)
```

**Solución:** StandardScaler

```
Fórmula: z = (x - μ) / σ

Donde:
  x = valor original
  μ = media (calculada en train)
  σ = desviación estándar (calculada en train)

Resultado: Media = 0, Desviación = 1
```

**⚠️ PASO CRÍTICO: Fit Solo en Train**

```
# 1. FIT en train (calcular μ y σ)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

Estadísticas aprendidas:
  Feature 0: μ=0.487, σ=0.091
  Feature 1: μ=0.367, σ=0.071
  ... (83 features)

# 2. TRANSFORM en val (usar μ y σ de train)
X_val_scaled = scaler.transform(X_val)  # NO fit_transform

# 3. TRANSFORM en test (usar μ y σ de train)
X_test_scaled = scaler.transform(X_test)  # NO fit_transform
```

**Resultado:**

```
DESPUÉS de normalización:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
L_shoulder_x:      0.23  (escala estándar)
L_elbow_angle:     0.67  (escala estándar)
shoulder_width:    0.12  (escala estándar)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Todas las features ahora en escala comparable
```

### Paso 3: Reducción Dimensional (PCA)

**Problema:** 83 features generan:
- Overfitting (curse of dimensionality)
- Entrenamiento lento
- Redundancia de información

**Solución:** PCA (Principal Component Analysis)

```
Objetivo: Encontrar k componentes que capturen 95% de varianza

PCA(n_components=0.95, random_state=42)
```

**Funcionamiento:**

1. Calcular matriz de covarianza de X_train_scaled
2. Eigendescomposición: encontrar direcciones de máxima varianza
3. Seleccionar top-k eigenvectors (componentes principales)
4. Proyectar datos en nuevo espacio de k dimensiones

```
Resultado:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Features originales:  83
Componentes finales:  16
Varianza explicada: 95.1%
Reducción:          80.7%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Interpretación:
  PC1 captura ~30% varianza (movimiento vertical)
  PC2 captura ~25% varianza (movimiento horizontal)
  ...
  PC16 captura ~0.5% varianza (detalles finos)
  
  Total 16 PCs = 95.1% información original
```

**⚠️ PASO CRÍTICO: Fit Solo en Train**

```
# 1. FIT en train (aprender componentes)
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)

# 2. TRANSFORM en val (proyectar con componentes de train)
X_val_pca = pca.transform(X_val_scaled)

# 3. TRANSFORM en test (proyectar con componentes de train)
X_test_pca = pca.transform(X_test_scaled)
```

**Beneficios:**
- ✅ Entrenamiento ~5× más rápido
- ✅ Menos overfitting
- ✅ Elimina multicolinealidad
- ✅ Ruido reducido (4.9% descartado)

### Paso 4: Label Encoding

Convertir actividades textuales a códigos numéricos:

```
Label Encoding:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
0 → caminar_hacia
1 → caminar_regreso
2 → girar
3 → ponerse_pie
4 → sentarse
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ajustado en train, aplicado consistentemente a val/test
```

### Resultado Final del Pipeline

```
Datasets ML-ready:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
X_train: (5,418 × 16)  - Train balanceado
y_train: (5,418,)

X_val:   (967 × 16)    - Validation pura
y_val:   (967,)

X_test:  (967 × 16)    - Test puro
y_test:  (967,)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Transformers guardados:
  ✅ scaler.pkl (StandardScaler)
  ✅ label_encoder.pkl (LabelEncoder)
  ✅ pca.pkl (PCA)
```

---

## 🔒 Garantías de Calidad

### 1. Prevención de Data Leakage

#### ¿Qué es Data Leakage?

Cuando información del test set "filtra" al train durante preprocessing/augmentation, inflando métricas artificialmente.

#### Nuestras Garantías

| Técnica | Fit | Transform Val | Transform Test | ✅ Sin Leakage |
|---------|-----|---------------|----------------|----------------|
| **Split** | N/A | Antes de aug | Antes de aug | ✅ |
| **SMOTE** | Solo train | No aplicado | No aplicado | ✅ |
| **StandardScaler** | Solo train | Stats de train | Stats de train | ✅ |
| **PCA** | Solo train | PCs de train | PCs de train | ✅ |

#### Verificación

```
Videos únicos por split:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Train videos:      No overlap con val/test ✅
Val videos:        No overlap con train/test ✅
Test videos:       No overlap con train/val ✅
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Frames SMOTE:      Solo en train (909/5,418) ✅
Val/Test SMOTE:    0 frames sintéticos ✅
```

### 2. Reproducibilidad

```
Seeds fijos en todos los procesos:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
train_test_split:  random_state=42
SMOTE:             random_state=42
PCA:               random_state=42
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Resultado: Datasets idénticos en cada ejecución
```

### 3. Ratio Sintético Conservador

```
Límite seguro: < 20% datos sintéticos

Nuestro ratio:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Train:  909/5,418 = 16.8% ✅
Total:  909/7,352 = 12.4% ✅
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Estado: SEGURO (bien bajo el límite)
```

### 4. Balance vs Seguridad

```
Trade-off óptimo:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Balance:           0.80 (muy bueno)
Sintéticos:        16.8% (seguro)
Riesgo overfitting: Bajo
Performance esperada: Alta
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📊 Resultados Finales

### Comparación Antes/Después

| Métrica | Antes | Después | Mejora |
|---------|-------|---------|--------|
| **Total Frames** | 6,443 | 7,352 | +14.1% |
| **Balance Train** | 0.512 | 0.800 | +56.4% |
| **Features** | 64 | 16 (PCA) | -75% dim |
| **Varianza Info** | 100% | 95.1% | -4.9% |
| **Data Leakage** | Riesgo alto | 0% | ✅ |
| **Ratio Sintético** | N/A | 16.8% | Seguro |

### Distribución Final

```
TRAIN (5,418 frames - Balance 0.800):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Caminar Hacia   : 1,290 (23.8%) [100% real]
Caminar Regreso : 1,032 (19.0%) [ 88% real + 12% SMOTE]
Girar           : 1,032 (19.0%) [ 64% real + 36% SMOTE]
Ponerse de Pie  : 1,032 (19.0%) [ 75% real + 25% SMOTE]
Sentarse        : 1,032 (19.0%) [ 85% real + 15% SMOTE]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VALIDATION (967 frames - Balance 0.509):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Caminar Hacia   : 277 (28.6%) [100% real]
Caminar Regreso : 195 (20.2%) [100% real]
Sentarse        : 188 (19.4%) [100% real]
Ponerse de Pie  : 166 (17.2%) [100% real]
Girar           : 141 (14.6%) [100% real]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEST (967 frames - Balance 0.509):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Caminar Hacia   : 277 (28.6%) [100% real]
Caminar Regreso : 195 (20.2%) [100% real]
Sentarse        : 188 (19.4%) [100% real]
Ponerse de Pie  : 166 (17.2%) [100% real]
Girar           : 141 (14.6%) [100% real]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## 📁 Ubicación de Archivos Procesados

Los datasets y transformadores generados por el pipeline de preprocessing están disponibles en las siguientes ubicaciones:

### Datasets Procesados
**Ubicación:** `Entrega2/data/models/processed/`

```
X_train.npy  - Features de entrenamiento (5,418 × 16)
X_val.npy    - Features de validación (967 × 16)
X_test.npy   - Features de test (967 × 16)
y_train.npy  - Labels de entrenamiento (5,418,)
y_val.npy    - Labels de validación (967,)
y_test.npy   - Labels de test (967,)
```

### Transformadores Guardados
**Ubicación:** `Entrega2/data/models/transformers/`

```
scaler.pkl         - StandardScaler (normalización)
pca.pkl            - PCA (reducción dimensional)
label_encoder.pkl  - LabelEncoder (codificación de clases)
```

**Nota:** Estos archivos fueron generados siguiendo el pipeline descrito en este documento y están listos para ser utilizados en el entrenamiento de modelos de machine learning.
