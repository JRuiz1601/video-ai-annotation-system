# Sistema de Anotación de Video para Análisis de Actividades

#### -- Estado del Proyecto: Activo ✅

## Miembros del Equipo

|Nombre     |  Email   | 
|-----------|-----------------|
[Juan Esteban Ruiz](https://github.com/JRuiz1601)| juan.ruizome@u.icesi.edu.co |
|[Juan David Quintero](https://github.com/Juanda2005123) | juan.quintero@u.icesi.edu.co |
|[Tomas Quintero](https://github.com/tomasquin2003) | tomas.quintero@u.icesi.edu.co |

## Introducción/Objetivo del Proyecto
El propósito de este proyecto es desarrollar un sistema automatizado de clasificación de actividades humanas básicas utilizando análisis de coordenadas articulares extraídas mediante MediaPipe. El sistema identificará cinco actividades específicas: caminar hacia la cámara, caminar de regreso, girar, sentarse y ponerse de pie, con una precisión superior al 85% y capacidad de procesamiento en tiempo real. Este desarrollo contribuye al avance de sistemas de análisis de movimiento no invasivos aplicables en rehabilitación, deporte e investigación biomecánica.

### Metodologías Utilizadas
* Análisis Exploratorio de Datos (EDA)
* Aprendizaje Automático Supervisado
* Visualización de Datos
* Modelado Predictivo
* Procesamiento de Video en Tiempo Real
* Metodología CRISP-DM
* Validación Cruzada
* Feature Engineering
* SMOTE (Synthetic Minority Over-sampling Technique)
* PCA (Principal Component Analysis)

### Tecnologías
* Python 3.10
* MediaPipe 0.10.21 (Google)
* OpenCV
* Scikit-learn
* Random Forest
* Pandas, NumPy
* Matplotlib, Seaborn
* Jupyter Notebooks
* Gradio (Deployment)
* Git/GitHub

## Descripción del Proyecto
Este sistema utiliza la biblioteca MediaPipe de Google para extraer coordenadas de 33 puntos clave articulares de videos en tiempo real. A partir de estas coordenadas (x,y,z,visibility), se calculan 83 características geométricas (distancias, ángulos, ratios) que se reducen a 16 componentes mediante PCA. El modelo Random Forest entrenado clasifica automáticamente las actividades realizadas con **98.76% de accuracy** en test set.

**Fuentes de Datos**: 90 videos de 18 personas realizando las 5 actividades específicas, capturados en condiciones controladas (ángulo frontal, iluminación estable, fondo limpio). Dataset final de 7,352 frames balanceados mediante SMOTE.

**Análisis y Modelado**: 
- Extracción de 132 features crudas (33 landmarks × 4 coordenadas)
- Cálculo de 83 características geométricas (distancias, ángulos, ratios)
- Normalización StandardScaler
- Reducción PCA a 16 componentes (95.1% varianza explicada)
- Entrenamiento Random Forest con validación cruzada
- Bootstrap confidence intervals (IC 95%: [98.0%, 99.4%])

**Desafíos Principales**:
- ✅ **Resueltos en Offline:**
  - Variabilidad en movimientos humanos entre diferentes usuarios
  - Normalización para diferentes tipos de cuerpo y distancias de cámara
  - Overfitting (validado con test accuracy superior a validation)
  - Data leakage (verificado con tests forenses)
  
- ⚠️ **Identificados en Deployment:**
  - **Gap offline-online crítico:** 98.76% offline vs ~40% online
  - Falta de contexto temporal (frame-by-frame vs secuencias)
  - Condiciones no controladas (iluminación variable, ángulos diversos, fondos desordenados)
  - Baja diversidad de datos de entrenamiento (18 personas, 1 ángulo)

## Comenzando

### Opción 1: Clonar Repositorio (Para Desarrollo)

1. Clona este repositorio:
```bash
git clone https://github.com/JRuiz1601/video-ai-annotation-system.git
cd video-ai-annotation-system
```

2. Los datos sin procesar se mantienen en `Entrega1/data/videos/` dentro de este repositorio.
   *Los videos originales se almacenan localmente debido a su tamaño. Para obtener acceso, contacta al equipo.*

3. Los scripts de procesamiento/transformación de datos están en `Entrega1/src/data/`

4. Los notebooks de análisis están distribuidos por entregas (ver sección siguiente)

5. **Instalación y Setup**:
```bash
cd Entrega2/
pip install -r requirements.txt
```

### Opción 2: Probar el Sistema Desplegado (Recomendado para Demo)

⚠️ **IMPORTANTE:** El servidor Gradio en Google Colab se desconecta automáticamente después de **90 minutos de inactividad**. Si el link caduca, sigue estos pasos para re-lanzar el demo:

#### Instrucciones de Despliegue Rápido:

1️⃣ **Abre Google Colab:**
   - Ve a [https://colab.research.google.com/](https://colab.research.google.com/)
   - Click en "Archivo" → "Abrir notebook"
   - Selecciona la pestaña **"GitHub"**
   - Pega la URL: `https://github.com/JRuiz1601/video-ai-annotation-system`
   - Abre el notebook: 07_gradio_webcam_demo.ipynb

2️⃣ **Sube los Modelos Entrenados:**
   
   Cuando ejecutes la **Celda 3** (sección "SUBIR Y CARGAR MODELOS"), se abrirá un botón de carga. Debes subir estos **4 archivos** en orden:

   ```
   📦 Archivos requeridos (ubicados en el repositorio):
   
   1. randomforest_model.pkl       (Entrega2/data/trained_models/)
   2. scaler.pkl                    (Entrega2/data/models/transformers/)
   3. pca.pkl                       (Entrega2/data/models/transformers/)
   4. label_encoder.pkl             (Entrega2/data/models/transformers/)
   ```

   **Descarga directa desde GitHub:**
   - Opción A: Clona el repo y localiza los archivos
   - Opción B: [Descarga el ZIP del proyecto](https://github.com/JRuiz1601/video-ai-annotation-system/archive/refs/heads/main.zip) y extrae los archivos

3️⃣ **Ejecuta las Celdas en Orden:**

   - **Celda 1 (Instalación de Dependencias):**
     ```python
     # Al ejecutar, aparecerá una advertencia de NumPy
     ⚠️ ADVERTENCIA: "Restart session to use updated packages"
     
     👉 Click en "RESTART SESSION" (botón rojo que aparece)
     👉 Luego continúa con la siguiente celda
     ```
   
   - **Celdas 2-6:** Ejecuta normalmente (Shift + Enter en cada una)
   
   - **Celda 7 (Lanzar Aplicación):**
     ```python
     # Esta celda generará:
     ✅ URL Local:   http://127.0.0.1:7860
     ✅ URL Pública: https://xxxxx.gradio.live  ← COMPARTE ESTE LINK
     ```

4️⃣ **Usa la Aplicación:**
   - Click en la **URL Pública** (`https://xxxxx.gradio.live`)
   - Permite acceso a tu cámara cuando el navegador lo solicite
   - Colócate frente a la cámara (cuerpo completo visible)
   - Realiza alguna de las 5 actividades:
     - 🚶 Caminar hacia la cámara
     - 🚶‍♂️ Caminar de regreso
     - 🔄 Girar
     - 🧍 Ponerse de pie
     - 🪑 Sentarse

5️⃣ **Si el Link Expira:**
   - Vuelve a Colab
   - Ejecuta solo la **Celda 7** nuevamente
   - Obtén un nuevo link público

---

## Entregas y Documentación Principal

### 📂 Entrega 1 (13 octubre 2025) - ✅ Completa
* Documento de Fundamentos - Preguntas, metodología, métricas y EDA
* 01_setup_mediapipe.ipynb - Configuración inicial del pipeline
* 02_eda_inicial.ipynb - Análisis exploratorio de coordenadas
* **Resultados:** 90 videos procesados, 7,352 frames, 33 landmarks por frame

### 📂 Entrega 2 (27 octubre 2025) - ✅ Completa
* 03_data_preprocessing.ipynb - SMOTE, normalización, splits
* 04_feature_engineering.ipynb - 83 features geométricas, PCA
* 05_model_training.ipynb - Random Forest vs MLP
* 06_model_evaluation_realistic.ipynb - Evaluación en test, bootstrap
* Reporte de Evaluación Random Forest
* **Resultados:** 
  - Random Forest: **98.76% test accuracy** (12 errores de 967 frames)
  - MLP: 98.97% test accuracy (10 errores)
  - Selección: Random Forest (3x más rápido, interpretable)

### 📂 Entrega 3 (17 noviembre 2025) - ✅ Completa
* 07_gradio_webcam_demo.ipynb - **Demo en vivo con webcam**
* Desafíos de Deployment - Gap offline-online documentado
* Análisis de Impactos - Evaluación en contexto real
* Plan de Despliegue - Arquitectura y estrategia
* **Resultados:**
  - ✅ Deployment funcional en Gradio (Google Colab)
  - ⚠️ **Gap crítico:** 98.76% offline vs ~40% online
  - ⚠️ Solo "Caminar Hacia" funciona bien (~85%)
  - ❌ Otras 4 actividades: 20-40% accuracy online

---

## Estado del Proyecto por Entregas

| Entrega | Estado | Fecha Límite | Completitud | Métricas Clave |
|---------|--------|--------------|-------------|----------------|
| **Entrega 1** | ✅ Completa | 13 octubre 2025 | 100% | 90 videos, 7,352 frames |
| **Entrega 2** | ✅ Completa | 27 octubre 2025 | 100% | 98.76% test accuracy |
| **Entrega 3** | ✅ Completa | 17 noviembre 2025 | 100% | Deployment funcional, gap documentado |

---

## Métricas Alcanzadas del Proyecto

### ✅ Offline (Test Set Controlado)
- **Accuracy Global**: **98.76%** ✅ (objetivo: ≥85%)
- **F1-Score Promedio**: **98.76%** ✅ (objetivo: ≥80%)
- **F1-Score por Clase**: 
  - Caminar Hacia: 98.9% ✅
  - Caminar Regreso: 100.0% ✅
  - Girar: 99.6% ✅
  - Ponerse Pie: 97.9% ✅
  - Sentarse: 97.3% ✅
- **Latencia de Inferencia**: **0.003s** (<3ms) ✅ (objetivo: <100ms)
- **FPS Teórico**: **333 fps** ✅ (objetivo: ≥15 fps)

### ⚠️ Online (Webcam en Producción)
- **Accuracy Global Estimada**: **~40%** ❌ (objetivo: ≥85%)
- **Gap Offline-Online**: **-59%** (crítico)
- **Actividades Funcionales**: Solo "Caminar Hacia" (~85%)
- **Actividades Problemáticas**: 
  - Caminar Regreso: ~25% (-75% gap)
  - Girar: ~35% (-64% gap)
  - Sentarse: ~25% (-71% gap)
  - Ponerse Pie: ~30% (-68% gap)

### 🔍 Causas del Gap (Documentadas)
1. **Falta de ángulos diversos** (solo frontal en training)
2. **Iluminación homogénea** (training controlado)
3. **Fondos limpios** (sin clutter en training)
4. **Baja diversidad demográfica** (18 personas)
5. **Sin contexto temporal** (frame-by-frame)

### 🚀 Mejoras Propuestas
- **Corto plazo:** Buffer temporal (30 frames) → +15-20%
- **Mediano plazo:** 1,800 videos (15 personas × 4 ángulos) → +30-40%
- **Largo plazo:** LSTM temporal → +40-50%

---

## Estructura del Repositorio

```
video-ai-annotation-system/
│
├── Entrega1/                    # Fundamentos y EDA
│   ├── data/
│   │   └── videos/              # Videos originales (90 videos)
│   ├── notebooks/
│   │   ├── 01_setup_mediapipe.ipynb
│   │   └── 02_eda_inicial.ipynb
│   └── docs/
│       └── entrega1_fundamentos.md
│
├── Entrega2/                    # Modelado y Evaluación
│   ├── data/
│   │   ├── processed/           # X_train, y_train, etc.
│   │   ├── trained_models/      # randomforest_model.pkl
│   │   └── models/
│   │       └── transformers/    # scaler.pkl, pca.pkl, label_encoder.pkl
│   ├── notebooks/
│   │   ├── 03_data_preprocessing.ipynb
│   │   ├── 04_feature_engineering.ipynb
│   │   ├── 05_model_training.ipynb
│   │   └── 06_model_evaluation_realistic.ipynb
│   └── docs/
│       ├── random_forest_evaluation_report.md
│       └── deployment_plan.md
│
├── Entrega3/                    # Despliegue
│   ├── notebooks/
│   │   └── 07_gradio_webcam_demo.ipynb  ← DEMO EN VIVO
│   ├── docs/
│   │   ├── deployment_challenges.md     ← Gap offline-online
│   │   └── impact_analysis.md           ← Evaluación de impactos
│   └── video/
│       └── project_demo.mp4             ← Video de presentación (10 min)
│
└── README.md                    # Este archivo
```

---

## Lecciones Aprendidas

### 🎓 Técnicas
1. **"98% offline ≠ 98% online"** - La evaluación en test set controlado NO garantiza performance en producción
2. **Contexto temporal es crítico** - Actividades humanas son secuencias, no frames aislados
3. **Diversidad de datos > Cantidad** - 90 videos de 18 personas < 300 videos de 10 personas en 4 ángulos
4. **Prototipo ≠ Producto** - Demo funcional requiere órdenes de magnitud más trabajo para producción

### 🔬 Metodológicas
5. **Data leakage forense** - Verificar matemáticamente 0 duplicados entre train/val/test
6. **Bootstrap confidence intervals** - Validar robustez estadística (IC 95%: [98.0%, 99.4%])
7. **Tests unitarios de features** - Asegurar feature parity entre training y serving
8. **Monitoreo continuo** - Detectar drift de features en producción

---

## Publicaciones y Referencias

### Papers Implementados
1. **MediaPipe Pose** - Bazarevsky et al. (2020) - Google Research
2. **SMOTE** - Chawla et al. (2002) - Journal of Artificial Intelligence Research
3. **Random Forest** - Breiman (2001) - Machine Learning

### Trabajos Relacionados Citados
4. **"Real-world HAR using Smartphone Sensors"** (IEEE Sensors 2019) - Documenta gap 30-40% lab vs wild
5. **"Temporal Segment Networks for Action Recognition"** (ECCV 2016) - Superioridad modelos temporales
6. **"Bridging the Gap between Training and Inference"** (CVPR 2022) - Data augmentation para producción

---

## Contribuciones

Este proyecto está abierto a contribuciones. Si encuentras bugs, tienes sugerencias o quieres mejorar el sistema:

1. Fork el repositorio
2. Crea una rama (`git checkout -b feature/mejora`)
3. Commit tus cambios (`git commit -m 'Agrega nueva feature'`)
4. Push a la rama (`git push origin feature/mejora`)
5. Abre un Pull Request

**Áreas de mejora prioritarias:**
- 🔴 Solucionar gap offline-online (ver `deployment_challenges.md`)
- 🟡 Expandir dataset con más personas y ángulos
- 🟡 Implementar buffer temporal (30 frames)
- 🟢 Migrar a modelos temporales (LSTM)

---

**Universidad ICESI** | **Facultad de Ingeniería, Diseño y Ciencias Aplicadas** | **Inteligencia Artificial 1** | **2025-2**