# Sistema de Anotación de Video para Análisis de Actividades

Este proyecto es parte del curso **Inteligencia Artificial 1** de la Maestría en Inteligencia Artificial Aplicada, Universidad ICESI, Cali Colombia.

#### -- Estado del Proyecto: Activo

**Líder del Equipo: [Juan Esteban Ruiz](https://github.com/[github handle])(@slackHandle)**  

## Miembros del Equipo

|Nombre     |  Email   | 
|-----------|-----------------|
|[Juan Esteban Ruiz](https://github.com/JRuiz1601| juan.ruizome@u.icesi.edu.co |
|[Juan David Quintero](https://github.com/[github handle]| @juan.quintero |
|[Tomas Quintero](https://github.com/[github handle]) | @tomas.quintero |

## Contacto
* ¡Puedes contactar al líder del equipo o al instructor si tienes preguntas o estás interesado en contribuir!

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

### Tecnologías
* Python 3.9
* MediaPipe (Google)
* OpenCV
* Scikit-learn
* XGBoost
* Pandas, NumPy
* Matplotlib, Seaborn
* Jupyter Notebooks
* Git/GitHub

## Descripción del Proyecto
Este sistema utiliza la biblioteca MediaPipe de Google para extraer coordenadas de 33 puntos clave articulares de videos en tiempo real. A partir de estas coordenadas (x,y,z,visibility), se entrenan modelos de clasificación supervisada (SVM, Random Forest, XGBoost) para identificar automáticamente las actividades realizadas.

**Fuentes de Datos**: Videos de personas realizando las 5 actividades específicas, capturados desde diferentes ángulos, condiciones de iluminación y velocidades. Dataset objetivo de 250+ videos con 20+ participantes diversos.

**Análisis y Modelado**: Extracción de características temporales y espaciales, normalización para diferentes tipos de cuerpo y distancias de cámara, entrenamiento de múltiples algoritmos con optimización de hiperparámetros y validación cruzada.

**Desafíos Principales**:
- Variabilidad en movimientos humanos entre diferentes usuarios
- Diferentes velocidades de ejecución de actividades
- Oclusiones parciales y missing data en detección de pose
- Generalización a nuevos usuarios no vistos durante entrenamiento
- Requisitos de tiempo real (<100ms por clasificación)

## Comenzando
Instrucciones para contribuidores:

1. Clona este repositorio ([ayuda aquí](https://help.github.com/articles/cloning-a-repository/)):
```
git clone https://github.com/[usuario]/sistema-anotacion-video-ia.git
cd sistema-anotacion-video-ia
```

2. Los datos sin procesar se mantienen en [`Entrega1/data/videos/`](./Entrega1/data/videos/) dentro de este repositorio.
   *Los videos originales se almacenan localmente debido a su tamaño. Para obtener acceso, contacta al equipo.*

3. Los scripts de procesamiento/transformación de datos están en [`Entrega1/src/data/`](./Entrega1/src/data/)

4. Los notebooks de análisis están en [`Entrega1/notebooks/`](./Entrega1/notebooks/)

5. **Instalación y Setup**:
```
cd Entrega1/
pip install -r requirements.txt
```

Para setup detallado, consulta las [instrucciones de instalación](./Entrega1/docs/setup_instructions.md)

## Entregas y Documentación Principal

### 📂 Entrega 1 (Semana 12) - Fundamentos
* [Documento de Fundamentos](./Entrega1/docs/entrega1_fundamentos.md) - Preguntas, metodología, métricas y EDA
* [Setup MediaPipe](./Entrega1/notebooks/01_setup_mediapipe.ipynb) - Configuración inicial del pipeline
* [EDA Inicial](./Entrega1/notebooks/02_eda_inicial.ipynb) - Análisis exploratorio de coordenadas

### 📂 Entrega 2 (Semana 14) - Modelado
* [Entrenamiento de Modelos](./Entrega2/notebooks/model_training.ipynb)
* [Evaluación Comparativa](./Entrega2/notebooks/model_evaluation.ipynb)
* [Optimización de Hiperparámetros](./Entrega2/notebooks/hyperparameter_tuning.ipynb)

### 📂 Entrega 3 (Semana 17) - Despliegue
* [Sistema en Tiempo Real](./Entrega3/src/realtime_system.py)
* [Interfaz Gráfica](./Entrega3/src/gui_application.py)
* [Documentación Final](./Entrega3/docs/reporte_final.pdf)

## Estado del Proyecto por Entregas

| Entrega | Estado | Fecha Límite | Completitud |
|---------|--------|--------------|-------------|
| **Entrega 1** | ✅ Completa | 13 octubre 2025 | 100% |
| **Entrega 2** | 🔄 En Progreso | 27 octubre 2025 | 0% |
| **Entrega 3** | ⏳ Planificada | 17 noviembre 2025 | 0% |

## Métricas Objetivo del Proyecto

- **Accuracy Global**: ≥85%
- **F1-Score por Clase**: ≥80% para cada actividad
- **Latencia de Inferencia**: <100ms por video
- **FPS en Tiempo Real**: ≥15 fps
- **Robustez Cross-Usuario**: ≥80% con usuarios no vistos

---

**Universidad ICESI** | **Facultad de Ingeniería, Diseño y Ciencias Aplicadas** | **Inteligencia Artificial 1** | **2025-2**
