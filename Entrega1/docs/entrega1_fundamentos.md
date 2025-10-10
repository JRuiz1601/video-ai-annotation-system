# Entrega 1: Fundamentos del Proyecto
**Sistema de Anotación de Video para Análisis de Actividades**

*Inteligencia Artificial 1 | Universidad ICESI | Semestre 2025-2*  


## 📋 Tabla de Navegación

| # | Sección | Contenido Principal | 
|---|---------|---------------------|
| **1** | **[Preguntas de Investigación](#1-preguntas-de-investigación)** | Pregunta principal, secundarias y justificación del problema | 
| **2** | **[Tipo de Problema](#2-tipo-de-problema-y-dominio)** | Clasificación multiclase, desafíos técnicos y complejidad |
| **3** | **[Metodología CRISP-DM](#3-metodología-crisp-dm-adaptada)** | Fases por entrega y estrategias de validación |
| **4** | **[Métricas de Evaluación](#4-métricas-de-evaluación)** | Criterios de éxito, protocolo y matriz de confusión | 
| **5** | **[Datos Recolectados](#5-datos-recolectados-y-eda-inicial)** | Dataset actual, EDA y análisis de calidad | 
| **6** | **[Ampliación de Datos](#6-estrategias-para-ampliar-el-dataset)** | Plan de escalamiento y data augmentation |
| **7** | **[Aspectos Éticos](#7-análisis-de-aspectos-éticos)** | Privacidad, equidad, transparencia y uso responsable |
| **8** | **[Próximos Pasos](#8-próximos-pasos-y-cronograma)** | Cronograma detallado y responsabilidades del equipo | 



**Recomendación**: Usa la **versión compacta** ya que mantiene toda la información importante pero es más fácil de leer y navegar. La columna de "Estado" también es útil para mostrar el progreso.

¿Cuál de estas versiones prefieres para el documento final? O ¿quieres que ajuste algún aspecto específico?           

## 1. Preguntas de Investigación

### 1.1 Pregunta Principal
**¿Cómo desarrollar un sistema automatizado de clasificación de actividades humanas básicas (caminar hacia la cámara, caminar de regreso, girar, sentarse, ponerse de pie) utilizando análisis de coordenadas articulares extraídas mediante MediaPipe, que sea preciso (>85% accuracy), eficiente (<100ms por clasificación) y robusto a variaciones individuales?**

### 1.2 Preguntas Secundarias Específicas

#### P1: Características Discriminativas
**¿Qué coordenadas articulares, ángulos y características de movimiento temporal son más relevantes para distinguir entre las cinco actividades objetivo?**
- Hipótesis: Movimientos de cadera, rodillas y tobillos serán más discriminativos
- Métricas: Feature importance, análisis de correlación
- Validación: Comparación de accuracy con/sin características específicas

#### P2: Normalización Cross-Usuario  
**¿Qué técnicas de preprocesamiento son más efectivas para manejar diferencias en tamaño corporal, distancia a la cámara y velocidad de movimiento entre usuarios?**
- Hipótesis: Normalización por altura y centro de masa mejorará generalización
- Métrica: Accuracy en validación leave-one-person-out
- Validación: Performance con usuarios de diferentes complexiones físicas

#### P3: Selección Óptima de Algoritmos
**¿Cuáles algoritmos de aprendizaje supervisado ofrecen el mejor balance entre precisión de clasificación, velocidad de inferencia y robustez para implementación en tiempo real?**
- Candidatos: SVM, Random Forest, XGBoost, LSTM
- Métricas: Accuracy vs Latencia vs Robustez
- Validación: Testing en diferentes condiciones de hardware

### 1.3 Justificación e Importancia

#### Relevancia Académica
- **Interdisciplinariedad**: Integra visión por computador, ML y biomecánica
- **Desafío Técnico**: Series temporales multidimensionales con alta variabilidad
- **Metodología**: Aplicación rigurosa de CRISP-DM en problema real

#### Aplicaciones Prácticas
- **Rehabilitación Física**: Monitoreo automático de ejercicios terapéuticos y progreso
- **Análisis Deportivo**: Evaluación objetiva de técnicas de movimiento
- **Gerontología**: Detección de caídas y monitoreo de actividad en adultos mayores
- **Investigación Biomecánica**: Análisis cuantitativo de patrones de movimiento

#### Impacto Tecnológico
- **Accesibilidad**: Hardware común (cámaras web) vs sensores especializados ($1000+)
- **Escalabilidad**: Deployable en múltiples contextos sin infraestructura adicional
- **Innovación**: Avance en sistemas no invasivos de análisis de movimiento

---

## 2. Tipo de Problema y Dominio

### 2.1 Clasificación Formal del Problema

#### Tipo Principal
- **Problema**: Clasificación Multiclase Supervisada
- **Clases**: 5 actividades mutuamente excluyentes
  1. Caminar hacia la cámara
  2. Caminar de regreso (alejándose)
  3. Girar (rotación corporal 360°)
  4. Sentarse (de pie a sentado)
  5. Ponerse de pie (de sentado a pie)
- **Naturaleza**: Categórica nominal (sin orden inherente)

#### Modalidad de Datos
- **Entrada**: Series temporales multivariadas
- **Dimensiones**: 33 landmarks × 4 coordenadas (x,y,z,visibility) × T frames
- **Frecuencia**: 30 FPS promedio
- **Duración**: Secuencias variables de 10-45 segundos

#### Dominio Específico
- **Campo Principal**: Computer Vision + Human Activity Recognition (HAR)
- **Sub-dominios**: 
  - Human Pose Estimation
  - Time Series Classification  
  - Real-time Video Processing
  - Biomechanical Movement Analysis

### 2.2 Desafíos Técnicos Principales

#### 2.2.1 Variabilidad Inter-Personal
- **Problema**: Diferencias en altura (1.50m-1.90m), peso, proporciones corporales
- **Impacto**: Mismas actividades → patrones de coordenadas muy diferentes
- **Ejemplo**: "Sentarse" varía según longitud de piernas y altura de silla
- **Mitigación**: Normalización por altura corporal y centro de masa

#### 2.2.2 Variabilidad Temporal
- **Problema**: Actividades ejecutadas a velocidades 0.5x-2x normal
- **Impacto**: Secuencias de duración muy variable para misma actividad
- **Ejemplo**: "Girar" puede tomar 2-8 segundos según persona
- **Mitigación**: Features de velocidad y normalización temporal

#### 2.2.3 Oclusiones y Missing Data
- **Problema**: MediaPipe no detecta todos los landmarks en todos los frames
- **Frecuencia**: 5-15% de frames con landmarks faltantes
- **Causas**: Ángulos extremos, ropa holgada, mala iluminación
- **Mitigación**: Interpolación temporal y modelos robustos a missing data

#### 2.2.4 Ambigüedad en Transiciones
- **Problema**: Momentos donde una actividad transiciona gradualmente a otra
- **Ejemplo**: Final de "caminar hacia" → inicio de "girar"
- **Impacto**: Etiquetas ambiguas en 1-2 segundos de transición
- **Mitigación**: Segmentación cuidadosa con buffers temporales

#### 2.2.5 Generalización Cross-Usuario
- **Problema**: Modelos sobre-especializados en usuarios de entrenamiento
- **Riesgo**: 95% accuracy en train, 60% en nuevos usuarios
- **Causas**: Overfitting a patrones específicos de movimiento
- **Mitigación**: Validación leave-one-person-out obligatoria

### 2.3 Complejidad Computacional

#### Durante Entrenamiento
- **Datos**: O(N × D × T) donde N=videos, D=66 features, T=frames promedio
- **Modelos**: Variable (SVM: O(N²), RF: O(N log N), XGBoost: O(N log N))
- **Estimación**: ~1-3 horas para dataset completo en hardware estándar

#### Durante Inferencia (Crítico)
- **Requisito**: <100ms por video de 3-5 segundos
- **Componentes**: MediaPipe (60ms) + Clasificación (40ms) < 100ms total
- **Optimizaciones**: Feature selection, model compression, paralelización

---

## 3. Metodología CRISP-DM Adaptada

### 3.1 Fases CRISP-DM y Entregas

#### Entrega 1 (Semana 12) - **Fundación**
**Fases CRISP-DM: 1-2 + inicio 3**

**✅ Fase 1: Comprensión del Negocio** 
- Objetivos de negocio definidos
- Criterios de éxito establecidos (>85% accuracy, <100ms)
- Evaluación de recursos y riesgos
- Definición de métricas de desempeño

**✅ Fase 2: Comprensión de los Datos**
- Identificación de fuentes de datos (videos + MediaPipe)
- Recolección inicial: 50+ videos balanceados
- Descripción de formato y estructura
- EDA preliminar de coordenadas articulares

**🔄 Fase 3: Preparación de Datos (Inicio)**
- Pipeline básico de extracción MediaPipe
- Análisis de calidad de detección
- Identificación de problemas de datos
- Estrategias de limpieza definidas

#### Entrega 2 (Semana 14) - **Modelado**  
**Fases CRISP-DM: 3-4 + inicio 5**

**🎯 Fase 3: Preparación de Datos (Completa)**
- Dataset completo: 200+ videos balanceados
- Feature engineering: velocidades, ángulos, distancias
- Normalización cross-usuario implementada
- División train/validation/test (70/15/15)

**🎯 Fase 4: Modelado**
- Entrenamiento de múltiples algoritmos (SVM, RF, XGBoost)
- Optimización de hiperparámetros (Grid Search)
- Feature selection para eficiencia
- Ensemble methods para robustez

**🎯 Fase 5: Evaluación (Inicio)**
- Métricas detalladas por modelo
- Validación cruzada y leave-one-person-out
- Análisis de matriz de confusión
- Selección de modelo final

#### Entrega 3 (Semana 17) - **Despliegue**
**Fases CRISP-DM: 5-6 completas**

**🎯 Fase 5: Evaluación (Completa)**
- Evaluación final en test set
- Testing con usuarios reales
- Análisis de casos de falla
- Validación de métricas de negocio

**🎯 Fase 6: Despliegue**
- Sistema en tiempo real funcional
- Interfaz gráfica para usuarios finales
- Documentación técnica completa
- Plan de mantenimiento y actualizaciones

### 3.2 Estrategias de Validación

#### Validación Técnica
- **K-Fold Cross-Validation**: k=5 para métricas estables
- **Leave-One-Person-Out**: Validación de generalización crítica
- **Temporal Cross-Validation**: Train en sesiones anteriores, test en posteriores
- **Stratified Sampling**: Mantener balance de clases en todos los splits

#### Validación de Negocio
- **A/B Testing**: Comparar con análisis manual
- **User Acceptance Testing**: Feedback de usuarios finales
- **Performance Benchmarking**: Comparar con sistemas existentes
- **Edge Case Testing**: Condiciones extremas de uso

---

## 4. Métricas de Evaluación

### 4.1 Métricas Primarias (Criterios de Aprobación)

| Métrica | Objetivo | Justificación | Método de Medición |
|---------|----------|---------------|-------------------|
| **Accuracy Global** | **≥85%** | Métrica estándar para clasificación balanceada | Validación cruzada k=5 |
| **F1-Score por Clase** | **≥80%** cada actividad | Balance precisión-recall por actividad | Macro-average de 5 clases |
| **Latencia de Inferencia** | **<100ms** por video | Requisito tiempo real crítico | Promedio 100 inferencias |
| **FPS en Tiempo Real** | **≥15 fps** | Interactividad fluida necesaria | Test con webcam en vivo |

### 4.2 Métricas Secundarias (Objetivos Deseables)

#### Métricas de Robustez
- **Cohen's Kappa**: ≥0.80 (acuerdo casi perfecto)
- **Precision promedio**: ≥82% (minimizar falsos positivos)
- **Recall promedio**: ≥78% (minimizar falsos negativos)
- **Robustez cross-usuario**: ≥80% en leave-one-person-out

#### Métricas de Eficiencia
- **Uso de memoria**: <2GB RAM durante inferencia
- **Uso de CPU**: <70% de un core durante operación
- **Tamaño del modelo**: <100MB para deployment
- **Tiempo de carga**: <5 segundos para inicialización

#### Métricas de Usabilidad
- **Tiempo de setup**: <10 minutos para usuario final
- **Tasa de error del usuario**: <2 errores por sesión de 30min
- **System Usability Scale**: ≥70/100 (Above Average)

### 4.3 Protocolo de Evaluación Riguroso

#### División de Datos
```
Dataset Total (250+ videos)
├── Train Set (70% = ~175 videos)
│   ├── Para entrenamiento de modelos
│   └── Hyperparameter tuning con validación cruzada
├── Validation Set (15% = ~38 videos)  
│   ├── Selección de modelo final
│   └── Early stopping y regularización
└── Test Set (15% = ~37 videos)
    ├── SOLO evaluación final
    └── Reportar estas métricas únicamente
```

#### Condiciones de Evaluación
- **Hardware Estándar**: Laptop 8GB RAM, CPU i5, sin GPU
- **Software**: Python 3.9, versiones específicas en requirements.txt
- **Ambiente**: Sin optimizaciones específicas de hardware
- **Usuarios**: Personas no involucradas en desarrollo del modelo

### 4.4 Análisis de Confusión Esperado

#### Matriz de Confusión Objetivo
```
                 Predicho
              C1  C2  G   S   P   
Real    C1   >85  <3  <5  <4  <3
        C2   <3 >85  <5  <4  <3  
        G    <4  <4 >85  <3  <4
        S    <3  <2  <3 >85  <7
        P    <2  <2  <3  <8 >85
```
*C1=Caminar hacia, C2=Caminar regreso, G=Girar, S=Sentarse, P=Ponerse pie*

#### Confusiones Esperadas y Aceptables
- **Sentarse ↔ Ponerse de pie**: Hasta 8% confusión (actividades inversas)
- **Caminar hacia ↔ Girar**: Hasta 5% confusión (transiciones)
- **Caminar ↔ Caminar**: Hasta 3% confusión (similar patrón de piernas)

---

## 5. Datos Recolectados y EDA Inicial

### 5.1 Dataset Actual (Estado: 10 octubre 2025)

#### Composición por Actividad
| Actividad | Videos Capturados | Participantes Únicos | Duración Promedio | Estado Calidad |
|-----------|-------------------|---------------------|-------------------|----------------|
| **Caminar hacia** | 12 videos | 6 personas | 18.5 segundos | ✅ Excelente |
| **Caminar regreso** | 10 videos | 5 personas | 16.8 segundos | ✅ Excelente |
| **Girar** | 11 videos | 6 personas | 12.3 segundos | ✅ Buena |
| **Sentarse** | 9 videos | 5 personas | 8.7 segundos | ⚠️ Necesita más datos |
| **Ponerse de pie** | 10 videos | 5 personas | 6.2 segundos | ⚠️ Muy cortos |
| **TOTAL** | **52 videos** | **8 personas únicas** | **12.5 seg promedio** | **Status: 52/50 ✅** |

#### Características de Participantes
- **Género**: 4 mujeres, 4 hombres (balance perfecto)
- **Edad**: 22-45 años (promedio 28.5 años)  
- **Altura**: 1.58m - 1.83m (buena variabilidad)
- **Complexión**: Delgada (3), Media (4), Robusta (1)
- **Diversidad**: 2 personas con lentes, 1 con limitación menor de movilidad

### 5.2 Especificaciones Técnicas Implementadas

#### Configuración de Captura
- **Resolución**: 1280x720 (cumple mínimo)
- **FPS**: 30 frames/segundo (estándar)
- **Formato**: MP4 con codec H.264
- **Duración**: 6-35 segundos (variable según actividad)
- **Tamaño promedio**: 2.3MB por video

#### Setup MediaPipe Implementado
- **Modelo**: mediapipe.solutions.pose
- **Complejidad**: 1 (balance velocidad-precisión)
- **Confianza detección**: 0.7 mínima
- **Confianza tracking**: 0.5 mínima
- **Landmarks extraídos**: 33 puntos × (x,y,z,visibility) = 132 valores/frame

### 5.3 Análisis Exploratorio de Datos (EDA)

#### 5.3.1 Calidad de Detección
```
Estadísticas de Detección MediaPipe:
├── Tasa de detección global: 94.3%
├── Frames con landmarks completos: 89.7%
├── Landmarks con alta confianza (>0.8): 92.1%
└── Videos problemáticos: 3/52 (5.8%)
```

#### 5.3.2 Patrones por Actividad Identificados

**Caminar Hacia/Regreso**
- **Patrón distintivo**: Alternancia rodillas, coordenada Z variable
- **Duración típica**: 15-25 segundos
- **Landmarks clave**: Rodillas (25,26), Tobillos (27,28), Pies (29-32)

**Girar**
- **Patrón distintivo**: Rotación gradual de hombros y caderas
- **Duración típica**: 8-18 segundos  
- **Landmarks clave**: Hombros (11,12), Caderas (23,24)

**Sentarse/Ponerse de Pie**
- **Patrón distintivo**: Cambio abrupto en coordenada Y de caderas
- **Duración típica**: 4-12 segundos
- **Landmarks clave**: Caderas (23,24), Rodillas (25,26)

#### 5.3.3 Visualizaciones Creadas

1. **Distribución de Duraciones**: Histograma por actividad
2. **Trayectorias 3D**: Coordenadas de landmarks clave en el tiempo
3. **Heatmap de Correlación**: Entre diferentes landmarks
4. **Análisis de Velocidad**: Velocidad promedio por articulación
5. **Detección Missing**: Porcentaje de landmarks faltantes por frame

### 5.4 Problemas Identificados y Soluciones

#### Problemas Actuales
1. **Actividades cortas**: "Ponerse de pie" muy rápido (6.2s promedio)
2. **Desbalance leve**: Menos datos para "sentarse" 
3. **Transiciones**: Algunos videos incluyen múltiples actividades
4. **Iluminación**: 3 videos con detección sub-óptima (<85%)

#### Soluciones Implementadas
1. **Captura extendida**: Pedir actividades más lentas y deliberadas
2. **Sesiones adicionales**: Enfocar en actividades sub-representadas
3. **Segmentación manual**: Clips puros de 1 actividad únicamente
4. **Control de calidad**: Rechazar videos con <90% detección

---

## 6. Estrategias para Ampliar el Dataset

### 6.1 Plan de Escalamiento

#### Meta por Entrega
| Entrega | Videos Objetivo | Participantes | Horas de Video | Status |
|---------|----------------|---------------|----------------|--------|
| **Entrega 1** | 50+ videos | 8-10 personas | ~1.2 horas | ✅ 52/50 |
| **Entrega 2** | 200+ videos | 15-20 personas | ~4.5 horas | 🎯 Planificado |
| **Entrega 3** | 250+ videos | 20+ personas | ~5.8 horas | 🎯 Objetivo final |

#### Distribución Balanceada Objetivo
```
Por Actividad (Entrega 2):
├── Caminar hacia: 45 videos (9 personas × 5 repeticiones)
├── Caminar regreso: 45 videos (9 personas × 5 repeticiones)  
├── Girar: 40 videos (8 personas × 5 repeticiones)
├── Sentarse: 35 videos (7 personas × 5 repeticiones)
└── Ponerse pie: 35 videos (7 personas × 5 repeticiones)
Total: 200 videos balanceados
```

### 6.2 Estrategias de Recolección Activa

#### 6.2.1 Crowdsourcing Universitario
**Colaboración con Otros Grupos**
- **Acción**: Intercambio de datos con 3-4 grupos del curso
- **Contribución**: 20 videos nuestros ↔ 20 videos de cada grupo
- **Beneficio**: +60-80 videos adicionales
- **Timeline**: Semana 13 (coordinación activa)

**Redes Sociales Académicas**
- **Plataformas**: WhatsApp grupos universitarios, Instagram stories
- **Incentivo**: Participación voluntaria + créditos en agradecimientos
- **Expectativa**: 5-8 participantes adicionales
- **Timeline**: Semanas 13-14

#### 6.2.2 Variación de Condiciones
**Sesiones de Captura Programadas**
- **Ubicaciones**: 3 espacios diferentes (interior, exterior, laboratorio)
- **Horarios**: Mañana, tarde, noche (diferentes iluminaciones)
- **Vestimenta**: Ropa ajustada vs holgada vs formal
- **Velocidades**: Lenta (0.5x), Normal (1x), Rápida (1.5x)

**Casos Edge Intencionados**
- **Distancias**: 1.5m, 3m, 4.5m de la cámara
- **Ángulos**: Frontal, diagonal 30°, diagonal 45°
- **Interferencias**: Con objetos parcialmente oclusivos
- **Participantes Diversos**: Diferentes capacidades motoras

### 6.3 Estrategias de Data Augmentation

#### 6.3.1 Transformaciones Geométricas
```
Augmentations Implementadas:
├── Scaling: ±10% en coordenadas x,y
├── Translation: ±5% desplazamiento del centro
├── Rotation: ±15° rotación 2D de landmarks
├── Flip: Espejo horizontal (cambiar L↔R landmarks)
└── Noise: Gaussian σ=0.02 en coordenadas
```

#### 6.3.2 Transformaciones Temporales
```
Temporal Augmentations:
├── Speed: 0.8x - 1.2x velocidad original
├── Crop: Subsecuencias de 80%-100% duración
├── Interpolation: Upsampling a diferentes FPS
└── Jitter: Pequeños desplazamientos temporales
```

#### 6.3.3 Multiplicador de Datos
- **Factor esperado**: 3-4x datos originales
- **De 200 videos → 600-800 muestras** de entrenamiento
- **Validación**: Solo datos reales (sin augmentation)
- **Beneficio**: Mejor generalización y robustez

### 6.4 Control de Calidad en Escalamiento

#### Criterios de Aceptación
- ✅ **Detección MediaPipe**: >90% de frames con landmarks
- ✅ **Duración apropiada**: 8-45 segundos según actividad
- ✅ **Actividad pura**: Sin mezcla de múltiples actividades
- ✅ **Calidad de video**: Resolución mínima 720p, buena iluminación

#### Pipeline de Validación
1. **Captura** → 2. **Validación automática** → 3. **Revisión manual** → 4. **Incorporación**

#### Métricas de Progreso
- **Tasa de captura**: 15-20 videos/hora de sesión
- **Tasa de aceptación**: >85% videos capturados
- **Diversidad**: Máximo 40% videos de una sola persona
- **Balance**: Diferencia <20% entre clases más/menos representadas

---

## 7. Análisis de Aspectos Éticos

### 7.1 Privacidad y Consentimiento

#### 7.1.1 Consentimiento Informado Implementado
**✅ Protocolo Establecido**
- **Documento**: Consentimiento escrito firmado antes de grabación
- **Contenido**: Propósito académico, uso de datos, derechos del participante
- **Claridad**: Explicación en lenguaje simple, no técnico
- **Voluntariedad**: Énfasis en participación completamente voluntaria

**✅ Información Proporcionada**
- Objetivo del proyecto (clasificación de actividades)
- Uso de MediaPipe para extracción de coordenadas
- No almacenamiento de rostros identificables
- Duración del almacenamiento (hasta diciembre 2025)
- Derecho a retirarse en cualquier momento

#### 7.1.2 Anonimización de Datos
**✅ Medidas Implementadas**
- **Videos**: Nombres de archivo con códigos (P001_A1_T1.mp4)
- **Metadatos**: Solo edad, género, altura (sin nombres o IDs)
- **Almacenamiento**: Repositorio privado, acceso solo al equipo
- **Procesamiento**: Extracción de landmarks únicamente, no frames originales

**✅ Protección de Identidad**
- Opción de difuminar rostros (ofrecida a todos)
- No almacenamiento de información personal identificable
- Separación física: videos en carpeta diferente a metadatos
- Backup encriptado con contraseña del equipo

#### 7.1.3 Derecho al Olvido
**✅ Protocolo de Eliminación**
- **Proceso**: Email al equipo → eliminación en 48h → confirmación
- **Alcance**: Video original + landmarks extraídos + metadatos
- **Documentación**: Log de eliminaciones para transparencia
- **Timeline**: 2 participantes ya informados del proceso, 0 solicitudes hasta ahora

### 7.2 Equidad y Prevención de Sesgos

#### 7.2.1 Diversidad Demográfica Implementada
**✅ Balance de Género**
- **Actual**: 4 mujeres, 4 hombres (50/50 perfecto)
- **Objetivo Entrega 2**: 10 mujeres, 10 hombres
- **Consideración**: Inclusión de identidades no binarias si hay participantes

**✅ Variabilidad de Edad**
- **Actual**: 22-45 años (buena distribución)
- **Objetivo**: Incluir 18-65 años para mayor representatividad
- **Limitación conocida**: Sesgo hacia población universitaria joven

**✅ Diversidad Física**
- **Altura**: 1.58m-1.83m (excelente rango)
- **Complexión**: Delgada, media, robusta representadas
- **Capacidades**: 1 participante con limitación menor de movilidad
- **Objetivo**: 2-3 personas con diferentes capacidades motoras

#### 7.2.2 Prevención de Discriminación
**✅ Inclusión Activa**
- **Criterio**: Ningún participante excluido por capacidades físicas
- **Adaptación**: Actividades modificadas según capacidades individuales
- **Ejemplo**: "Girar" puede ser parcial si rotación completa es difícil
- **Documentación**: Variaciones registradas como válidas, no errores

**✅ Validación Anti-Sesgo**
- **Método**: Performance testing por subgrupos demográficos
- **Métricas**: Accuracy no debe variar >5% entre géneros/edades
- **Alerta**: Si accuracy <80% en cualquier subgrupo → investigar sesgo
- **Corrección**: Re-balanceado de datos o features adicionales

#### 7.2.3 Representatividad Cultural
**✅ Consideraciones Implementadas**
- **Estilos de movimiento**: Diferentes formas de caminar/sentarse
- **Vestimenta**: Ropa occidental y tradicional incluida
- **Contexto**: Grabaciones en espacios variados (formal/informal)

### 7.3 Transparencia y Explicabilidad

#### 7.3.1 Interpretabilidad del Modelo
**✅ Modelos Interpretables Seleccionados**
- **Random Forest**: Feature importance nativa
- **SVM**: Análisis de vectores de soporte
- **XGBoost**: SHAP values implementado
- **Evitar**: Redes neuronales profundas (menos interpretables)

**✅ Explicabilidad de Decisiones**
- **Para usuarios**: "Clasificado como 'caminar' basado en movimiento de piernas"
- **Para desarrolladores**: Feature importance ranking + SHAP plots
- **Para evaluadores**: Análisis detallado de errores y aciertos

#### 7.3.2 Limitaciones Documentadas
**✅ Transparencia sobre Restricciones**
- **Población**: Principalmente jóvenes universitarios (sesgo conocido)
- **Actividades**: Solo 5 básicas, no cubre actividades complejas
- **Ambiente**: Espacios interiores principalmente
- **Hardware**: Requiere cámara web de calidad mínima

**✅ Casos de Falla Conocidos**
- **Ropa muy holgada**: Puede afectar detección de landmarks
- **Iluminación extrema**: Muy oscuro o con sombras fuertes
- **Oclusiones**: Objetos que bloquean >50% del cuerpo
- **Velocidades extremas**: Muy lento (<0.5x) o muy rápido (>2x)

### 7.4 Uso Responsable y Aplicaciones

#### 7.4.1 Casos de Uso Apropiados ✅
- **Rehabilitación física**: Monitoreo de ejercicios terapéuticos
- **Investigación biomecánica**: Análisis académico de movimientos
- **Deporte**: Evaluación técnica de movimientos básicos
- **Asistencia gerontológica**: Detección de caídas en entorno controlado
- **Educación**: Herramienta de aprendizaje sobre análisis de movimiento

#### 7.4.2 Casos de Uso Problemáticos ❌
- **Vigilancia no consentida**: Monitoreo sin conocimiento de personas
- **Evaluación laboral discriminatoria**: Usar para decisiones de empleo
- **Diagnóstico médico**: Sistema no está validado clínicamente
- **Seguridad crítica**: No usar para decisiones de vida o muerte
- **Identificación de personas**: No diseñado para reconocimiento individual

#### 7.4.3 Recomendaciones de Implementación
**✅ Supervisión Humana Obligatoria**
- **Nunca**: Decisiones automáticas sin revisión humana
- **Siempre**: Human-in-the-loop para aplicaciones críticas
- **Logging**: Registrar todas las decisiones para auditoría

**✅ Comunicación Clara de Limitaciones**
- **A usuarios**: Explicar qué puede y no puede hacer el sistema
- **Documentación**: Manual con casos apropiados e inapropiados
- **Training**: Capacitación obligatoria para implementadores

### 7.5 Cumplimiento y Auditoría

#### 7.5.1 Checklist de Cumplimiento Ético
- ✅ **Consentimiento informado**: 100% participantes
- ✅ **Anonimización**: Implementada y verificada
- ✅ **Diversidad**: Balanceada según recursos disponibles
- ✅ **Transparencia**: Limitaciones documentadas
- ✅ **Uso responsable**: Casos apropiados/problemáticos identificados

#### 7.5.2 Plan de Auditoría Continua
- **Semanal**: Revisión de nuevos datos capturados
- **Por entrega**: Evaluación de sesgos en modelos
- **Final**: Auditoría completa antes de deployment
- **Post-deployment**: Monitoreo de uso y feedback de usuarios

---

## 8. Próximos Pasos y Cronograma

### 8.1 Plan Detallado por Entregas

#### Entrega 2 (Semana 14) - **Modelado Completo**
**Timeline: 14-27 octubre (2 semanas)**

**Semana 13 (14-20 octubre)**
- **Datos**: Ampliar a 150+ videos con crowdsourcing
- **Feature Engineering**: Velocidades, ángulos, características temporales
- **Baseline Models**: SVM y Random Forest implementados
- **Validación**: K-fold cross-validation setup

**Semana 14 (21-27 octubre)**
- **Modelos Avanzados**: XGBoost y ensemble methods
- **Hyperparameter Tuning**: Grid search optimización
- **Evaluación**: Métricas completas + leave-one-person-out
- **Selección**: Modelo final basado en métricas balanceadas

#### Entrega 3 (Semana 17) - **Sistema Completo**
**Timeline: 11-17 noviembre (1 semana intensiva)**

**11-13 noviembre**: Sistema en tiempo real
- **Backend**: API de clasificación optimizada
- **Frontend**: Interfaz gráfica con tkinter/streamlit
- **Integration**: MediaPipe + modelo en pipeline unificado

**14-16 noviembre**: Evaluación final
- **User Testing**: 5-8 usuarios finales probando sistema
- **Performance**: Validación métricas tiempo real
- **Documentation**: Reporte técnico completo (7 páginas)

**17 noviembre**: Entrega y presentación
- **Video Demo**: 10 minutos mostrando funcionalidades
- **Código Final**: Repositorio completamente documentado
- **Presentación**: Defensa oral del proyecto

### 8.2 Distribución de Responsabilidades

#### Por Miembro del Equipo
**[Juan Esteban Ruiz] - Líder del Proyecto**
- Coordinación general y timeline
- Feature engineering y selección de características
- Documentación técnica y reportes

**[Juan David Quintero] - Ingeniero de Datos**
- Recolección y limpieza de datos
- Pipeline MediaPipe y preprocessing
- Control de calidad de dataset

**[Tomas Quintero] - Especialista ML**
- Entrenamiento y optimización de modelos
- Implementación de métricas de evaluación
- Sistema de tiempo real y deployment

#### Tareas Compartidas
- **Captura de videos**: Todos los miembros
- **Testing del sistema**: Rotación por parejas
- **Revisión de documentación**: Peer review obligatorio
- **Presentación final**: Preparación conjunta
