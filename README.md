# Segmentación de Células con CellPose

Este proyecto implementa un sistema modular para la segmentación automática de células en imágenes microscópicas utilizando CellPose, un algoritmo de segmentación celular basado en aprendizaje profundo.

## Tabla de Contenidos

1. [Descripción General](#descripción-general)
2. [Requisitos](#requisitos)
3. [Estructura del Proyecto](#estructura-del-proyecto)
4. [Guía de Uso](#guía-de-uso)
5. [Aspectos Técnicos](#aspectos-técnicos)
6. [Consideraciones](#consideraciones)
7. [Procesamiento de las Imágenes](#procesamiento-de-las-imágenes)
8. [Análisis de Resultados](#análisis-de-resultados)

## Descripción General

El sistema procesa automáticamente imágenes microscópicas para identificar y segmentar células, generando máscaras de segmentación, visualizaciones y metadatos. Está diseñado para procesar grandes colecciones de imágenes (6336 en este caso) siguiendo un patrón de nomenclatura específico (`Maaxt0_i_xyz`).

CellPose es una herramienta de segmentación basada en redes neuronales que ha sido entrenada en grandes conjuntos de datos de imágenes celulares. A diferencia de otros métodos, CellPose no requiere un entrenamiento específico para cada tipo de imagen, funcionando como un modelo generalizado para diferentes tipos de células.

## Requisitos

```
numpy>=1.19.0
opencv-python>=4.5.0
cellpose==2.0.0
matplotlib>=3.3.0
pyyaml>=5.4.0
tqdm>=4.50.0
scikit-learn==1.6.1

```

Para instalar las dependencias:

```bash
pip install -r requirements.txt
```

## Estructura del Proyecto

```
cellpose_segmentation/
│
├── config/
│   └── config.yaml         # Configuraciones del proyecto
│
├── data/
│   ├── __init__.py
│   ├── data_loader.py      # Para cargar y organizar los datos
│   └── data_preprocessor.py # Procesamiento previo de imágenes
│
├── models/
│   ├── __init__.py
│   └── cellpose_model.py   # Configuración y uso del modelo CellPose
│
├── utils/
│   ├── __init__.py
│   ├── visualization.py    # Funciones para visualizar resultados
│   └── file_utils.py       # Utilidades para manejo de archivos
│
├── segmentation/
│   ├── __init__.py
│   └── segmentation.py     # Funciones para segmentación
│
├── main.py                 # Punto de entrada principal
├── run_segmentation.py     # Script para ejecutar el proceso de segmentación
└── requirements.txt        # Dependencias del proyecto
```

## Guía de Uso

### Configuración

Antes de usar el sistema, edita el archivo `config/config.yaml` para establecer:

1. Directorio de entrada (`input_dir`): donde se encuentran las imágenes
2. Directorio de salida (`output_dir`): donde se guardarán los resultados
3. Parámetros de CellPose como:
   - `model_type`: tipo de modelo (cyto, nuclei, etc.)
   - `diameter`: diámetro aproximado de células en píxeles
   - `channels`: canales a usar para la segmentación

### Ejecución Básica

Para procesar todas las imágenes:

```bash
python main.py --config config/config.yaml
```

### Opciones Avanzadas

Procesar por escena:

```bash
python main.py --config config/config.yaml --mode scene
```

Guardar visualizaciones detalladas:

```bash
python main.py --config config/config.yaml --save-viz
```

Procesar una escena específica:

```bash
python run_segmentation.py --config config/config.yaml --scene 55
```

## Aspectos Técnicos

### Flujo de Procesamiento

1. **Carga de Datos**: 
   - Las imágenes se cargan utilizando OpenCV
   - Se organizan según el patrón de nombre (escenas)

2. **Preprocesamiento**:
   - Conversión a RGB si es necesario
   - Normalización de valores de píxeles

3. **Segmentación con CellPose**:
   - Aplicación del modelo preentrenado
   - Ajuste mediante parámetros de umbral y diámetro

4. **Postprocesamiento**:
   - Generación de máscaras
   - Cálculo de estadísticas (número de células, áreas)
   - Creación de visualizaciones

5. **Almacenamiento de Resultados**:
   - Máscaras de segmentación
   - Imágenes compuestas (original + contornos)
   - Metadatos en formato JSON
   - Figuras de visualización (opcional)

### Módulos Clave

#### DataLoader (data_loader.py)

Responsable de:
- Encontrar imágenes que coinciden con el patrón especificado
- Organizar imágenes por escena basándose en su nomenclatura
- Generar rutas de salida para los diferentes archivos resultantes

El patrón de nombramiento `Maxt0_i_xyz` se interpreta como:
- `i`: representa el número de escena (para i=0, se representa como '00')
- `xyz`: representa el número dentro de la escena (comienza con '000', '001', etc.)

#### DataPreprocessor (data_preprocessor.py)

Prepara las imágenes para ser procesadas por CellPose:
- Carga imágenes utilizando OpenCV
- Convierte imágenes a RGB (CellPose espera imágenes en RGB)
- Normaliza las imágenes si es necesario

#### CellPoseModel (cellpose_model.py)

Encapsula la funcionalidad del modelo CellPose:
- Inicializa el modelo con parámetros apropiados
- Implementa métodos para segmentar imágenes individuales o lotes
- Maneja la interfaz con la biblioteca CellPose

Parámetros clave:
- `model_type`: Determina el modelo preentrenado ('cyto' para citoplasma, 'nuclei' para núcleos)
- `diameter`: Tamaño aproximado de células en píxeles (crucial para buen rendimiento)
- `flow_threshold`: Umbral para la detección del flujo de células
- `cellprob_threshold`: Umbral de probabilidad para identificar una región como célula

#### Visualizer (visualization.py)

Genera visualizaciones de los resultados:
- Superpone máscaras de segmentación sobre imágenes originales
- Crea figuras que muestran imagen original, máscara y superposición
- Utiliza esquemas de color apropiados para la visualización

#### FileUtils (file_utils.py)

Maneja operaciones de archivo:
- Guarda máscaras de segmentación como imágenes
- Guarda imágenes compuestas con contornos
- Genera y guarda metadatos (JSON) con estadísticas

#### SegmentationPipeline (segmentation.py)

Orquesta todo el proceso de segmentación:
- Coordina los diferentes componentes
- Implementa el flujo de procesamiento completo
- Proporciona métodos para procesar imágenes individuales, lotes o escenas completas
- Genera estadísticas sobre los resultados

## Consideraciones

### Ajuste de Parámetros

El rendimiento de CellPose depende principalmente de estos parámetros:

1. **Diámetro celular** (`diameter`):
   - Debe ser ajustado según el tamaño típico de células en las imágenes
   - Valores incorrectos pueden llevar a sub o sobre-segmentación
   - Recomendación: prueba con diferentes valores (15-50 píxeles) para encontrar el óptimo

2. **Umbrales** (`flow_threshold` y `cellprob_threshold`):
   - Controlan la sensibilidad de la detección
   - Valores bajos: más células detectadas pero posibles falsos positivos
   - Valores altos: menos células pero más confiables

3. **Tipo de modelo** (`model_type`):
   - 'cyto': optimizado para células completas (citoplasma)
   - 'nuclei': optimizado para núcleos celulares
   - Elige según el aspecto de las células en tus imágenes

### Rendimiento y Optimización

Para procesar conjuntos grandes de imágenes eficientemente:

1. **Procesamiento por lotes**: 
   - Implementado mediante `process_batch` para reducir sobrecarga
   - Tamaño de lote configurable (por defecto: 10 imágenes)

2. **GPU vs CPU**:
   - CellPose puede usar GPU mediante PyTorch
   - Recomendado para conjuntos grandes (6336 imágenes)
   - Configurable mediante el parámetro `gpu` en `config.yaml`

3. **Memoria**:
   - Monitoriza el uso de memoria cuando procesas lotes grandes
   - Reduce el tamaño de lote si hay problemas de memoria

### Limitaciones

1. **Tipos de células**:
   - CellPose funciona mejor con células relativamente redondeadas
   - Células muy irregulares o alargadas pueden ser difíciles de segmentar

2. **Calidad de imagen**:
   - Bajo contraste o ruido excesivo pueden afectar la calidad de segmentación
   - Considera preprocesamiento adicional para imágenes problemáticas

3. **Densidad celular**:
   - Áreas con células muy densas o superpuestas pueden ser difíciles de segmentar correctamente

## Procesamiento de las Imágenes

### Nomenclatura de Imágenes

El sistema está diseñado para procesar imágenes con el patrón `Maaxt0_i_xyz.jpg`:

- `i`: Número de escena (se utilizan valores como "00", "01", "55", etc.)
- `xyz`: Número dentro de la escena (secuencia como "000", "001", "002", etc.)

Ejemplo: `Maxt0_55_008.jpg` pertenece a la escena 55, imagen 8 dentro de esa escena.

### Organización por Escenas

Las imágenes se organizan automáticamente por escenas, permitiendo:
- Procesamiento independiente de cada escena
- Estadísticas específicas por escena
- Comparación entre diferentes escenas

### Resultados Generados

Para cada imagen procesada, se generan:

1. **Máscara de segmentación** (`*_mask.jpg`):
   - Imagen donde cada célula tiene un ID único (valor de píxel)
   - Fondo representado con valor 0

2. **Imagen compuesta** (`*_composite.jpg`):
   - Imagen original con contornos celulares superpuestos
   - Facilita la verificación visual de resultados

3. **Metadatos** (`*_metadata.json`):
   - Número de células detectadas
   - Áreas celulares (total, media, mediana)
   - Información sobre el procesamiento

4. **Visualización** (opcional, `*_visualization.png`):
   - Figura con imagen original, máscara y composición
   - Útil para evaluación detallada y presentaciones

## Análisis de Resultados

### Estadísticas Generadas

El sistema calcula automáticamente:

1. **Por imagen**:
   - Número de células
   - Área de cada célula
   - Área media y mediana

2. **Por escena**:
   - Total de células
   - Promedio de células por imagen
   - Variación entre imágenes

3. **Global**:
   - Total de imágenes procesadas
   - Total de células detectadas
   - Tiempo de procesamiento

### Interpretación

Para analizar los resultados:

1. **Evaluación visual**:
   - Revisa las imágenes compuestas para verificar segmentación
   - Busca patrones de sub o sobre-segmentación

2. **Análisis cuantitativo**:
   - Compara densidad celular entre escenas
   - Analiza distribución de tamaños celulares
   - Identifica tendencias o anomalías

3. **Validación**:
   - Si es posible, compara con segmentación manual en muestra representativa
   - Calcula métricas como IoU (Intersection over Union) o F1-score

### Posibles Extensiones

El sistema es modular y puede extenderse para:

1. **Clasificación celular**:
   - Agregar módulos para clasificar células por tipo o estado
   - Implementar extracción de características morfológicas

2. **Análisis temporal**:
   - Seguimiento de células entre imágenes secuenciales
   - Estudio de cambios morfológicos en el tiempo

3. **Integración con análisis estadísticos**:
   - Exportación a formatos compatibles con R o Python para análisis avanzados
   - Generación de gráficos y reportes automáticos
