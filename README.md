# Detección de DeepFakes usando Redes Neuronales Convolucionales en MATLAB
## Proyecto del Máster en Ingeniería de Telecomunicaciones - Asignatura de Reconocimiento Biométrico - EPS/UAM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![MATLAB Version](https://img.shields.io/badge/MATLAB-R2024b%2B-blue.svg)

Este proyecto implementa y evalúa sistemas de detección de DeepFakes utilizando técnicas de aprendizaje profundo (Deep Learning) con Redes Neuronales Convolucionales (CNNs) en el entorno MATLAB. Se exploran análisis intra-database e inter-database utilizando arquitecturas como AlexNet y una CNN.

**Desarrollado por:**
* [Miguel Carralero Lanchares](https://www.linkedin.com/in/miguel-carralero-lanchares/) <a href="https://www.linkedin.com/in/miguel-carralero-lanchares/" target="_blank"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" alt="LinkedIn" width="16" style="vertical-align:middle; margin-left:4px"/></a>
* [Íñigo Cameo de Dios](https://www.linkedin.com/in/%C3%AD%C3%B1igo-cameo-de-dios-23b388212/) <a href="https://www.linkedin.com/in/%C3%AD%C3%B1igo-cameo-de-dios-23b388212/" target="_blank"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" alt="LinkedIn" width="16" style="vertical-align:middle; margin-left:4px"/></a>
* [Francisco Orcha](https://www.linkedin.com/in/francisco-orcha-38a5831b3/) <a href="https://www.linkedin.com/in/francisco-orcha-38a5831b3/" target="_blank"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" alt="LinkedIn" width="16" style="vertical-align:middle; margin-left:4px"/></a>
* [Miguel Bueno](https://www.linkedin.com/in/miguel-bueno-mora-a63389263/) <a href="https://www.linkedin.com/in/miguel-bueno-mora-a63389263/" target="_blank"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" alt="LinkedIn" width="16" style="vertical-align:middle; margin-left:4px"/></a>

## Descripción General del Proyecto

El objetivo es desarrollar y evaluar sistemas de detección de DeepFakes. El proyecto se divide en tres tareas principales:

1.  **Task 1 - Análisis Intra-Database:** Desarrollo y evaluación de un sistema de detección de DeepFakes utilizando la base de datos UADFV. Se emplea transfer learning con una red AlexNet modificada.
2.  **Task 2 - Análisis Inter-Database:** Evaluación del modelo entrenado en la Tarea 1 sobre una base de datos diferente (Celeb-DF) para analizar su capacidad de generalización.
3.  **Task 3 - Propuesta de Mejora (Inter-Database):** Implementación de una nueva arquitectura CNN personalizada y entrenamiento con una base de datos ampliada (UADFV + imágenes espejo) para mejorar los resultados en el análisis inter-database sobre Celeb-DF.

El proyecto incluye un script de preprocesamiento (`encontrar_caras.m`) para detectar y recortar rostros de las imágenes, que luego son utilizados para entrenar y evaluar los modelos.

## Tecnologías Utilizadas

*   **Lenguaje y Entorno:** MATLAB (R2024b o similar)
*   **Toolboxes de MATLAB:**
    *   Deep Learning Toolbox (para AlexNet, capas de CNN, entrenamiento)
    *   Image Processing Toolbox (para `imcrop`, `imread`, etc.)
    *   Computer Vision Toolbox (o similar, para `vision.CascadeObjectDetector`)
*   **Arquitecturas CNN:** AlexNet (modificada), CNN personalizada.
*   **Técnicas:** Transfer Learning, Data Augmentation (espejo).

## Estructura del Repositorio

```
+-- .gitignore
+-- LICENSE
+-- README.md
+-- data/
| +-- Task1/
| | +-- development_cropped/    (Caras recortadas de UADFV para entrenamiento de Task 1)
| | | +-- fake/
| | | +-- real/
| | +-- evaluation_cropped/     (Caras recortadas de UADFV para evaluación de Task 1)
| | | +-- fake/
| | | +-- real/
| +-- Task2/
| | +-- cropped_faces/          (Caras recortadas de Celeb-DF para entrenamiento de Task 2)
| | | +-- fake/
| | | +-- real/
| | +-- evaluation/             (Dataset Celeb-DF original para evaluación de Task 2)
| | | +-- fake/
| | | +-- real/
| +-- Task3/
| | +-- development_cropped/    (Caras recortadas para entrenamiento de Task 3)
| | | +-- fake/
| | | +-- real/
| | +-- evaluation_cropped/     (Caras recortadas para evaluación de Task 3)
| | | +-- fake/
| | | +-- real/
+-- code/
| +-- encontrar_caras.m (Script de preprocesamiento para recortar caras)
| +-- Task_1.m (Script para la Tarea 1: AlexNet en UADFV)
| +-- Task_2.m (Script para la Tarea 2: Modelo de Tarea 1 en Celeb-DF)
| +-- Task_3.m (Script para la Tarea 3: CNN personalizada)
+-- models/
| +-- AlexNet_task2.mat (Modelo entrenado en Task 1 y usado en Task 2)
| +-- AlexNet_task3.mat (Modelo entrenado en la Task3)
+-- docs/
| +-- DeepFakes_Detection_Lab_Report.pdf (Informe detallado del proyecto)
+-- ...
```

## Preparación del Entorno y Datos

### 1. Obtener el Código del Repositorio
Para obtener una copia local de este proyecto, clone el repositorio usando Git:

```bash
git clone https://github.com/MiguelCarra/deepfake-detection-matlab.git
cd deepfake-detection-matlab
```

### 2. Software Requerido
*   MATLAB (versión R2024b o posterior recomendada).
*   Toolboxes de MATLAB:
    *   **Deep Learning Toolbox**
    *   **Deep Learning Toolbox Model for AlexNet Network Support Package** (Para Task 1 y Task 2. Se puede instalar desde el Add-Ons Explorer de MATLAB).
    *   Image Processing Toolbox
    *   Computer Vision Toolbox (o el toolbox que provea `vision.CascadeObjectDetector`)

### 3. Uso de los Datos y Modelos Incluidos (Recomendado para Ejecución Rápida)

Este repositorio **incluye las bases de datos con caras ya recortadas y los modelos pre-entrenados necesarios** para replicar los resultados del informe.

*   **Datos Preprocesados:**
    *   Las imágenes con rostros recortados para **Task 1** se encuentran en:
        *   `data/Task1/development_cropped/` (entrenamiento/desarrollo UADFV, sin data augmentation)
        *   `data/Task1/evaluation_cropped/` (evaluación UADFV)
    *   Las imágenes con rostros recortados para entrenamiento de **Task 3** (UADFV con data augmentation de espejo) se encuentran en:
        *   `data/Task3/development_cropped/`
    *   Las imágenes con rostros recortados para evaluación de **Task 2 y Task 3** (provenientes de Celeb-DF) se encuentran en:
        *   `data/Task2/cropped_faces/`

*   **Modelo Pre-entrenado para Task 2:**
    *   El modelo entrenado por `Task_1.m` y necesario para `Task_2.m` está disponible en `models/AlexNet_task2.mat`.
    *   Si ejecuta `Task_1.m`, este generará un nuevo modelo. Para usarlo en `Task_2.m`, deberá guardarlo/renombrarlo como `AlexNet_task2.mat` en la carpeta `models/`. Si no desea re-entrenar Task 1, puede usar directamente el modelo provisto.

Con estos datos y modelos, puede proceder directamente a la sección **"Ejecución de las Tareas"**.

### 4. Creación y Preprocesamiento de Datasets Propios (Opcional)

Si desea generar sus propias bases de datos a partir de los datasets originales UADFV y Celeb-DF, siga estos pasos:

**4.1. Descarga de Datasets Originales:**

*   **Este repositorio no incluye los datasets originales UADFV y Celeb-DF debido a su tamaño.**
*   **Dataset UADFV (para Task 1 y Task 3):**
    *   Descargue el dataset UADFV.
    *   Dentro de la carpeta `data/Task1/` (créela si no existe en la raíz del proyecto), cree las subcarpetas `development_original` y `evaluation_original`.
    *   Dentro de `data/Task1/development_original/`, cree subcarpetas `fake/` y `real/` y coloque las imágenes de entrenamiento/desarrollo UADFV correspondientes.
    *   Dentro de `data/Task1/evaluation_original/`, cree subcarpetas `fake/` y `real/` y coloque las imágenes de evaluación UADFV correspondientes.
*   **Dataset Celeb-DF (para Task 2 y Task 3):**
    *   Descargue el dataset Celeb-DF.
    *   Dentro de la carpeta `data/Task2/` (créela si no existe), cree una subcarpeta `celeb_df_original` (o el nombre que prefiera para los datos crudos).
    *   Dentro de `data/Task2/celeb_df_original/`, cree subcarpetas `fake/` y `real/` y coloque los vídeos o imágenes de Celeb-DF correspondientes. *Nota: `encontrar_caras.m` está configurado para procesar imágenes. Si descarga vídeos, deberá extraer los frames previamente.*

**4.2. Preprocesamiento de Caras Recortadas con `encontrar_caras.m`:**

El script `code/encontrar_caras.m` se encarga de detectar y recortar rostros. Deberá ejecutarlo configurándolo adecuadamente para cada conjunto de datos. Asegúrese de estar en la carpeta `code/` al ejecutarlo.

*   **Para Task 1 (Entrenamiento/Desarrollo UADFV - SIN Data Augmentation):**
    *   Modifique `encontrar_caras.m`:
        *   `evalFolder = fullfile('..', 'data', 'Task1', 'development_original');`
        *   `outputFolder = fullfile('..', 'data', 'Task1', 'development_cropped');`
    *   Asegúrese de que las líneas de data augmentation (espejo) estén comentadas.
    *   Ejecute el script. Las caras recortadas se guardarán en `data/Task1/development_cropped/`.

*   **Para Task 1 (Evaluación UADFV):**
    *   Modifique `encontrar_caras.m`:
        *   `evalFolder = fullfile('..', 'data', 'Task1', 'evaluation_original');`
        *   `outputFolder = fullfile('..', 'data', 'Task1', 'evaluation_cropped');`
    *   Asegúrese de que las líneas de data augmentation (espejo) estén comentadas.
    *   Ejecute el script. Las caras recortadas se guardarán en `data/Task1/evaluation_cropped/`.

*   **Para Task 3 (Entrenamiento/Desarrollo UADFV - CON Data Augmentation):**
    *   Modifique `encontrar_caras.m`:
        *   `evalFolder = fullfile('..', 'data', 'Task1', 'development_original');` (usa el mismo origen que Task 1 dev)
        *   `outputFolder = fullfile('..', 'data', 'Task3', 'development_cropped');` (guarda en la carpeta específica de Task 3)
    *   Asegúrese de descomentar las líneas relevantes para aplicar data augmentation (espejo).
    *   Ejecute el script. Las caras recortadas y aumentadas se guardarán en `data/Task3/development_cropped/`.

*   **Para Task 2 y Task 3 (Evaluación Celeb-DF):**
    *   Modifique `encontrar_caras.m`:
        *   `evalFolder = fullfile('..', 'data', 'Task2', 'celeb_df_original');` (o la ruta donde guardó Celeb-DF)
        *   `outputFolder = fullfile('..', 'data', 'Task2', 'cropped_faces');`
    *   Asegúrese de que las líneas de data augmentation (espejo) estén comentadas.
    *   Ejecute el script. Las caras recortadas se guardarán en `data/Task2/cropped_faces/`.

Tras estos pasos, tendrá las siguientes carpetas con sus datos procesados listos para ser usados por los scripts de las tareas:
*   `data/Task1/development_cropped/`   (UADFV desarrollo, sin augmentation)
*   `data/Task1/evaluation_cropped/`    (UADFV evaluación)
*   `data/Task3/development_cropped/`   (UADFV desarrollo, con augmentation)
*   `data/Task2/cropped_faces/`         (Celeb-DF evaluación)

## Ejecución de las Tareas

Todos los scripts deben ejecutarse desde MATLAB, teniendo la carpeta `code/` como directorio actual.

1.  **Ejecutar Tarea 1 (`Task_1.m`):**
    *   Utiliza los datos de `data/Task1/development_cropped/` para entrenar y `data/Task1/evaluation_cropped/` para evaluar.
    *   Entrenará la AlexNet modificada.
    *   Guardará el modelo entrenado (ej. `trainedNet`). **Si desea usar este nuevo modelo para la Tarea 2, recuerde mover/renombrar este modelo a `../models/AlexNet_task2.mat`.**
    *   Mostrará curvas ROC y métricas de rendimiento.

2.  **Ejecutar Tarea 2 (`Task_2.m`):**
    *   Carga el modelo desde `../models/AlexNet_task2.mat` (ya sea el provisto o el que usted generó y guardó desde Task 1).
    *   Utiliza los datos de `data/Task2/cropped_faces/` para la evaluación inter-database.
    *   Mostrará la curva ROC y métricas de rendimiento.

3.  **Ejecutar Tarea 3 (`Task_3.m`):**
    *   Utiliza los datos de `data/Task3/development_cropped/` (UADFV con data augmentation) para entrenar la CNN personalizada.
    *   Utiliza los datos de `data/Task2/cropped_faces/` para la evaluación.
    *   Mostrará curvas ROC y métricas de rendimiento. El modelo entrenado puede ser guardado si modifica el script.
    *   
## Resultados y Discusión

Los resultados detallados, incluyendo métricas de exactitud, AUC, y las curvas ROC para cada tarea, se encuentran en el **[Informe Completo del Laboratorio](docs/DeepFakes_Detection_Lab_Report.pdf)**.

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.
