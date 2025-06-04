# Detección de DeepFakes usando Redes Neuronales Convolucionales en MATLAB
## Proyecto del Máster en Ingeniería de Telecomunicaciones - Asignatura de Reconocimiento Biométrico - EPS/UAM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![MATLAB Version](https://img.shields.io/badge/MATLAB-R2024b%2B-blue.svg)

Este proyecto implementa y evalúa sistemas de detección de DeepFakes utilizando técnicas de aprendizaje profundo (Deep Learning) con Redes Neuronales Convolucionales (CNNs) en el entorno MATLAB. Se exploran análisis intra-database e inter-database utilizando arquitecturas como AlexNet y una CNN personalizada.


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

*   **Lenguaje y Entorno:** MATLAB (R2022a o similar)
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
| | +-- fake/
| | +-- real/
| +-- Task2/
| | +-- cropped_faces/          (Caras recortadas de Celeb-DF para entrenamiento de Task 2)
| | +-- fake/
| | +-- real/
| | +-- evaluation/             (Dataset Celeb-DF original para evaluación de Task 2)
| | | +-- fake/
| | | +-- real/
| +-- Task3/
| | +-- development_cropped/    (Caras recortadas para entrenamiento de Task 3)
| | | +-- fake/
| | | +-- real/
| | +-- evaluation_cropped/     (Caras recortadas para evaluación de Task 3)
| | +-- fake/
| | +-- real/
+-- code/
| +-- encontrar_caras.m (Script de preprocesamiento para recortar caras)
| +-- Task_1.m (Script para la Tarea 1: AlexNet en UADFV)
| +-- Task_2.m (Script para la Tarea 2: Modelo de Tarea 1 en Celeb-DF)
| +-- Task_3.m (Script para la Tarea 3: CNN personalizada)
+-- models/
| +-- AlexNet_task2.mat (Modelo entrenado en Task 1 y usado en Task 2. Renombrado desde AlexNetTrained5.mat)
| +-- (Posiblemente otros modelos como el de Task 3 si se guarda)
+-- docs/
| +-- DeepFakes_Detection_Lab_Report.pdf (Informe detallado del proyecto)
+-- ...
```

## Preparación del Entorno y Datos

### 1. Software Requerido
*   MATLAB (versión R2020a o posterior recomendada).
*   Toolboxes de MATLAB:
    *   **Deep Learning Toolbox**
    *   **Deep Learning Toolbox Model for AlexNet Network Support Package** (Para Task 1 y Task 2. Se puede instalar desde el Add-On Explorer de MATLAB).
    *   Image Processing Toolbox
    *   Computer Vision Toolbox (o el toolbox que provea `vision.CascadeObjectDetector`)

### 2. Configuración de los Datasets

**Este repositorio no incluye los datasets UADFV y Celeb-DF debido a su tamaño.** Siga estos pasos para configurarlos:

1.  **Dataset UADFV (para Task 1):**
    *   Descargue el dataset UADFV.
    *   Dentro de la carpeta `data/Task1/` (créela si no existe), cree las subcarpetas `development` y `evaluation`.
    *   Dentro de `data/Task1/development/`, cree subcarpetas `fake/` y `real/` y coloque las imágenes de entrenamiento/desarrollo UADFV correspondientes.
    *   Dentro de `data/Task1/evaluation/`, cree subcarpetas `fake/` y `real/` y coloque las imágenes de evaluación UADFV correspondientes.

2.  **Dataset Celeb-DF (para Task 2 y Task 3):**
    *   Descargue el dataset Celeb-DF.
    *   Dentro de la carpeta `data/Task2/` (créela si no existe), cree una subcarpeta `celeb_df_original` (o el nombre que prefiera para los datos crudos).
    *   Dentro de `data/Task2/celeb_df_original/`, cree subcarpetas `fake/` y `real/` y coloque las imágenes de evaluación Celeb-DF correspondientes.

### 3. Preprocesamiento de Caras Recortadas

Los scripts de las tareas utilizan imágenes con rostros recortados. El script `code/encontrar_caras.m` se encarga de este preprocesamiento.

**Ejecute `encontrar_caras.m` tres veces, configurándolo adecuadamente para cada conjunto de datos:**

*   **Para Task 1 (Entrenamiento/Desarrollo UADFV):**
    *   Modifique `encontrar_caras.m`:
        *   `evalFolder = fullfile('..', 'data', 'Task1', 'development');`
        *   `outputFolder = fullfile('..', 'data', 'Task1', 'development_cropped');`
    *   Si desea aplicar data augmentation (espejo) para Task 3, descomente las líneas relevantes en `encontrar_caras.m` ANTES de ejecutar para este conjunto.
    *   Ejecute el script desde la carpeta `code/`.

*   **Para Task 1 (Evaluación UADFV):**
    *   Modifique `encontrar_caras.m`:
        *   `evalFolder = fullfile('..', 'data', 'Task1', 'evaluation');`
        *   `outputFolder = fullfile('..', 'data', 'Task1', 'evaluation_cropped');`
    *   Asegúrese de que las líneas de data augmentation (espejo) estén comentadas.
    *   Ejecute el script desde la carpeta `code/`.

*   **Para Task 2 y Task 3 (Evaluación Celeb-DF):**
    *   Modifique `encontrar_caras.m`:
        *   `evalFolder = fullfile('..', 'data', 'Task2', 'celeb_df_original');` (o la ruta donde guardó Celeb-DF)
        *   `outputFolder = fullfile('..', 'data', 'Task2', 'cropped_faces');`
    *   Asegúrese de que las líneas de data augmentation (espejo) estén comentadas.
    *   Ejecute el script desde la carpeta `code/`.

Tras estos pasos, tendrá las siguientes carpetas con datos procesados:
*   `data/Task1/development_cropped/`
*   `data/Task1/evaluation_cropped/`
*   `data/Task2/cropped_faces/`

*Nota: `Task_3.m` utiliza `../data/Task1/development_cropped` para entrenamiento y `../data/Task2/cropped_faces` para evaluación, según las rutas en su código.*

### 4. Modelo Pre-entrenado

*   Para `Task_2.m`, se necesita el modelo entrenado en `Task_1.m`. El script `Task_2.m` espera encontrarlo en `../models/AlexNet_task2.mat`.
*   Asegúrese de que después de ejecutar `Task_1.m`, el modelo guardado (`trainedNet`) se copie o se guarde como `AlexNet_task2.mat` dentro de la carpeta `models/` (créela en la raíz del repositorio si no existe).

## Ejecución de las Tareas

Todos los scripts deben ejecutarse desde MATLAB, teniendo la carpeta `code/` como directorio actual.

1.  **Ejecutar Tarea 1 (`Task_1.m`):**
    *   Este script utiliza los datos de `data/Task1/development_cropped/` para entrenar y `data/Task1/evaluation_cropped/` para evaluar.
    *   Entrenará la AlexNet modificada.
    *   Guardará el modelo entrenado (ej. `trainedNet`). **Recuerde mover/renombrar este modelo a `models/AlexNet_task2.mat` para la Tarea 2.**
    *   Mostrará curvas ROC y métricas de rendimiento.

2.  **Ejecutar Tarea 2 (`Task_2.m`):**
    *   Este script carga el modelo `models/AlexNet_task2.mat`.
    *   Utiliza los datos de `data/Task2/cropped_faces/` para la evaluación inter-database.
    *   Mostrará la curva ROC y métricas de rendimiento.

3.  **Ejecutar Tarea 3 (`Task_3.m`):**
    *   Este script utiliza los datos de `data/Task1/development_cropped/` (posiblemente con data augmentation si se activó en `encontrar_caras.m`) para entrenar la CNN personalizada.
    *   Utiliza los datos de `data/Task2/cropped_faces/` para la evaluación.
    *   Mostrará curvas ROC y métricas de rendimiento.

## Resultados y Discusión

Los resultados detallados, incluyendo métricas de exactitud, AUC, y las curvas ROC para cada tarea, se encuentran en el **[Informe Completo del Laboratorio](docs/DeepFakes_Detection_Lab_Report.pdf)**.

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.
