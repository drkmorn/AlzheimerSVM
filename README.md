# Clasificación de Etapas del Alzheimer mediante Imágenes de Radiografías Cerebrales

Este proyecto tiene como objetivo desarrollar un modelo de aprendizaje supervisado para clasificar imágenes de radiografías cerebrales y detectar distintas etapas de la enfermedad de Alzheimer. Utiliza técnicas de procesamiento de imágenes, reducción de dimensionalidad con PCA y modelos de clasificación basados en SVM.

## Motivación

La enfermedad de Alzheimer es una de las principales causas de muerte en personas mayores. Una detección temprana puede ayudar a ralentizar su avance. Este proyecto busca apoyar, mas no reemplazar, el diagnóstico médico mediante un modelo de clasificación automatizado.

## Objetivo

Clasificar imágenes cerebrales en una de cuatro categorías:
- Non Demented (sin demencia)
- Very Mild Demented (demencia muy leve)
- Mild Demented (demencia leve)
- Moderate Demented (demencia moderada)

El foco principal está en detectar correctamente la etapa **Very Mild Demented**, donde es más efectiva la intervención temprana.

## 🗂Dataset

El conjunto de datos fue obtenido de Kaggle:
[Alzheimer’s Dataset - 4 Class of Images](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images)
Contiene más de 5,000 imágenes categorizadas.

## Preprocesamiento

- **Redimensión**: recorte para obtener imágenes cuadradas (176x176 → 176x176).
- **Escala de grises**: se usó una sola banda de color.
- **Normalización**: estandarización de los valores de píxeles.
- **Aplanamiento**: cada imagen se transformó en un vector unidimensional.

## Modelos y Técnicas

Se probaron múltiples variantes de Support Vector Machines (SVM):

### SVM sin PCA
- **Kernel Lineal**: mejor rendimiento general sin sobreajuste grave (accuracy: ~97.8%).
- **Kernel Polinomial** y **Gaussiano**: resultados menos precisos.

### SVM con PCA
- **PCA** aplicado para reducir la dimensionalidad de 30,976 a componentes principales.
- **Kernel Polinomial (grado 2, C=10)** con PCA: mejor modelo (accuracy: **99.41%**, F1: **98.9%**), excelente rendimiento en la categoría Very Mild Demented.

## Métricas de Evaluación

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Matriz de Confusión**

El modelo final tuvo únicamente un falso negativo en la categoría más crítica.

## Consideraciones

- El modelo **no reemplaza** una evaluación médica profesional.
- Está pensado como **herramienta de apoyo**.
- Se recomienda evaluar su rendimiento con nuevas imágenes, incluso fuera del set original.

## Referencias

- Alzheimer’s Association (2024). [Datos y cifras](https://www.alz.org/alzheimer-demencia/datos-y-f cifras)
- CDC (2024). [Aging and Dementia](https://www.cdc.gov/aging/publications/features/Alz-Greater-Risk.html)
- Kaggle Dataset: https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images

---


### Resultado Final

> Se superó la meta del 90% de efectividad  
> Se alcanzó un rendimiento del **99.41%**, clasificando correctamente la mayoría de los casos críticos.


