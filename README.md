# Análisis Predictivo de Condición de Vehículos

## 📋 Información del Proyecto
**Actividad:** Autónoma 7 - Programación 2  
**Nombre:** Joaquin Villacreses Moreno
**Semestre:** Segundo "C"
**Periodo Académico:** 2S-2025  
**Universidad:** Universidad Nacional de Chimborazo (UNACH)  
**Fecha:** 30/01/2026

## 🎯 Objetivo
Implementar un sistema de clasificación para predecir la condición de vehículos (New, Like New, Used) utilizando técnicas de machine learning con Python.

## 🚀 Instalación y Ejecución

### Requisitos Previos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Instalación
```bash
# 1. Clonar o descomprimir el proyecto
# 2. Instalar dependencias
pip install -r requirements.txt
```

### Ejecución del Proyecto
```bash
# Ejecutar desde la raíz del proyecto
python main.py
```

## 📁 Estructura del Proyecto
```
proyecto/
├── data/
│   └── car_price_prediction_with_missing.csv  # Dataset original
├── notebooks/
│   └── Joaquin_Villacreses_Notebook_U3T2.ipynb  # Análisis exploratorio
├── src/                                       # Módulos Python
│   ├── __init__.py
│   ├── carga_datos.py
│   ├── config.py
│   ├── features.py
│   ├── guardado_datos.py
│   ├── limpieza_datos.py
│   ├── models.py
│   ├── outliers.py
│   ├── scale.py
│   └── visualizaciones.py
├── main.py                                    # Punto de entrada principal
├── README.md
└── requirements.txt                           # Dependencias del proyecto
```

## ⚙️ Funcionalidades Implementadas

### 1. Preprocesamiento de Datos
- Carga y exploración del dataset
- Detección y tratamiento de valores faltantes
- Manejo de outliers
- Codificación de variables categóricas
- Escalado y normalización de variables numéricas

### 2. Modelado Predictivo
- Regresión Logística
- Random Forest Classifier
- Evaluación comparativa de modelos
- Métricas de rendimiento (Accuracy, Precision, Recall, F1-Score)

### 3. Visualizaciones
- Análisis exploratorio de datos
- Distribución de variables
- Matrices de correlación
- Comparación de resultados de modelos

## 📊 Resultados Esperados
Al ejecutar el proyecto se generarán:
1. **Dataset procesado** (`car_price_cleaned.csv`)
2. **Visualizaciones exploratorias** (`visualizaciones_exploratorias.png`)
3. **Distribución de la variable objetivo** (`distribucion_condicion.png`)
4. **Comparación de modelos** (`comparacion_modelos_clasificacion.png`)

## 🔧 Dependencias Técnicas
El proyecto utiliza las siguientes librerías principales:
- **pandas** y **numpy** para manipulación de datos
- **scikit-learn** para modelos de machine learning
- **matplotlib** y **seaborn** para visualizaciones
- **jupyter** para el notebook de análisis

## 📝 Notas
Este proyecto fue desarrollado como parte de la Actividad Autónoma 7 de la asignatura Programación 2, demostrando habilidades en:
- Modularización de código Python
- Procesamiento de datos reales
- Implementación de algoritmos de clasificación
- Evaluación de modelos de machine learning

---