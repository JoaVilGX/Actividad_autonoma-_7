#!/usr/bin/env python3
# ============================================================================
# MAIN - PUNTO DE ENTRADA PRINCIPAL DEL PROYECTO
# ============================================================================

# Añadir src al path para que Python encuentre los módulos
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Importar configuraciones
from config import librerias_config

# Importar módulos
from carga_datos import cargar_y_explorar_datos
from limpieza_datos import detectar_problemas, limpiar_datos
from outliers import manejar_outliers
from features import transformar_caracteristicas_categoricas
from scale import escalar_variables_numericas
from visualizaciones import crear_visualizaciones_exploratorias
from guardado_datos import guardar_dataset_limpio
from models import (inspeccionar_dataset_procesado, graficar_distribucion_objetivo, 
                     preparar_caracteristicas, dividir_datos, escalar_caracteristicas,
                     entrenar_regresion_logistica, entrenar_random_forest,
                     evaluar_modelos, crear_visualizaciones_comparativas)

def main():
    """Función principal que ejecuta todo el flujo del proyecto"""
    
    print("=" * 80)
    print("INICIO DEL PROYECTO: ANÁLISIS DE PRECIOS DE VEHÍCULOS")
    print("=" * 80)
    
    # ============================================================================
    # FASE 1: PREPROCESAMIENTO DE DATOS
    # ============================================================================
    print("\n\n" + "=" * 80)
    print("FASE 1: PREPROCESAMIENTO DE DATOS")
    print("=" * 80)
    
    # 1. Carga y exploración inicial
    df = cargar_y_explorar_datos()
    
    # 2. Detección de problemas
    detectar_problemas(df)
    
    # 3. Limpieza de datos
    df_clean = limpiar_datos(df)
    
    # 4. Manejo de outliers
    df_clean = manejar_outliers(df_clean)
    
    # 5. Transformación de variables categóricas
    df_clean = transformar_caracteristicas_categoricas(df_clean)
    
    # 6. Estandarización de variables numéricas
    df_clean = escalar_variables_numericas(df_clean)
    
    # 7. Visualizaciones exploratorias
    crear_visualizaciones_exploratorias(df_clean)
    
    # 8. Guardado del dataset limpio
    guardar_dataset_limpio(df_clean)
    
    # ============================================================================
    # FASE 2: MODELADO Y EVALUACIÓN
    # ============================================================================
    print("\n\n" + "=" * 80)
    print("FASE 2: MODELADO Y EVALUACIÓN")
    print("=" * 80)
    
    # 1. Inspección del dataset procesado
    df_clean = inspeccionar_dataset_procesado()
    
    # 2. Visualización de distribución de la variable objetivo
    graficar_distribucion_objetivo(df_clean)
    
    # 3. Preparación de features
    X, y, class_names = preparar_caracteristicas(df_clean)
    
    # 4. División de datos
    X_train, X_test, y_train, y_test = dividir_datos(X, y)
    
    # 5. Escalado de características
    X_train_scaled, X_test_scaled = escalar_caracteristicas(X_train, X_test)
    
    # 6. Entrenamiento de modelos
    logreg_model = entrenar_regresion_logistica(X_train_scaled, y_train)
    rf_model = entrenar_random_forest(X_train_scaled, y_train)
    
    # 7. Evaluación de modelos
    y_pred_logreg, y_pred_proba_logreg, y_pred_rf, y_pred_proba_rf, metrics_logreg, metrics_rf = evaluar_modelos(
        logreg_model, rf_model, X_test_scaled, y_test
    )
    
    # 8. Visualización comparativa de resultados
    crear_visualizaciones_comparativas(
        y_test, y_pred_logreg, y_pred_proba_logreg,
        y_pred_rf, y_pred_proba_rf, metrics_logreg,
        metrics_rf, X, X_test_scaled, rf_model
    )
    
    print("\n" + "=" * 80)
    print("PROYECTO COMPLETADO EXITOSAMENTE")
    print("=" * 80)

if __name__ == "__main__":
    main()