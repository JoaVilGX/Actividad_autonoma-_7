# ============================================================================
# CARGA Y EXPLORACIÓN INICIAL DEL DATASET
# ============================================================================
import os
import pandas as pd

def cargar_y_explorar_datos():
    print("=" * 80)
    print("ANÁLISIS EXPLORATORIO Y PREPROCESAMIENTO")
    print("=" * 80)
    
    # Obtener el directorio actual del archivo carga_datos.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Subir un nivel (a la raíz del proyecto) y luego entrar a data
    data_path = os.path.join(current_dir, '..', 'data', 'car_price_prediction_with_missing.csv')
    # Normalizar la ruta (eliminar '..' y hacerla absoluta)
    data_path = os.path.normpath(data_path)
    
    # Cargar el dataset
    df = pd.read_csv(data_path)
    
    print("\n1. DIMENSIONES Y ESTRUCTURA INICIAL:")
    print("-" * 40)
    print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
    print(f"\nColumnas disponibles: {list(df.columns)}")
    
    # Mostrar información básica
    print("\n2. TIPOS DE DATOS Y VALORES FALTANTES INICIALES:")
    print("-" * 40)
    print(df.info())
    
    # Estadísticas descriptivas
    print("\n3. ESTADÍSTICAS DESCRIPTIVAS (NUMÉRICAS):")
    print("-" * 40)
    print(df.describe())
    
    print("\n4. ESTADÍSTICAS DESCRIPTIVAS (CATEGÓRICAS):")
    print("-" * 40)
    categorical_cols = ['Brand', 'Fuel Type', 'Transmission', 'Condition', 'Model']
    for col in categorical_cols:
        if col in df.columns:
            print(f"\n{col}:")
            print(df[col].value_counts().head())
    
    return df