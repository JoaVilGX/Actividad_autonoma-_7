# ============================================================================
# DETECCIÓN DE PROBLEMAS Y LIMPIEZA DE DATOS
# ============================================================================
import pandas as pd

def detectar_problemas(df):
    print("\n" + "=" * 80)
    print("PROBLEMAS IDENTIFICADOS EN EL DATASET")
    print("=" * 80)
    
    # 3.1. Valores faltantes
    print("\n1. VALORES FALTANTES POR COLUMNA:")
    print("-" * 40)
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Valores_Faltantes': missing_values,
        'Porcentaje': missing_percentage
    })
    print(missing_df[missing_df['Valores_Faltantes'] > 0])
    
    # 3.2. Filas completamente vacías
    print("\n2. FILAS COMPLETAMENTE VACÍAS:")
    print("-" * 40)
    completely_empty = df.isnull().all(axis=1).sum()
    print(f"Filas completamente vacías: {completely_empty}")
    
    # 3.3. Valores atípicos potenciales
    print("\n3. VALORES ATÍPICOS POTENCIALES:")
    print("-" * 40)
    numeric_cols = ['Year', 'Engine Size', 'Mileage', 'Price']
    for col in numeric_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            print(f"{col}: {len(outliers)} outliers potenciales")

def limpiar_datos(df):
    print("\n" + "=" * 80)
    print("PROCESO DE LIMPIEZA DE DATOS")
    print("=" * 80)
    
    # Crear una copia para la limpieza
    df_clean = df.copy()
    
    # 4.1. Eliminar filas completamente vacías
    print("\n1. ELIMINANDO FILAS COMPLETAMENTE VACÍAS:")
    print("-" * 40)
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna(how='all')
    print(f"Filas eliminadas: {initial_rows - len(df_clean)}")
    print(f"Filas restantes: {len(df_clean)}")
    
    # 4.2. Eliminar filas donde Car ID es NaN (identificador único)
    print("\n2. ELIMINANDO FILAS SIN IDENTIFICADOR:")
    print("-" * 40)
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna(subset=['Car ID'])
    print(f"Filas eliminadas: {initial_rows - len(df_clean)}")
    print(f"Filas restantes: {len(df_clean)}")
    
    # 4.3. Convertir Car ID a entero
    df_clean['Car ID'] = df_clean['Car ID'].astype(int)
    
    # 4.4. Imputación de valores faltantes
    print("\n3. IMPUTACIÓN DE VALORES FALTANTES:")
    print("-" * 40)
    
    # Variables numéricas: imputar con mediana
    numeric_cols = ['Year', 'Engine Size', 'Mileage', 'Price']
    for col in numeric_cols:
        if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            print(f"{col}: {df_clean[col].isnull().sum()} nulos restantes (imputados con mediana {median_val:.2f})")
    
    # Variables categóricas: imputar con moda
    categorical_cols = ['Brand', 'Fuel Type', 'Transmission', 'Condition', 'Model']
    for col in categorical_cols:
        if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0]
            df_clean[col].fillna(mode_val, inplace=True)
            print(f"{col}: {df_clean[col].isnull().sum()} nulos restantes (imputados con moda '{mode_val}')")
    
    # 4.5. Verificación final de valores faltantes
    print("\n4. VERIFICACIÓN FINAL DE VALORES FALTANTES:")
    print("-" * 40)
    print(f"Total de valores faltantes restantes: {df_clean.isnull().sum().sum()}")
    
    return df_clean