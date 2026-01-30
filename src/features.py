# ============================================================================
# TRANSFORMACIÓN Y CODIFICACIÓN DE VARIABLES CATEGÓRICAS
# ============================================================================
import pandas as pd

def transformar_caracteristicas_categoricas(df_clean):
    print("\n" + "=" * 80)
    print("TRANSFORMACIÓN DE VARIABLES CATEGÓRICAS")
    print("=" * 80)
    
    categorical_cols = ['Brand', 'Fuel Type', 'Transmission', 'Condition', 'Model']
    
    # 6.1. Verificar valores únicos en variables categóricas
    print("\n1. VALORES ÚNICOS POR VARIABLE CATEGÓRICA:")
    print("-" * 40)
    for col in categorical_cols:
        if col in df_clean.columns:
            unique_count = df_clean[col].nunique()
            print(f"{col}: {unique_count} valores únicos")
            if unique_count < 10:
                print(f"  Valores: {df_clean[col].unique()}")
    
    # 6.2. Codificación de variables ordinales (si las hay)
    print("\n2. CODIFICACIÓN DE VARIABLES ORDINALES:")
    print("-" * 40)
    
    # Condition tiene un orden natural: New > Like New > Used
    condition_mapping = {'New': 2, 'Like New': 1, 'Used': 0}
    df_clean['Condition_encoded'] = df_clean['Condition'].map(condition_mapping)
    print("Variable 'Condition' codificada:")
    print(df_clean[['Condition', 'Condition_encoded']].head())
    
    # 6.3. One-Hot Encoding para variables nominales
    print("\n3. ONE-HOT ENCODING PARA VARIABLES NOMINALES:")
    print("-" * 40)
    
    # Variables para one-hot encoding
    nominal_vars = ['Fuel Type', 'Transmission']
    
    for var in nominal_vars:
        if var in df_clean.columns:
            # Usar pd.get_dummies()
            dummies = pd.get_dummies(df_clean[var], prefix=var.replace(' ', '_'))
            df_clean = pd.concat([df_clean, dummies], axis=1)
            print(f"{var}: {dummies.shape[1]} columnas creadas")
    
    # 6.4. Codificación de etiquetas para marcas (Label Encoding)
    print("\n4. LABEL ENCODING PARA MARCAS:")
    print("-" * 40)
    
    # Crear un mapeo único para cada marca
    brands = df_clean['Brand'].unique()
    brand_mapping = {brand: i for i, brand in enumerate(brands)}
    df_clean['Brand_encoded'] = df_clean['Brand'].map(brand_mapping)
    print(f"Marcas únicas codificadas: {len(brands)}")
    print("Muestra del mapeo:")
    for brand, code in list(brand_mapping.items())[:5]:
        print(f"  {brand}: {code}")
    
    return df_clean