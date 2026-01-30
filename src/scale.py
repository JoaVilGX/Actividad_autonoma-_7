# ============================================================================
# ESTANDARIZACIÓN/NORMALIZACIÓN DE VARIABLES NUMÉRICAS
# ============================================================================

def escalar_variables_numericas(df_clean):
    print("\n" + "=" * 80)
    print("ESTANDARIZACIÓN DE VARIABLES NUMÉRICAS")
    print("=" * 80)
    
    # 7.1. Seleccionar variables numéricas para estandarizar
    numeric_to_scale = ['Year', 'Engine Size', 'Mileage']
    
    print("\n1. ESTADÍSTICAS ANTES DE LA ESTANDARIZACIÓN:")
    print("-" * 40)
    for col in numeric_to_scale:
        if col in df_clean.columns:
            print(f"{col}: Media={df_clean[col].mean():.2f}, Desv={df_clean[col].std():.2f}")
    
    # 7.2. Estandarización manual (Z-score)
    print("\n2. APLICANDO ESTANDARIZACIÓN Z-SCORE:")
    print("-" * 40)
    
    for col in numeric_to_scale:
        if col in df_clean.columns:
            mean_val = df_clean[col].mean()
            std_val = df_clean[col].std()
            df_clean[f'{col}_standardized'] = (df_clean[col] - mean_val) / std_val
            print(f"{col}: Nueva columna '{col}_standardized' creada")
    
    # 7.3. Normalización Min-Max para Price (variable objetivo)
    print("\n3. NORMALIZACIÓN MIN-MAX PARA PRECIO:")
    print("-" * 40)
    
    if 'Price' in df_clean.columns:
        min_price = df_clean['Price'].min()
        max_price = df_clean['Price'].max()
        df_clean['Price_normalized'] = (df_clean['Price'] - min_price) / (max_price - min_price)
        print(f"Price: Normalizado entre 0 y 1")
        print(f"  Min original: {min_price:.2f}, Max original: {max_price:.2f}")
        print(f"  Min normalizado: {df_clean['Price_normalized'].min():.4f}")
        print(f"  Max normalizado: {df_clean['Price_normalized'].max():.4f}")
    
    return df_clean