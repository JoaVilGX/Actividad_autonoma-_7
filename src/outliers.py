# ============================================================================
# MANEJO DE OUTLIERS
# ============================================================================
import numpy as np

def detectar_y_manejar_outliers(df, column, method='cap'):
    # Detecta y maneja outliers en una columna específica
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    outliers_count = outliers_mask.sum()

    print(f"{column}:")
    print(f"  - IQR: {IQR:.2f}")
    print(f"  - Límites: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"  - Outliers detectados: {outliers_count}")

    if method == 'cap' and outliers_count > 0:
        # Limitar los valores a los límites
        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
        print(f"  - Método aplicado: Capping")
    elif method == 'remove' and outliers_count > 0:
        # Eliminar outliers
        df = df[~outliers_mask]
        print(f"  - Método aplicado: Eliminación ({outliers_count} filas eliminadas)")

    return df

def manejar_outliers(df_clean):
    print("\n" + "=" * 80)
    print("MANEJO DE OUTLIERS")
    print("=" * 80)
    
    # 5.2. Aplicar manejo de outliers a variables numéricas clave
    print("\nDETECCIÓN Y MANEJO DE OUTLIERS:")
    print("-" * 40)
    
    # Para Price y Mileage usaremos capping (limitar)
    df_clean = detectar_y_manejar_outliers(df_clean, 'Price', method='cap')
    df_clean = detectar_y_manejar_outliers(df_clean, 'Mileage', method='cap')
    
    return df_clean