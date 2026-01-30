# ============================================================================
# GUARDAR DATASET LIMPIO
# ============================================================================
import pandas as pd

def guardar_dataset_limpio(df_clean):
    print("\n" + "=" * 80)
    print("GUARDANDO DATASET PROCESADO")
    print("=" * 80)
    
    # Guardar dataset limpio
    df_clean.to_csv('car_price_cleaned.csv', index=False)
    print("Dataset limpio guardado como: 'car_price_cleaned.csv'")
    
    # Mostrar muestra final
    print("\nMUESTRA FINAL DEL DATASET PROCESADO:")
    print("-" * 40)
    print(df_clean.head())
    
    # Estadísticas finales de las nuevas columnas
    print("\nESTADÍSTICAS DE COLUMNAS TRANSFORMADAS:")
    print("-" * 40)
    transformed_cols = [col for col in df_clean.columns if 'encoded' in col or 'standardized' in col or 'normalized' in col]
    for col in transformed_cols:
        if col in df_clean.columns:
            print(f"{col}: Min={df_clean[col].min():.4f}, Max={df_clean[col].max():.4f}, Mean={df_clean[col].mean():.4f}")