# ============================================================================
# VISUALIZACIONES EXPLORATORIAS
# ============================================================================
import matplotlib.pyplot as plt
import seaborn as sns

def crear_visualizaciones_exploratorias(df_clean):
    print("\n" + "=" * 80)
    print("VISUALIZACIONES EXPLORATORIAS")
    print("=" * 80)
    
    # Crear figura con múltiples subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 8.1. Distribución de precios
    ax1 = plt.subplot(3, 3, 1)
    df_clean['Price'].hist(bins=30, edgecolor='black', alpha=0.7)
    ax1.set_title('Distribución de Precios')
    ax1.set_xlabel('Precio')
    ax1.set_ylabel('Frecuencia')
    ax1.grid(True, alpha=0.3)
    
    # 8.2. Precio por tipo de combustible
    ax2 = plt.subplot(3, 3, 2)
    fuel_price = df_clean.groupby('Fuel Type')['Price'].mean().sort_values()
    fuel_price.plot(kind='bar', color='skyblue', edgecolor='black')
    ax2.set_title('Precio Promedio por Tipo de Combustible')
    ax2.set_xlabel('Tipo de Combustible')
    ax2.set_ylabel('Precio Promedio')
    ax2.tick_params(axis='x', rotation=45)
    
    # 8.3. Precio por condición del vehículo
    ax3 = plt.subplot(3, 3, 3)
    condition_price = df_clean.groupby('Condition')['Price'].mean()
    condition_order = ['Used', 'Like New', 'New']
    condition_price = condition_price.reindex(condition_order)
    condition_price.plot(kind='bar', color='lightgreen', edgecolor='black')
    ax3.set_title('Precio Promedio por Condición')
    ax3.set_xlabel('Condición')
    ax3.set_ylabel('Precio Promedio')
    
    # 8.4. Relación entre año y precio
    ax4 = plt.subplot(3, 3, 4)
    plt.scatter(df_clean['Year'], df_clean['Price'], alpha=0.5, s=10)
    ax4.set_title('Relación: Año vs Precio')
    ax4.set_xlabel('Año')
    ax4.set_ylabel('Precio')
    ax4.grid(True, alpha=0.3)
    
    # 8.5. Relación entre kilometraje y precio
    ax5 = plt.subplot(3, 3, 5)
    plt.scatter(df_clean['Mileage'], df_clean['Price'], alpha=0.5, s=10, color='red')
    ax5.set_title('Relación: Kilometraje vs Precio')
    ax5.set_xlabel('Kilometraje')
    ax5.set_ylabel('Precio')
    ax5.grid(True, alpha=0.3)
    
    # 8.6. Distribución de marcas (top 10)
    ax6 = plt.subplot(3, 3, 6)
    top_brands = df_clean['Brand'].value_counts().head(10)
    top_brands.plot(kind='bar', color='orange', edgecolor='black')
    ax6.set_title('Top 10 Marcas más Frecuentes')
    ax6.set_xlabel('Marca')
    ax6.set_ylabel('Cantidad')
    ax6.tick_params(axis='x', rotation=45)
    
    # 8.7. Precio por tipo de transmisión
    ax7 = plt.subplot(3, 3, 7)
    transmission_price = df_clean.groupby('Transmission')['Price'].mean()
    transmission_price.plot(kind='bar', color='purple', edgecolor='black')
    ax7.set_title('Precio Promedio por Transmisión')
    ax7.set_xlabel('Tipo de Transmisión')
    ax7.set_ylabel('Precio Promedio')
    
    # 8.8. Distribución de tamaño del motor
    ax8 = plt.subplot(3, 3, 8)
    df_clean['Engine Size'].hist(bins=20, edgecolor='black', alpha=0.7, color='brown')
    ax8.set_title('Distribución del Tamaño del Motor')
    ax8.set_xlabel('Tamaño del Motor')
    ax8.set_ylabel('Frecuencia')
    ax8.grid(True, alpha=0.3)
    
    # 8.9. Matriz de correlación (variables numéricas principales)
    ax9 = plt.subplot(3, 3, 9)
    numeric_cols_corr = ['Year', 'Engine Size', 'Mileage', 'Price']
    correlation_matrix = df_clean[numeric_cols_corr].corr()
    im = ax9.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
    ax9.set_title('Matriz de Correlación')
    ax9.set_xticks(range(len(numeric_cols_corr)))
    ax9.set_yticks(range(len(numeric_cols_corr)))
    ax9.set_xticklabels(numeric_cols_corr, rotation=45)
    ax9.set_yticklabels(numeric_cols_corr)
    plt.colorbar(im, ax=ax9)
    
    # Ajustar layout
    plt.tight_layout()
    plt.savefig('visualizaciones_exploratorias.png', dpi=300, bbox_inches='tight')
    plt.show()