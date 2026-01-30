# ============================================================================
# MODELADO Y COMPARACIÓN DE RESULTADOS - CLASIFICACIÓN
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def inspeccionar_dataset_procesado():
    print("=" * 80)
    print("MODELADO Y COMPARACIÓN DE RESULTADOS - CLASIFICACIÓN")
    print("=" * 80)
    
    # Cargar dataset limpio
    df_clean = pd.read_csv('car_price_cleaned.csv')
    
    print("\n1. INSPECCIÓN DEL DATASET PROCESADO:")
    print("-" * 40)
    print(f"Dimensiones: {df_clean.shape}")
    print(f"\nVariables disponibles:")
    print(df_clean.columns.tolist())
    
    return df_clean

def graficar_distribucion_objetivo(df_clean):
    # Variable objetivo: Condition
    print("Variable objetivo: Condition")
    print("Distribución de clases:")
    condition_dist = df_clean['Condition'].value_counts()
    print(condition_dist)
    
    # Visualizar distribución de la variable objetivo
    plt.figure(figsize=(10, 6))
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    bars = plt.bar(condition_dist.index, condition_dist.values, color=colors, edgecolor='black')
    plt.title('Distribución de la Variable Objetivo: Condition', fontsize=16, fontweight='bold')
    plt.xlabel('Condición del Vehículo', fontsize=12)
    plt.ylabel('Cantidad de Vehículos', fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    
    # Agregar valores encima de las barras
    for bar, value in zip(bars, condition_dist.values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{value}\\n({value/len(df_clean)*100:.1f}%)',
                 ha='center', va='bottom', fontsize=11)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('distribucion_condicion.png', dpi=300, bbox_inches='tight')
    plt.show()

def preparar_caracteristicas(df_clean):
    print("\n3. SELECCIÓN Y PREPARACIÓN DE FEATURES:")
    print("-" * 40)
    
    # Seleccionar características relevantes para predecir la condición
    features = [
        # Características técnicas
        'Year',
        'Engine Size',
        'Mileage',
        
        # Características codificadas
        'Brand_encoded', # Usar la versión codificada de la marca
        'Condition',  # Usaremos esta como variable objetivo
        
        # Variables dummy de tipo de combustible
        'Fuel_Type_Diesel',
        'Fuel_Type_Electric',
        'Fuel_Type_Hybrid',
        'Fuel_Type_Petrol',
        
        # Variables dummy de transmisión
        'Transmission_Automatic',
        'Transmission_Manual',
        
        # Características estandarizadas para evitar problemas de escala en algunos modelos
        'Year_standardized',
        'Engine Size_standardized',
        'Mileage_standardized'
    ]
    
    # Verificar qué features existen realmente en el dataset
    existing_features = [f for f in features if f in df_clean.columns]
    print(f"Features seleccionadas: {len(existing_features)}")
    print(f"\nLista de features:")
    for i, feature in enumerate(existing_features, 1):
        print(f"{i:2d}. {feature}")
    
    # Separar características (X) y variable objetivo (y)
    X = df_clean[existing_features].copy()
    
    # Asegurarnos de que Condition esté en X para luego separarlo
    if 'Condition' in X.columns:
        # Variable objetivo: Condition (valores: 0=Used, 1=Like New, 2=New)
        y = X['Condition']
        X = X.drop('Condition', axis=1)
    else:
        # Si no está, usar la columna original Condition y convertirla
        condition_mapping = {'Used': 0, 'Like New': 1, 'New': 2}
        y = df_clean['Condition'].map(condition_mapping)
    
    print(f"\nDimensiones de X: {X.shape}")
    
    # Verificar balance de clases
    print("\nDistribución de clases en y:")
    class_dist = pd.Series(y).value_counts().sort_index()
    class_names = {0: 'Used', 1: 'Like New', 2: 'New'}
    for class_id, count in class_dist.items():
        class_name = class_names.get(class_id, f'Class_{class_id}')
        percentage = count / len(y) * 100
        print(f"  {class_name}: {count} muestras ({percentage:.1f}%)")
    
    return X, y, class_names

def dividir_datos(X, y):
    print("\n" + "=" * 80)
    print("DIVISIÓN DE DATOS EN ENTRENAMIENTO Y PRUEBA")
    print("=" * 80)
    
    # Dividir los datos (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTamaño del conjunto de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Tamaño del conjunto de prueba: {X_test.shape[0]} muestras")
    
    return X_train, X_test, y_train, y_test

def estandarizar_manual(train_data, test_data, columns):
    """
    Estandarización manual de columnas específicas
    Formula: (x - mean) / std
    """
    train_scaled = train_data.copy()
    test_scaled = test_data.copy()

    for col in columns:
        if col in train_data.columns:
            # Calcular media y desviación estándar del conjunto de entrenamiento
            mean_val = train_data[col].mean()
            std_val = train_data[col].std()

            # Aplicar transformación a entrenamiento y prueba
            if std_val > 0:  # Evitar división por cero
                train_scaled[col] = (train_data[col] - mean_val) / std_val
                test_scaled[col] = (test_data[col] - mean_val) / std_val
            else:
                train_scaled[col] = 0
                test_scaled[col] = 0

    return train_scaled, test_scaled

def escalar_caracteristicas(X_train, X_test):
    print("\n" + "=" * 80)
    print("ESCALADO DE CARACTERÍSTICAS NUMÉRICAS")
    print("=" * 80)
    
    # Identificar columnas numéricas que no son dummy (0/1)
    numeric_cols = ['Year', 'Engine Size', 'Mileage', 'Brand_encoded']
    # Filtrar las que existen en X
    numeric_cols = [col for col in numeric_cols if col in X_train.columns]
    
    print(f"\nColumnas numéricas a escalar: {numeric_cols}")
    
    # Aplicar escalado
    X_train_scaled, X_test_scaled = estandarizar_manual(X_train, X_test, numeric_cols)
    
    print("\nEscalado completado.")
    print("\nEstadísticas después del escalado (primeras 5 muestras de entrenamiento):")
    for col in numeric_cols[:3]:  # Mostrar solo las primeras 3 para no saturar
        print(f"\n{col}:")
        print(f"  Media: {X_train_scaled[col].mean():.4f}")
        print(f"  Desviación estándar: {X_train_scaled[col].std():.4f}")
    
    return X_train_scaled, X_test_scaled

def entrenar_regresion_logistica(X_train_scaled, y_train):
    print("\n" + "=" * 80)
    print("MODELO 1: REGRESIÓN LOGÍSTICA MULTICLASE")
    print("=" * 80)
    
    # Crear y entrenar el modelo de Regresión Logística
    print("\nEntrenando Regresión Logística...")
    logreg_model = LogisticRegression(
        solver='lbfgs',             # Algoritmo adecuado para multiclase
        max_iter=1000,              # Número máximo de iteraciones
        random_state=42,            # Para reproducibilidad
        C=1.0                       # Parámetro de regularización
    )
    
    logreg_model.fit(X_train_scaled, y_train)
    print("Regresión Logística entrenada exitosamente.")
    
    return logreg_model

def entrenar_random_forest(X_train_scaled, y_train):
    print("\n" + "=" * 80)
    print("MODELO 2: RANDOM FOREST")
    print("=" * 80)
    
    # Crear y entrenar el modelo de Random Forest
    print("\nEntrenando Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,      # Número de árboles en el bosque
        max_depth=10,          # Profundidad máxima de cada árbol
        min_samples_split=5,   # Mínimo de muestras para dividir un nodo
        min_samples_leaf=2,    # Mínimo de muestras en una hoja
        random_state=42,       # Para reproducibilidad
        n_jobs=-1              # Usar todos los núcleos disponibles
    )
    
    rf_model.fit(X_train_scaled, y_train)
    print("Random Forest entrenado exitosamente.")
    
    return rf_model

def calcular_metricas(y_true, y_pred, y_proba, model_name):
    """Calcula y muestra métricas de evaluación para un modelo"""
    
    print(f"\n{model_name}")
    print("-" * 40)
    
    # Métricas básicas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # AUC-ROC (para multiclase, one-vs-rest)
    try:
        auc_score = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
        print(f"AUC-ROC:   {auc_score:.4f}")
    except:
        print("AUC-ROC:   No calculado")
    
    # Reporte de clasificación detallado
    print("\nReporte de Clasificación Detallado:")
    print(classification_report(y_true, y_pred,
                              target_names=['Used', 'Like New', 'New']))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def evaluar_modelos(logreg_model, rf_model, X_test_scaled, y_test):
    print("\n" + "=" * 80)
    print("EVALUACIÓN Y COMPARACIÓN DE MODELOS")
    print("=" * 80)
    
    # Hacer predicciones
    y_pred_logreg = logreg_model.predict(X_test_scaled)
    y_pred_proba_logreg = logreg_model.predict_proba(X_test_scaled)
    
    y_pred_rf = rf_model.predict(X_test_scaled)
    y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)
    
    # Evaluar Regresión Logística
    print("\nEVALUACIÓN DEL MODELO:")
    metrics_logreg = calcular_metricas(y_test, y_pred_logreg, y_pred_proba_logreg,
                                      "REGRESIÓN LOGÍSTICA")
    
    # Evaluar Random Forest
    metrics_rf = calcular_metricas(y_test, y_pred_rf, y_pred_proba_rf,
                                  "RANDOM FOREST")
    
    return y_pred_logreg, y_pred_proba_logreg, y_pred_rf, y_pred_proba_rf, metrics_logreg, metrics_rf

def crear_visualizaciones_comparativas(y_test, y_pred_logreg, y_pred_proba_logreg, 
                                    y_pred_rf, y_pred_proba_rf, metrics_logreg, 
                                    metrics_rf, X, X_test_scaled, rf_model):
    print("\n" + "=" * 80)
    print("VISUALIZACIÓN COMPARATIVA DE RESULTADOS")
    print("=" * 80)
    
    # Crear figura con múltiples subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 7.1. Matrices de confusión
    print("\nGenerando visualizaciones...")
    
    # Matriz de confusión - Regresión Logística
    ax1 = plt.subplot(3, 3, 1)
    cm_logreg = confusion_matrix(y_test, y_pred_logreg)
    sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Used', 'Like New', 'New'],
                yticklabels=['Used', 'Like New', 'New'])
    ax1.set_title('Matriz de Confusión - Regresión Logística', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicción')
    ax1.set_ylabel('Real')
    
    # Matriz de confusión - Random Forest
    ax2 = plt.subplot(3, 3, 2)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Used', 'Like New', 'New'],
                yticklabels=['Used', 'Like New', 'New'])
    ax2.set_title('Matriz de Confusión - Random Forest', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicción')
    ax2.set_ylabel('Real')
    
    # 7.2. Comparación de métricas
    ax3 = plt.subplot(3, 3, 3)
    models = ['Regresión Logística', 'Random Forest']
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1']
    
    # Preparar datos para el gráfico de barras
    metric_values = {
        'Regresión Logística': [
            metrics_logreg['accuracy'],
            metrics_logreg['precision'],
            metrics_logreg['recall'],
            metrics_logreg['f1']
        ],
        'Random Forest': [
            metrics_rf['accuracy'],
            metrics_rf['precision'],
            metrics_rf['recall'],
            metrics_rf['f1']
        ]
    }
    
    x = np.arange(len(metrics_to_compare))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, metric_values['Regresión Logística'], width,
                    label='Regresión Logística', color='skyblue', edgecolor='black')
    bars2 = ax3.bar(x + width/2, metric_values['Random Forest'], width,
                    label='Random Forest', color='lightgreen', edgecolor='black')
    
    ax3.set_xlabel('Métricas')
    ax3.set_ylabel('Valor')
    ax3.set_title('Comparación de Métricas por Modelo', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics_to_compare)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Agregar valores encima de las barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 7.3. Importancia de características (solo para Random Forest)
    ax4 = plt.subplot(3, 3, 4)
    feature_importance = rf_model.feature_importances_
    feature_names = X.columns
    
    # Ordenar características por importancia
    indices = np.argsort(feature_importance)[::-1]
    top_features = 10  # Mostrar las 10 más importantes
    
    ax4.barh(range(top_features), feature_importance[indices[:top_features]],
             color='coral', edgecolor='black')
    ax4.set_yticks(range(top_features))
    ax4.set_yticklabels([feature_names[i] for i in indices[:top_features]])
    ax4.invert_yaxis()
    ax4.set_xlabel('Importancia')
    ax4.set_title('Top 10 Características Más Importantes (Random Forest)',
                  fontsize=14, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    
    # 7.4. Gráfico de barras de accuracy por clase
    ax5 = plt.subplot(3, 3, 5)
    
    # Calcular accuracy por clase para cada modelo
    classes = ['Used', 'Like New', 'New']
    logreg_class_acc = []
    rf_class_acc = []
    
    for i, class_name in enumerate(classes):
        # Índices donde la clase real es i
        idx_class = (y_test == i)
        
        if sum(idx_class) > 0:
            # Accuracy para esta clase
            logreg_acc_class = accuracy_score(y_test[idx_class], y_pred_logreg[idx_class])
            rf_acc_class = accuracy_score(y_test[idx_class], y_pred_rf[idx_class])
        else:
            logreg_acc_class = 0
            rf_acc_class = 0
        
        logreg_class_acc.append(logreg_acc_class)
        rf_class_acc.append(rf_acc_class)
    
    x_class = np.arange(len(classes))
    width_class = 0.35
    
    bars_logreg_class = ax5.bar(x_class - width_class/2, logreg_class_acc, width_class,
                                label='Regresión Logística', color='skyblue', edgecolor='black')
    bars_rf_class = ax5.bar(x_class + width_class/2, rf_class_acc, width_class,
                            label='Random Forest', color='lightgreen', edgecolor='black')
    
    ax5.set_xlabel('Clase')
    ax5.set_ylabel('Accuracy por Clase')
    ax5.set_title('Accuracy por Clase de Condición', fontsize=14, fontweight='bold')
    ax5.set_xticks(x_class)
    ax5.set_xticklabels(classes)
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    # 7.5. Curva ROC multiclase (One-vs-Rest)
    ax6 = plt.subplot(3, 3, 6)
    
    # Binarizar las etiquetas para curva ROC
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_test_bin.shape[1]
    
    # Calcular curva ROC para cada clase y cada modelo
    fpr_logreg = dict()
    tpr_logreg = dict()
    roc_auc_logreg = dict()
    
    fpr_rf = dict()
    tpr_rf = dict()
    roc_auc_rf = dict()
    
    for i in range(n_classes):
        # Regresión Logística
        fpr_logreg[i], tpr_logreg[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba_logreg[:, i])
        roc_auc_logreg[i] = auc(fpr_logreg[i], tpr_logreg[i])
        
        # Random Forest
        fpr_rf[i], tpr_rf[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba_rf[:, i])
        roc_auc_rf[i] = auc(fpr_rf[i], tpr_rf[i])
    
    # Plotear todas las curvas ROC
    colors = ['blue', 'red', 'green']
    class_names_roc = ['Used', 'Like New', 'New']
    
    for i, color in zip(range(n_classes), colors):
        # Regresión Logística
        ax6.plot(fpr_logreg[i], tpr_logreg[i], color=color, lw=2,
                 label=f'LogReg {class_names_roc[i]} (AUC = {roc_auc_logreg[i]:.2f})')
        
        # Random Forest (línea punteada)
        ax6.plot(fpr_rf[i], tpr_rf[i], color=color, lw=2, linestyle='--',
                 label=f'RF {class_names_roc[i]} (AUC = {roc_auc_rf[i]:.2f})')
    
    ax6.plot([0, 1], [0, 1], 'k--', lw=2)  # Línea de referencia
    ax6.set_xlim([0.0, 1.0])
    ax6.set_ylim([0.0, 1.05])
    ax6.set_xlabel('Tasa de Falsos Positivos')
    ax6.set_ylabel('Tasa de Verdaderos Positivos')
    ax6.set_title('Curvas ROC por Clase (One-vs-Rest)', fontsize=14, fontweight='bold')
    ax6.legend(loc="lower right", fontsize=9)
    ax6.grid(alpha=0.3)
    
    # 7.6. Distribución de probabilidades para una clase específica (ej: "New")
    ax7 = plt.subplot(3, 3, 7)
    
    # Seleccionar la clase "New" (índice 2)
    class_idx = 2
    class_name = "New"
    
    # Obtener probabilidades para la clase "New"
    logreg_probs_new = y_pred_proba_logreg[:, class_idx]
    rf_probs_new = y_pred_proba_rf[:, class_idx]
    
    # Filtrar instancias donde la clase real es "New"
    real_new_idx = (y_test == class_idx)
    logreg_probs_real_new = logreg_probs_new[real_new_idx]
    rf_probs_real_new = rf_probs_new[real_new_idx]
    
    # Crear histogramas
    ax7.hist(logreg_probs_real_new, bins=20, alpha=0.5, label='Regresión Logística',
             color='blue', edgecolor='black')
    ax7.hist(rf_probs_real_new, bins=20, alpha=0.5, label='Random Forest',
             color='green', edgecolor='black')
    
    ax7.set_xlabel(f'Probabilidad predicha para "{class_name}"')
    ax7.set_ylabel('Frecuencia')
    ax7.set_title(f'Distribución de Probabilidades (Clase Real = {class_name})',
                  fontsize=14, fontweight='bold')
    ax7.legend()
    ax7.grid(alpha=0.3)
    
    # 7.7. Tiempo de entrenamiento y predicción (simulado)
    ax8 = plt.subplot(3, 3, 8)
    
    # Valores simulados para tiempo (en segundos)
    train_times = [0.8, 3.5]  # Regresión Logística vs Random Forest
    pred_times = [0.02, 0.15]  # Regresión Logística vs Random Forest
    
    x_time = np.arange(len(models))
    width_time = 0.35
    
    bars_train = ax8.bar(x_time - width_time/2, train_times, width_time,
                         label='Entrenamiento', color='orange', edgecolor='black')
    bars_pred = ax8.bar(x_time + width_time/2, pred_times, width_time,
                        label='Predicción', color='purple', edgecolor='black')
    
    ax8.set_xlabel('Modelo')
    ax8.set_ylabel('Tiempo (segundos)')
    ax8.set_title('Tiempo de Ejecución (Simulado)', fontsize=14, fontweight='bold')
    ax8.set_xticks(x_time)
    ax8.set_xticklabels(models)
    ax8.legend()
    ax8.grid(axis='y', alpha=0.3)
    
    # 7.8. Resumen de resultados
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')  # Ocultar ejes
    
    # Crear tabla de resumen
    summary_data = [
        ['Métrica', 'Regresión Logística', 'Random Forest', 'Diferencia'],
        ['Accuracy', f"{metrics_logreg['accuracy']:.4f}", f"{metrics_rf['accuracy']:.4f}",
         f"{metrics_rf['accuracy'] - metrics_logreg['accuracy']:.4f}"],
        ['Precision', f"{metrics_logreg['precision']:.4f}", f"{metrics_rf['precision']:.4f}",
         f"{metrics_rf['precision'] - metrics_logreg['precision']:.4f}"],
        ['Recall', f"{metrics_logreg['recall']:.4f}", f"{metrics_rf['recall']:.4f}",
         f"{metrics_rf['recall'] - metrics_logreg['recall']:.4f}"],
        ['F1-Score', f"{metrics_logreg['f1']:.4f}", f"{metrics_rf['f1']:.4f}",
         f"{metrics_rf['f1'] - metrics_logreg['f1']:.4f}"]
    ]
    
    # Crear tabla
    table = ax9.table(cellText=summary_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Estilo de la tabla
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Encabezado
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2E86AB')
        elif j == 3:  # Columna de diferencia
            diff_value = float(cell.get_text().get_text())
            if diff_value > 0:
                cell.set_facecolor('#A7D49B')  # Verde para mejor
            elif diff_value < 0:
                cell.set_facecolor('#F4B6C2')  # Rojo para peor
    
    ax9.set_title('Resumen Comparativo de Resultados', fontsize=14, fontweight='bold',
                  y=0.95, x=0.5)
    
    # Ajustar layout
    plt.suptitle('COMPARACIÓN DE MODELOS DE CLASIFICACIÓN - CONDICIÓN DE VEHÍCULOS',
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('comparacion_modelos_clasificacion.png', dpi=300, bbox_inches='tight')
    plt.show()