import os
import matplotlib
matplotlib.use('Agg')  # Utilizamos el backend no interactivo "Agg" para evitar errores de Tkinter
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

# Módulos para selección de variables, modelado y evaluación
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut, learning_curve, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

# Para paralelización
from joblib import Parallel, delayed

# Módulos para abrir ventanas de diálogo y seleccionar archivos
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Se oculta la ventana principal de Tkinter
root = Tk()
root.withdraw()

# =============================================================================
# CREACIÓN DE SUBCARPETAS PARA GUARDAR LAS IMÁGENES
# =============================================================================
base_folder = r'C:\Linux'
subfolders = {
    "resultados": "Resultados",
    "curvas": "Curvas de aprendizaje",
    "diagramas": "Diagramas de residuales",
    "histogramas": "Histogramas de residuales"
}
for folder in subfolders.values():
    os.makedirs(os.path.join(base_folder, folder), exist_ok=True)

# =============================================================================
# 1. CARGA DE DATOS Y REDUCCIÓN DE VARIABLES (CON CAPACIDAD NO LINEAL)
# =============================================================================
# Selección interactiva del archivo Excel para el TRAINING SET.
print("Seleccione el archivo Excel para el Training Set.")
training_file = askopenfilename(title="Seleccione el archivo Excel del Training Set",
                                filetypes=[("Archivos Excel", "*.xlsx")])
training_set = pd.read_excel(training_file)
print(f"Archivo de training set cargado: {training_file}")

# Se asume que los datos ya están estandarizados por z-score.
# La primera columna contiene información para el usuario y no se utiliza en el análisis.
# Usamos todas las columnas excepto la primera y extraemos 'ΔΔG' como variable respuesta.
X = training_set.iloc[:, 1:].drop(columns=['ΔΔG'])
y = training_set['ΔΔG']

# --- Optimización de alpha y selección de variables con LASSO ---
# Usamos LassoCV para obtener el mejor parámetro de regularización (alpha).
alphas = np.logspace(-4, 1, 50)  # Valores de alpha entre 1e-4 y 10
lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42, n_jobs=-1, max_iter=100000)
lasso_cv.fit(X, y)
best_alpha_cv = lasso_cv.alpha_
print("Mejor alpha encontrado en LassoCV:", best_alpha_cv)

# Con el alpha óptimo, ajustamos LASSO para determinar los coeficientes.
lasso_model = Lasso(alpha=best_alpha_cv, random_state=42, max_iter=10000)
lasso_model.fit(X, y)
coef = lasso_model.coef_
# Seleccionamos las variables cuyos coeficientes sean distintos de cero
selected_idx = np.where(coef != 0)[0]
selected_vars = X.columns[selected_idx]
X_reduced = X[selected_vars]

print("\nVARIABLES SELECCIONADAS CON LASSO:")
for var in selected_vars:
    print(var)
print(f"Número total de variables seleccionadas: {len(selected_vars)}")

# Guardar las variables utilizadas en un archivo de salida
data_selected = pd.concat([training_set.iloc[:, :1], X[selected_vars]], axis=1)
output_vars_path = os.path.join(base_folder, 'variables.xlsx')
data_selected.to_excel(output_vars_path, index=False, engine='openpyxl')
print(f"\nArchivo de variables utilizadas guardado en: {output_vars_path}")

# =============================================================================
# 2. CARGA DE DATOS PARA PREDICCIÓN EN NUEVOS CASOS (VALIDATION SET)
# =============================================================================
print("Seleccione el archivo Excel para el Validation Set.")
validation_file = askopenfilename(title="Seleccione el archivo Excel del Validation Set",
                                  filetypes=[("Archivos Excel", "*.xlsx")])
validation_set = pd.read_excel(validation_file)
print(f"Archivo de validation set cargado: {validation_file}")

# Se evita únicamente la primera columna (información de usuario) y se ignora 'ΔΔG' si existe.
X_validation = validation_set.iloc[:, 1:].drop(columns=['ΔΔG'], errors='ignore')
# Se aseguran las mismas variables seleccionadas en el entrenamiento.
X_validation_reduced = X_validation[selected_vars]

# =============================================================================
# 3. DEFINICIÓN Y OPTIMIZACIÓN DEL MODELO: LASSO
# =============================================================================
# Configuración de validación cruzada para búsqueda de hiperparámetros (k=5)
cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

# Refinamiento de la búsqueda de hiperparámetros para LASSO.
param_grid_lasso = {'alpha': np.logspace(-4, 1, 50)}
grid_search = GridSearchCV(Lasso(random_state=42, max_iter=10000), 
                           param_grid_lasso, cv=cv_strategy, scoring='r2', n_jobs=-1)
grid_search.fit(X_reduced, y)
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("\nMEJOR PARÁMETRO PARA LASSO:")
print(best_params)
print(f"Mejor R² en validación (GridSearchCV): {best_score:.4f}")

# Entrenamiento del modelo final LASSO con el mejor parámetro sobre todo el conjunto
final_model = Lasso(alpha=best_params['alpha'], random_state=42, max_iter=10000)
final_model.fit(X_reduced, y)

# =============================================================================
# 3.1 VALIDACIÓN INTERNA CON LOOCV (PARALELIZADO)
# =============================================================================
def loocv_predict(train_idx, test_idx):
    X_train = X_reduced.iloc[train_idx]
    X_test = X_reduced.iloc[test_idx]
    y_train = y.iloc[train_idx]
    # Se utiliza LASSO con el alpha optimizado y mayor cantidad de iteraciones
    model_cv = Lasso(alpha=best_params['alpha'], random_state=42, max_iter=10000)
    model_cv.fit(X_train, y_train)
    return model_cv.predict(X_test)[0]

loo = LeaveOneOut()
# Procesa cada iteración LOOCV en paralelo usando todos los núcleos disponibles.
y_pred_loocv = Parallel(n_jobs=-1)(
    delayed(loocv_predict)(train_idx, test_idx) for train_idx, test_idx in loo.split(X_reduced)
)
y_pred_loocv = np.array(y_pred_loocv)

# Cálculo de métricas para LOOCV:
r2_loocv = r2_score(y, y_pred_loocv)
rmse_loocv = np.sqrt(mean_squared_error(y, y_pred_loocv))
mape_loocv = mean_absolute_percentage_error(y, y_pred_loocv) * 100  # Expresado en %

print(f"\nEvaluación LOOCV: R² = {r2_loocv:.4f}, RMSE = {rmse_loocv:.4f}, MAPE = {mape_loocv:.2f}%")

# =============================================================================
# VALIDACIÓN INTERNA CON K-FOLD (5-FOLD)
# =============================================================================
y_pred_kfold = cross_val_predict(Lasso(alpha=best_params['alpha'], random_state=42, max_iter=10000),
                                 X_reduced, y, cv=cv_strategy, n_jobs=-1)
r2_kfold = r2_score(y, y_pred_kfold)
rmse_kfold = np.sqrt(mean_squared_error(y, y_pred_kfold))
mape_kfold = mean_absolute_percentage_error(y, y_pred_kfold) * 100  # Expresado en %

print(f"\nEvaluación 5-Fold Cross-Validation: R² = {r2_kfold:.4f}, RMSE = {rmse_kfold:.4f}, MAPE = {mape_kfold:.2f}%")

# =============================================================================
# 3.2 PREDICCIÓN EN NUEVOS DATOS (VALIDATION SET)
# =============================================================================
validation_set_copy = validation_set.copy()
predictions_validation = final_model.predict(X_validation_reduced)
validation_set_copy[f'Predicted_ΔΔG_LASSO'] = predictions_validation
output_path = os.path.join(base_folder, 'validation_predictions_LASSO.xlsx')
validation_set_copy.to_excel(output_path, index=False)
print("Predicciones en nuevos datos guardadas en:", output_path)

# =============================================================================
# 3.3 GRÁFICO DE RESULTADOS: EXPERIMENTAL vs. PREDICHO
# =============================================================================
plt.figure(figsize=(8, 6))
# Predicciones en el conjunto de entrenamiento
train_pred = final_model.predict(X_reduced)
plt.scatter(y, train_pred, color='gray', alpha=0.6, label='Entrenamiento')
# Predicciones en nuevos datos: se usa 'ΔΔG' si está presente o el índice
if 'ΔΔG' in validation_set_copy.columns:
    plt.scatter(validation_set_copy['ΔΔG'], predictions_validation, marker='x', color='blue', label='Nuevos datos')
    all_measured = np.concatenate([y.values, validation_set_copy['ΔΔG'].values])
else:
    plt.scatter(range(len(predictions_validation)), predictions_validation, marker='x', color='blue', label='Nuevos datos')
    all_measured = y.values
min_val = np.min(all_measured)
max_val = np.max(all_measured)
plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='black')
plt.xlabel("ΔΔG‡ Experimental (kcal/mol)")
plt.ylabel("ΔΔG‡ Predicho (kcal/mol)")
plt.title("Experimental vs. Predicho - LASSO")
plt.legend()
plt.grid(True)
plt.tight_layout()
output_file = os.path.join(base_folder, subfolders["resultados"], 'LASSO - Experimental vs. Predicho.png')
plt.savefig(output_file)
print("Gráfico Experimental vs. Predicho guardado en:", output_file)
plt.close()

# =============================================================================
# 3.4 ANÁLISIS DE SOBREAJUSTE:
#      - Curva de Aprendizaje
#      - Diagrama de Residuales
#      - Histograma de Residuales
# =============================================================================
# --- Curva de Aprendizaje ---
fractions = [0.10, 0.25, 0.50, 0.75, 1.0]
train_sizes_frac, train_scores_frac, val_scores_frac = learning_curve(
    estimator=final_model,
    X=X_reduced,
    y=y,
    cv=cv_strategy,
    train_sizes=fractions,
    scoring='r2',
    n_jobs=-1
)
train_scores_mean_frac = np.mean(train_scores_frac, axis=1)
val_scores_mean_frac = np.mean(val_scores_frac, axis=1)
    
plt.figure(figsize=(8, 6))
plt.plot(train_sizes_frac, train_scores_mean_frac, 'o-', color='r', label='R² en Entrenamiento')
plt.plot(train_sizes_frac, val_scores_mean_frac, 'o-', color='g', label='R² en Validación')
plt.xlabel("Tamaño del conjunto de entrenamiento")
plt.ylabel("R²")
plt.title("Curva de Aprendizaje - LASSO")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
output_file = os.path.join(base_folder, subfolders["curvas"], 'LASSO - Curva de Aprendizaje.png')
plt.savefig(output_file)
print("Curva de Aprendizaje guardada en:", output_file)
plt.close()

print("\nResultados de la curva de aprendizaje en puntos específicos:")
for frac, size, tr_score, val_score in zip(fractions, train_sizes_frac, train_scores_mean_frac, val_scores_mean_frac):
    print(f"Con {frac*100:.0f}% (tamaño = {int(size)} muestras): R² Entrenamiento = {tr_score:.4f}, R² Validación = {val_score:.4f}")

# --- Diagrama de Residuales ---
residuals = y - y_pred_loocv
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_loocv, residuals, alpha=0.5, color='b')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Valores Predichos (LOOCV)")
plt.ylabel("Residuales")
plt.title("Diagrama de Residuales - LASSO")
plt.grid(True)
plt.tight_layout()
output_file = os.path.join(base_folder, subfolders["diagramas"], 'LASSO - Diagrama de Residuales.png')
plt.savefig(output_file)
print("Diagrama de Residuales guardado en:", output_file)
plt.close()

# --- Histograma de Residuales ---
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, color='c', edgecolor='black')
plt.xlabel("Residuales")
plt.ylabel("Frecuencia")
plt.title("Histograma de Residuales - LASSO")
plt.grid(True)
plt.tight_layout()
output_file = os.path.join(base_folder, subfolders["histogramas"], 'LASSO - Histograma de Residuales.png')
plt.savefig(output_file)
print("Histograma de Residuales guardado en:", output_file)
plt.close()
