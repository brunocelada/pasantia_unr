import logging
import subprocess
import sys
import os

'''
Librerías necesarias:
- pandas
- numpy
- scikit-learn
'''

try:
    import pandas as pd
    logging.info("pandas ya está instalado.")
except ImportError:
    print("pandas no está instalado. Instalando...")
    logging.info("pandas no está instalado. Instalando...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    import pandas as pd

try:
    import numpy as np
    logging.info("numpy ya está instalado.")
except ImportError:
    print("numpy no está instalado. Instalando...")
    logging.info("numpy no está instalado. Instalando...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    import numpy as np


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

# Configurar logging
script_dir = os.path.dirname(os.path.abspath(__file__))  # Carpeta del script
log_path = os.path.join(script_dir, "registro_reduction.log")

logging.basicConfig(
    filename=log_path,
    filemode='w',  # o 'a' para agregar al final
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def cargar_datos(csv_path):
    logging.info("Iniciando lectura del archivo CSV.")
    try:
        # skiprows=[0]: omite la primera fila (vacía o sin relevancia).
        # header=0: usa la primera fila después del skiprows como encabezados.
        df = pd.read_csv(csv_path, skiprows=[0], header=0, sep=";")
        '''Para revisar el tipo de datos y sus valores, descomentar las siguientes líneas:'''
            # print("Columnas del DataFrame:")
            # print(df.columns)
            # print(df.head())

        df.reset_index(drop=True, inplace=True)
        logging.info(f"Archivo leído correctamente con {df.shape[0]} reacciones y {df.shape[1]-1} variables.")
        return df
    except Exception as e:
        logging.error(f"Error al leer el CSV: {e}")
        raise

def preparar_datos(df):
    y = df.iloc[:, -1]  # Se asume que la última columna es el resultado
    X = df.iloc[:, 1:-1]  # Desde columna 2 hasta la penúltima
    nombres_variables = X.columns
    '''Para revisar el tipo de datos y sus valores, descomentar las siguientes líneas:'''
        # print("Contenido de X:")
        # print(X.head())
        # print("Tipo de datos:")
        # print(X.dtypes)

    X_scaled = StandardScaler().fit_transform(X)
    logging.info("Datos escalados correctamente.")
    return X, X_scaled, y, nombres_variables

def aplicar_pca(X_scaled, nombres_variables, output_dir="C:\\Linux", varianza_deseada=0.95):
    logging.info("Aplicando PCA.")
    pca = PCA(n_components=varianza_deseada)
    X_pca = pca.fit_transform(X_scaled)
    componentes = pca.components_
    n_vars = X_pca.shape[1]
    logging.info(f"PCA conservó {n_vars} componentes que explican el {varianza_deseada*100:.1f}% de la varianza.")
    df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_vars)])
    df_pca.to_csv(os.path.join(output_dir, "pca_resultados.csv"), index=False)
    return df_pca

def aplicar_pls(X_scaled, y, output_dir="C:\\Linux", n_comp=10):
    logging.info("Aplicando PLS.")
    pls = PLSRegression(n_components=min(n_comp, X_scaled.shape[1]))
    pls.fit(X_scaled, y)
    X_pls = pls.transform(X_scaled)
    df_pls = pd.DataFrame(X_pls, columns=[f'PLS{i+1}' for i in range(X_pls.shape[1])])
    df_pls.to_csv(os.path.join(output_dir, "pls_resultados.csv"), index=False)
    logging.info(f"PLS generó {X_pls.shape[1]} componentes.")
    return df_pls

def aplicar_lasso(X_scaled, y, nombres_variables, output_dir="C:\\Linux"):
    logging.info("Aplicando LASSO.")
    lasso = LassoCV(cv=5, random_state=42).fit(X_scaled, y)
    seleccionadas = nombres_variables[lasso.coef_ != 0]
    logging.info(f"LASSO seleccionó {len(seleccionadas)} variables.")
    for var, coef in zip(nombres_variables, lasso.coef_):
        if coef != 0:
            logging.info(f"Variable conservada por LASSO: {var} (coef={coef:.4f})")
        else:
            logging.info(f"Variable eliminada por LASSO: {var}")
    df_lasso = pd.DataFrame(X_scaled[:, lasso.coef_ != 0], columns=seleccionadas)
    df_lasso.to_csv(os.path.join(output_dir, "lasso_resultados.csv"), index=False)
    return df_lasso

def aplicar_random_forest(X, y, nombres_variables, output_dir="C:\\Linux", top_n=30):
    logging.info("Aplicando Random Forest para selección de variables.")
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X, y)
    importancias = rf.feature_importances_
    indices_top = np.argsort(importancias)[-top_n:]
    seleccionadas = nombres_variables[indices_top]
    logging.info(f"Random Forest seleccionó las {top_n} variables más importantes.")
    for idx in indices_top:
        logging.info(f"Variable conservada por RF: {nombres_variables[idx]} (importancia={importancias[idx]:.4f})")
    df_rf = pd.DataFrame(X[:, indices_top], columns=seleccionadas)
    df_rf.to_csv(os.path.join(output_dir, "random_forest_resultados.csv"), index=False)
    return df_rf

def aplicar_rfe(X_scaled, y, nombres_variables, output_dir="C:\\Linux" ,n_vars=30):
    logging.info("Aplicando RFE.")
    modelo = LinearRegression()
    rfe = RFE(estimator=modelo, n_features_to_select=n_vars)
    rfe.fit(X_scaled, y)
    seleccionadas = nombres_variables[rfe.support_]
    logging.info(f"RFE seleccionó {n_vars} variables.")
    for idx, keep in enumerate(rfe.support_):
        if keep:
            logging.info(f"Variable conservada por RFE: {nombres_variables[idx]}")
        else:
            logging.info(f"Variable eliminada por RFE: {nombres_variables[idx]}")
    df_rfe = pd.DataFrame(X_scaled[:, rfe.support_], columns=seleccionadas)
    df_rfe.to_csv(os.path.join(output_dir, "rfe_resultados.csv"), index=False)
    return df_rfe

def main():
    ruta_csv = "C:\\Linux\\red_var_csv.csv"
    output_dir = "C:\\Linux"  # Ruta a guardar los resultados
    df = cargar_datos(ruta_csv)
    # Normaliza a z-score
    X, X_scaled, y, nombres = preparar_datos(df)

    # PCA (Análisis de Componentes Principales): varianza deseada del n%
    varianza_deseada = 0.95
    aplicar_pca(X_scaled, nombres, output_dir, varianza_deseada)

    # PLS (Partial Least Squares Regression): número de componentes
    n_comp = 10
    aplicar_pls(X_scaled, y, output_dir, n_comp)

    # LASSO (Least Absolute Shrinkage and Selection Operator)
    aplicar_lasso(X_scaled, y, nombres, output_dir)

    # Random Forest (Random Forest Feature Importance): top_n
    top_n = 30
    aplicar_random_forest(X.values, y, nombres, output_dir, top_n)

    # RFE (Recursive Feature Elimination): n_vars
    n_vars = 30
    aplicar_rfe(X_scaled, y, nombres, output_dir, n_vars)

if __name__ == "__main__":
    main()
