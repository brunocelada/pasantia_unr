#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Solución para evitar errores de hilos con matplotlib
import matplotlib.pyplot as plt
import optuna
from datetime import datetime
import warnings
import json
import re

# Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer, FunctionTransformer
from sklearn.decomposition import PCA, SparsePCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import RFE, RFECV, SelectFromModel, SequentialFeatureSelector
from sklearn.pipeline import Pipeline 

# Models
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from pygam import LinearGAM, s, l, f 
from pygam.terms import TermList 

# Gaussian Process Regressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern, RationalQuadratic, DotProduct

# LightGBM (Optional)
try:
    import lightgbm as lgb
    LGBM_INSTALLED = True
except ImportError:
    LGBM_INSTALLED = False
    print("Advertencia: LightGBM no está instalado. El modelo LGBM no estará disponible.")

# CatBoost (Optional)
try:
    from catboost import CatBoostRegressor
    CATBOOST_INSTALLED = True
except ImportError:
    CATBOOST_INSTALLED = False
    print("Advertencia: CatBoost no está instalado. El modelo CatBoost no estará disponible.")

# SHAP (Optional) for model interpretation
try:
    import shap
    SHAP_INSTALLED = True
except ImportError:
    SHAP_INSTALLED = False
    print("Advertencia: SHAP no está instalado. Los gráficos de SHAP no estarán disponibles.")


# Model selection and metrics
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, cross_val_score, GridSearchCV, learning_curve, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error, make_scorer, mean_absolute_error 
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning 
from scipy.stats import linregress 
from sklearn.inspection import permutation_importance


# Factor Analyzer for EFA
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

# Global random state for reproducibility
RANDOM_STATE = 42
# Suppress Optuna trial logging messages
optuna.logging.set_verbosity(optuna.logging.WARNING)


def rmse(y_true, y_pred):
    """Calcula la Raíz del Error Cuadrático Medio."""
    y_true_clean = np.asarray(y_true)[~np.isnan(y_pred)]
    y_pred_clean = np.asarray(y_pred)[~np.isnan(y_pred)]
    if len(y_true_clean) == 0:
        return np.nan
    return np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))

def mape(y_true, y_pred):
    """Calcula el Error Porcentual Absoluto Medio."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_nan_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[non_nan_mask], y_pred[non_nan_mask]
    
    non_zero_mask = y_true != 0
    if np.sum(non_zero_mask) == 0: 
        return np.nan if np.sum(y_pred != 0) > 0 else 0.0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

rmse_scorer = make_scorer(rmse, greater_is_better=False)
mape_scorer = make_scorer(mape, greater_is_better=False)

def sanitize_feature_names(feature_names):
    """Sanea los nombres de las características para compatibilidad con modelos como XGBoost."""
    sanitized_names = []
    temp_sanitized_names_set = set() 

    for i, name in enumerate(feature_names):
        name_str = str(name) 
        sanitized_name = re.sub(r'[^A-Za-z0-9_]+', '_', name_str) 
        sanitized_name = re.sub(r'_+', '_', sanitized_name) 
        sanitized_name = re.sub(r'^_+|_+$', '', sanitized_name) 

        if not sanitized_name: 
            sanitized_name = f"feature_{i}" 
        
        original_sanitized_name = sanitized_name
        count = 1
        while sanitized_name in temp_sanitized_names_set:
            sanitized_name = f"{original_sanitized_name}_{count}"
            count += 1
        
        sanitized_names.append(sanitized_name)
        temp_sanitized_names_set.add(sanitized_name)
        
    return sanitized_names

def load_data(file_path):
    """Carga datos desde un archivo Excel."""
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"Error: Archivo no encontrado en {file_path}")
        exit(1)
    except Exception as e:
        print(f"Error al cargar el archivo Excel {file_path}: {e}")
        exit(1)

    if df.shape[1] < 2:
        print(f"Error: Los datos en {file_path} deben tener al menos 2 columnas (identificador/características + objetivo).")
        exit(1)
        
    identifiers = df.iloc[:, 0].copy()
    X = df.iloc[:, 1:-1].copy()
    y = df.iloc[:, -1].copy()
    
    if X.ndim == 1: 
        X = X.to_frame()
        
    original_feature_names = X.columns.tolist()
    
    return X, y, identifiers, original_feature_names

def apply_scaler(X_train, X_val, scaler_type):
    """Aplica el método de escalado seleccionado."""
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    scaler = None

    if scaler_type == "zscore":
        scaler = StandardScaler()
        X_train_scaled_np = scaler.fit_transform(X_train)
        X_val_scaled_np = scaler.transform(X_val)
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
        X_train_scaled_np = scaler.fit_transform(X_train)
        X_val_scaled_np = scaler.transform(X_val)
    elif scaler_type == "decimal":
        scaler = FunctionTransformer(lambda x: x.astype(float) / 10.0) 
        X_train_scaled_np = scaler.fit_transform(X_train)
        X_val_scaled_np = scaler.transform(X_val)
    elif scaler_type == "robust":
        scaler = RobustScaler()
        X_train_scaled_np = scaler.fit_transform(X_train)
        X_val_scaled_np = scaler.transform(X_val)
    elif scaler_type == "unit": 
        scaler = Normalizer()
        X_train_scaled_np = scaler.fit_transform(X_train)
        X_val_scaled_np = scaler.transform(X_val)
    elif scaler_type == "none":
        X_train_scaled_np = X_train.values 
        X_val_scaled_np = X_val.values
    else:
        print(f"Error: Tipo de escalador desconocido '{scaler_type}'.")
        exit(1)
    
    X_train_scaled = pd.DataFrame(X_train_scaled_np, columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled_np, columns=X_val.columns, index=X_val.index)
            
    return X_train_scaled, X_val_scaled, scaler

def apply_reduction(X_train, y_train, X_val, reduction_type, original_feature_names):
    """Aplica el método de reducción de dimensionalidad seleccionado."""
    X_train_reduced_df = X_train.copy()
    X_val_reduced_df = X_val.copy()
    reducer_object = None 
    reduction_info = {"method": reduction_type, "n_features_before": X_train.shape[1]}
    new_feature_names = original_feature_names 

    y_train_ravel = np.ravel(y_train)
    dummy_regressor = LinearRegression(n_jobs=1) 

    cv_folds_for_gridsearch = min(5, X_train.shape[0])
    if cv_folds_for_gridsearch < 2 and X_train.shape[0] >= 2 : cv_folds_for_gridsearch = 2 
    elif X_train.shape[0] < 2: cv_folds_for_gridsearch = 1 

    if reduction_type == "pca":
        pipe = Pipeline([('pca', PCA(random_state=RANDOM_STATE)), ('regressor', dummy_regressor)])
        n_features, n_samples_total = X_train.shape[1], X_train.shape[0]
        if n_samples_total == 0 or n_features == 0:
             print("Advertencia: PCA no se puede aplicar con 0 muestras o 0 características. Omitiendo PCA.")
             reduction_info["status"] = "Omitido (0 muestras/características)"
        else:
            min_samples_in_cv_train_split = n_samples_total - (n_samples_total // cv_folds_for_gridsearch) if cv_folds_for_gridsearch > 1 else n_samples_total
            max_n_comp = min(n_features, min_samples_in_cv_train_split)
            if max_n_comp < 1: max_n_comp = 1 
            pca_n_components_options = list(range(1, int(max_n_comp) + 1))
            if not pca_n_components_options and max_n_comp >=1 : pca_n_components_options = [1]
            if not pca_n_components_options:
                print(f"Advertencia: PCA no puede determinar un rango válido de n_components (max_n_comp={max_n_comp}). Usando fallback.")
                reduction_info["status"] = "Omitido (rango de n_components inválido para GridSearchCV)"
            else:
                param_grid = {'pca__n_components': pca_n_components_options}
                grid_search = GridSearchCV(pipe, param_grid, cv=cv_folds_for_gridsearch, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0) 
                try:
                    grid_search.fit(X_train.values, y_train_ravel) 
                    reducer_object = grid_search.best_estimator_.named_steps['pca'] 
                    reduction_info["best_n_components"] = reducer_object.n_components_
                    reduction_info["explained_variance_ratio"] = reducer_object.explained_variance_ratio_.tolist()
                    X_train_transformed, X_val_transformed = reducer_object.transform(X_train.values), reducer_object.transform(X_val.values)
                    new_feature_names = [f"PC{i+1}" for i in range(reducer_object.n_components_)]
                    X_train_reduced_df = pd.DataFrame(X_train_transformed, columns=new_feature_names, index=X_train.index)
                    X_val_reduced_df = pd.DataFrame(X_val_transformed, columns=new_feature_names, index=X_val.index)
                except Exception as e: 
                    print(f"Advertencia: GridSearchCV de PCA falló ({e}). Usando PCA para explicar el 95% de la varianza como fallback.")
                    try:
                        reducer_object = PCA(n_components=0.95, random_state=RANDOM_STATE) 
                        X_train_transformed, X_val_transformed = reducer_object.fit_transform(X_train.values), reducer_object.transform(X_val.values)
                        new_feature_names = [f"PC{i+1}" for i in range(reducer_object.n_components_)]
                        X_train_reduced_df = pd.DataFrame(X_train_transformed, columns=new_feature_names, index=X_train.index)
                        X_val_reduced_df = pd.DataFrame(X_val_transformed, columns=new_feature_names, index=X_val.index)
                        reduction_info["best_n_components"] = reducer_object.n_components_
                        reduction_info["status"] = "Fallback al 95% de varianza explicada"
                    except Exception as e_fallback:
                        print(f"El fallback de PCA también falló: {e_fallback}. Usando características originales.")
                        new_feature_names = original_feature_names 
                        reduction_info["status"] = "PCA falló, usando características originales"
    
    elif reduction_type == "efa":
        if X_train.shape[1] < 2:
            print("Advertencia: EFA requiere al menos 2 características. Omitiendo EFA.")
            reduction_info["status"] = "Omitido (muy pocas características)"
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning); warnings.simplefilter("ignore", RuntimeWarning) 
                    chi_square_value, p_value = calculate_bartlett_sphericity(X_train.values) 
                    kmo_all, kmo_model = calculate_kmo(X_train.values) 
                reduction_info["bartlett_sphericity"] = {"chi_square": chi_square_value, "p_value": p_value}
                reduction_info["kmo"] = {"kmo_model": kmo_model}
                if p_value > 0.05 or kmo_model < 0.6: print("Advertencia: Los datos pueden no ser adecuados para EFA (p de Bartlett > 0.05 o KMO < 0.6).")
                max_factors = X_train.shape[1]
                best_n_factors = 0
                for n_factors_try in range(1, max_factors + 1):
                    fa_test = FactorAnalyzer(n_factors=n_factors_try, rotation=None, method='ml')
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", UserWarning); warnings.simplefilter("ignore", RuntimeWarning)
                            fa_test.fit(X_train.values) 
                        ev, v = fa_test.get_eigenvalues()
                        if len(ev[ev > 1]) >= n_factors_try : best_n_factors = n_factors_try
                        else: 
                            if best_n_factors == 0 and n_factors_try == 1: best_n_factors = 1
                            break 
                    except Exception:
                        if best_n_factors == 0 and n_factors_try == 1: best_n_factors = 1 
                        break
                if best_n_factors == 0 and max_factors > 0: 
                    best_n_factors = min(3, max_factors) if max_factors > 0 else 0 
                    print(f"Advertencia: Criterio de Kaiser de EFA. Usando por defecto {best_n_factors} factores.")
                reduction_info["selected_n_factors"] = best_n_factors
                if best_n_factors > 0:
                    reducer_object = FactorAnalyzer(n_factors=best_n_factors, rotation="varimax", method='ml')
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning); warnings.simplefilter("ignore", RuntimeWarning)
                        reducer_object.fit(X_train.values) 
                    X_train_transformed,X_val_transformed = reducer_object.transform(X_train.values), reducer_object.transform(X_val.values)
                    new_feature_names = [f"Factor{i+1}" for i in range(best_n_factors)]
                    X_train_reduced_df = pd.DataFrame(X_train_transformed, columns=new_feature_names, index=X_train.index)
                    X_val_reduced_df = pd.DataFrame(X_val_transformed, columns=new_feature_names, index=X_val.index)
                    reduction_info["factor_loadings"] = pd.DataFrame(reducer_object.loadings_, index=original_feature_names, columns=new_feature_names).to_dict()
                    reduction_info["factor_variance"] = reducer_object.get_factor_variance() 
                else:
                    print("Advertencia: EFA 0 factores. Usando características originales.")
                    new_feature_names = original_feature_names 
                    reduction_info["status"] = "Omitido (0 factores encontrados)"
            except Exception as e:
                print(f"Error en EFA: {e}. Usando características originales.")
                new_feature_names = original_feature_names 
                reduction_info["status"] = f"Error: {e}"

    elif reduction_type == "plsr":
        max_comps = min(X_train.shape[0], X_train.shape[1])
        if max_comps > 0:
            param_grid = {'n_components': list(range(1, max_comps + 1))}
            grid_search = GridSearchCV(PLSRegression(scale=False), param_grid, cv=cv_folds_for_gridsearch, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1) 
            try:
                grid_search.fit(X_train.values, y_train_ravel) 
                reducer_object = grid_search.best_estimator_
                reduction_info["best_n_components"] = reducer_object.n_components
                X_train_transformed, X_val_transformed = reducer_object.transform(X_train.values), reducer_object.transform(X_val.values)
                new_feature_names = [f"PLSC{i+1}" for i in range(reducer_object.n_components)]
                X_train_reduced_df = pd.DataFrame(X_train_transformed, columns=new_feature_names, index=X_train.index)
                X_val_reduced_df = pd.DataFrame(X_val_transformed, columns=new_feature_names, index=X_val.index)
                
                loadings_df = pd.DataFrame(reducer_object.x_loadings_, 
                                           index=original_feature_names, 
                                           columns=new_feature_names)
                reduction_info['plsr_loadings'] = loadings_df.to_dict()
                
            except ValueError as e:
                print(f"Advertencia: GridSearchCV de PLSR falló ({e}).")
                n_c = min(X_train.shape[0], X_train.shape[1])
                if n_c > 0:
                    reducer_object = PLSRegression(n_components=n_c, scale=False)
                    reducer_object.fit(X_train.values, y_train_ravel)
                    X_train_transformed, X_val_transformed = reducer_object.transform(X_train.values), reducer_object.transform(X_val.values)
                    new_feature_names = [f"PLSC{i+1}" for i in range(reducer_object.n_components)]
                    X_train_reduced_df = pd.DataFrame(X_train_transformed, columns=new_feature_names, index=X_train.index)
                    X_val_reduced_df = pd.DataFrame(X_val_transformed, columns=new_feature_names, index=X_val.index)
                    reduction_info["best_n_components"] = reducer_object.n_components
                    loadings_df = pd.DataFrame(reducer_object.x_loadings_, 
                                               index=original_feature_names, 
                                               columns=new_feature_names)
                    reduction_info['plsr_loadings'] = loadings_df.to_dict()

                else: 
                    new_feature_names, reduction_info["best_n_components"] = [], 0
                    X_train_reduced_df = pd.DataFrame(np.array([]).reshape(X_train.shape[0], 0), columns=new_feature_names, index=X_train.index)
                    X_val_reduced_df = pd.DataFrame(np.array([]).reshape(X_val.shape[0], 0), columns=new_feature_names, index=X_val.index)
        else: print("Advertencia: No hay suficientes características/muestras para PLSR.")

    elif reduction_type == "lasso": 
        if X_train.shape[1] > 0: 
            reducer_object = LassoCV(cv=cv_folds_for_gridsearch, random_state=RANDOM_STATE, max_iter=100000, n_jobs=-1, verbose=0) 
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                reducer_object.fit(X_train.values, y_train_ravel) 
            reduction_info["best_alpha"] = reducer_object.alpha_
            selector = SelectFromModel(reducer_object, prefit=True, threshold=1e-5) 
            X_train_transformed_np = selector.transform(X_train.values)
            X_val_transformed_np = selector.transform(X_val.values)
            selected_mask = selector.get_support()
            new_feature_names = [name for i, name in enumerate(original_feature_names) if selected_mask[i]]
            if not new_feature_names and X_train.shape[1] > 0: print("Advertencia: LASSO seleccionó 0 características.")
            X_train_reduced_df = pd.DataFrame(X_train_transformed_np, columns=new_feature_names, index=X_train.index)
            X_val_reduced_df = pd.DataFrame(X_val_transformed_np, columns=new_feature_names, index=X_val.index)
            reduction_info["n_selected_features"], reduction_info["selected_features_names"] = len(new_feature_names), new_feature_names
        else: print("Advertencia: No hay características para aplicar LASSO."); reduction_info["status"] = "Omitido (0 características de entrada)"

    elif reduction_type == "rfe": 
        if X_train.shape[1] > 0:
            estimator_for_rfe = LinearRegression(n_jobs=1) 
            cv_rfe = cv_folds_for_gridsearch
            if X_train.shape[0] < 2: print("Advertencia: No hay suficientes muestras para RFE CV."); reduction_info["status"] = "Omitido (muy pocas muestras para CV)"
            else:
                try:
                    reducer_object = RFECV(estimator_for_rfe, step=1, cv=cv_rfe, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0) 
                    reducer_object.fit(X_train.values, y_train_ravel) 
                    X_train_transformed_np, X_val_transformed_np = reducer_object.transform(X_train.values), reducer_object.transform(X_val.values)
                    reduction_info["n_selected_features"] = reducer_object.n_features_
                    selected_mask = reducer_object.get_support()
                    new_feature_names = [name for i, name in enumerate(original_feature_names) if selected_mask[i]]
                    if not new_feature_names and X_train.shape[1] > 0: print("Advertencia: RFE seleccionó 0 características.")
                    X_train_reduced_df = pd.DataFrame(X_train_transformed_np, columns=new_feature_names, index=X_train.index)
                    X_val_reduced_df = pd.DataFrame(X_val_transformed_np, columns=new_feature_names, index=X_val.index)
                    reduction_info["selected_features_names"] = new_feature_names
                except Exception as e: print(f"Error durante RFE: {e}. Usando características originales."); new_feature_names, reduction_info["status"] = original_feature_names, f"Error: {e}"
        else: print("Advertencia: No hay características para aplicar RFE."); reduction_info["status"] = "Omitido (0 características de entrada)"

    elif reduction_type == "sparsepca": 
        pipe = Pipeline([('sparsepca', SparsePCA(random_state=RANDOM_STATE, max_iter=100000)), ('regressor', dummy_regressor)])
        n_features, n_samples_total = X_train.shape[1], X_train.shape[0]
        if n_samples_total == 0 or n_features == 0: print("Advertencia: SparsePCA no se puede aplicar con 0 muestras o 0 características. Omitiendo."); reduction_info["status"] = "Omitido (0 muestras/características)"
        else:
            min_samples_in_cv_train_split = n_samples_total - (n_samples_total // cv_folds_for_gridsearch) if cv_folds_for_gridsearch > 1 else n_samples_total
            max_n_comp = min(n_features, min_samples_in_cv_train_split)
            if max_n_comp < 1: max_n_comp = 1
            spca_n_components_options = []
            num_points_n_comp = 7 
            if max_n_comp <= 1: spca_n_components_options = [1] if max_n_comp == 1 else []
            elif max_n_comp <= num_points_n_comp: spca_n_components_options = list(range(1, int(max_n_comp) + 1))
            else:
                _temp_list = np.unique(np.round(np.linspace(1, max_n_comp, num_points_n_comp)).astype(int))
                spca_n_components_options = [val for val in _temp_list if val >= 1]
                if not spca_n_components_options and max_n_comp > 0: spca_n_components_options = [1]
            if not spca_n_components_options and max_n_comp >= 1: spca_n_components_options = [1]
            if not spca_n_components_options: print(f"Advertencia: SparsePCA no puede determinar un rango válido de n_components (max_n_comp={max_n_comp}). Usando fallback."); reduction_info["status"] = "Omitido (rango de n_components inválido para GridSearchCV)"
            else:
                param_grid = {'sparsepca__n_components': spca_n_components_options, 'sparsepca__alpha': [0.1, 1, 10]}
                grid_search = GridSearchCV(pipe, param_grid, cv=cv_folds_for_gridsearch, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0) 
                try:
                    grid_search.fit(X_train.values, y_train_ravel)
                    reducer_object = grid_search.best_estimator_.named_steps['sparsepca']
                    reduction_info["best_params"] = {"n_components": reducer_object.n_components_, "alpha": reducer_object.alpha}
                    X_train_transformed, X_val_transformed = reducer_object.transform(X_train.values), reducer_object.transform(X_val.values)
                    new_feature_names = [f"SPC{i+1}" for i in range(reducer_object.n_components_)]
                    X_train_reduced_df, X_val_reduced_df = pd.DataFrame(X_train_transformed, columns=new_feature_names, index=X_train.index), pd.DataFrame(X_val_transformed, columns=new_feature_names, index=X_val.index)
                except Exception as e: 
                    print(f"Advertencia: GridSearchCV de SparsePCA falló ({e}). Usando SparsePCA con n_components por defecto y alpha=1 como fallback.")
                    n_c = min(X_train.shape[1], X_train.shape[0], 3) 
                    if n_c > 0:
                        try:
                            reducer_object = SparsePCA(n_components=n_c, alpha=1, random_state=RANDOM_STATE, max_iter=100000)
                            X_train_transformed, X_val_transformed = reducer_object.fit_transform(X_train.values), reducer_object.transform(X_val.values)
                            new_feature_names = [f"SPC{i+1}" for i in range(reducer_object.n_components_)]
                            X_train_reduced_df, X_val_reduced_df = pd.DataFrame(X_train_transformed, columns=new_feature_names, index=X_train.index), pd.DataFrame(X_val_transformed, columns=new_feature_names, index=X_val.index)
                            reduction_info["best_params"], reduction_info["status"] = {"n_components": n_c, "alpha": 1}, "Fallback a n_components/alpha por defecto"
                        except Exception as e_fallback: print(f"El fallback de SparsePCA también falló: {e_fallback}. Usando características originales."); new_feature_names, reduction_info["status"] = original_feature_names, "SparsePCA falló, usando características originales"
                    else: new_feature_names, reduction_info["status"] = original_feature_names, "SparsePCA omitido (no hay suficientes muestras/características para fallback)"
    
    elif reduction_type == "mihr": 
        print("Aplicando MIHR (Reductor Hecho en Casa)...")
        X_temp_mihr = X_train.copy() 
        if not isinstance(X_temp_mihr, pd.DataFrame): X_temp_mihr = pd.DataFrame(X_temp_mihr, columns=original_feature_names)
        collinearity_threshold, to_drop_collinear, cols_to_check = 0.9, set(), X_temp_mihr.columns.tolist()
        while True:
            if len(cols_to_check) < 2: break
            corr_matrix_current = X_temp_mihr[cols_to_check].corr().abs()
            upper = corr_matrix_current.where(np.triu(np.ones(corr_matrix_current.shape), k=1).astype(bool))
            col_to_drop_this_pass, max_corr_found, pair_to_evaluate = None, 0, None
            for i_outer in range(len(upper.columns)):
                for j_inner in range(i_outer + 1, len(upper.columns)): 
                    col1, col2 = upper.columns[i_outer], upper.columns[j_inner] 
                    current_corr = upper.loc[col1, col2]
                    if current_corr > collinearity_threshold:
                        if current_corr > max_corr_found: max_corr_found, pair_to_evaluate = current_corr, (col1, col2)
            if pair_to_evaluate:
                col1, col2 = pair_to_evaluate
                mean_corr1 = corr_matrix_current[col1].drop([col1, col2] if col2 in corr_matrix_current[col1].index else col1).abs().mean() if len(cols_to_check)>2 else 0
                mean_corr2 = corr_matrix_current[col2].drop([col2, col1] if col1 in corr_matrix_current[col2].index else col2).abs().mean() if len(cols_to_check)>2 else 0
                col_to_drop_this_pass = col1 if mean_corr1 >= mean_corr2 else col2
                if col_to_drop_this_pass in X_temp_mihr.columns: X_temp_mihr.drop(columns=[col_to_drop_this_pass], inplace=True); to_drop_collinear.add(col_to_drop_this_pass); cols_to_check = X_temp_mihr.columns.tolist() 
                else: break
            else: break 
        features_after_collinearity_removal = X_temp_mihr.columns.tolist()
        reduction_info.update({"collinear_features_dropped": list(to_drop_collinear), "collinearity_threshold_used": collinearity_threshold, "features_after_collinearity_removal": len(features_after_collinearity_removal)})
        print(f"MIHR - Después de la eliminación de colinealidad (umbral > {collinearity_threshold}): {len(features_after_collinearity_removal)} características restantes.")
        if X_temp_mihr.empty or X_temp_mihr.shape[1] == 0: print("MIHR - No quedan características. Usando características originales (si las hay)."); new_feature_names, reduction_info["status"] = original_feature_names, "Sin características después de la eliminación de colinealidad"
        else:
            sfs_cv_folds = min(3, X_temp_mihr.shape[0]) 
            if sfs_cv_folds < 2 and X_temp_mihr.shape[0] >=2 : sfs_cv_folds = 2
            elif X_temp_mihr.shape[0] < 2: sfs_cv_folds = 1 
            sfs_tolerance = 0.005 
            sfs = SequentialFeatureSelector(LinearRegression(n_jobs=1), n_features_to_select='auto', tol=sfs_tolerance, direction='forward', cv=sfs_cv_folds if sfs_cv_folds > 1 else None, scoring='neg_mean_squared_error', n_jobs=1)
            try:
                sfs.fit(X_temp_mihr.values, y_train_ravel) 
                reducer_object, selected_mask_sfs = sfs, sfs.get_support()
                new_feature_names = [name for i, name in enumerate(features_after_collinearity_removal) if selected_mask_sfs[i]]
                X_train_reduced_df, X_val_reduced_df = X_train[new_feature_names], X_val[new_feature_names]   
                reduction_info.update({"sfs_selected_features": new_feature_names, "sfs_n_features_selected": len(new_feature_names), "sfs_tolerance_used": sfs_tolerance})
            except Exception as e_sfs:
                print(f"MIHR - SFS falló: {e_sfs}. Usando características después de la eliminación de colinealidad (si las hay).")
                new_feature_names = features_after_collinearity_removal
                X_train_reduced_df = X_train[new_feature_names] if new_feature_names else pd.DataFrame(index=X_train.index)
                X_val_reduced_df = X_val[new_feature_names] if new_feature_names else pd.DataFrame(index=X_val.index)
                reduction_info["status"] = f"SFS falló: {e_sfs}"
        if not new_feature_names: print("MIHR - Resultó en 0 características. Usando fallback de características originales."); new_feature_names, X_train_reduced_df, X_val_reduced_df, reduction_info["status"] = original_feature_names, X_train.copy(), X_val.copy(), "MIHR resultó en 0 características, usando originales"

    elif reduction_type == "pfi":
        if X_train.shape[1] > 0:
            print("Aplicando selección de características por Importancia de Permutación (PFI)...")
            # Usar un modelo rápido como RandomForest para obtener las importancias
            estimator_for_pfi = RandomForestRegressor(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
            estimator_for_pfi.fit(X_train.values, y_train_ravel)

            result = permutation_importance(
                estimator_for_pfi, X_train.values, y_train_ravel,
                n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
            )
            
            importances_mean = result.importances_mean
            importances_std = result.importances_std

            # --- Nueva Lógica de Selección Estricta ---
            threshold_used = ""
            
            # Criterio 1 (Estricto): Importancia media > 2 * Desviación Estándar
            selected_mask_2_std = importances_mean > 2 * importances_std
            n_selected_2_std = np.sum(selected_mask_2_std)
            print(f"PFI: {n_selected_2_std} características cumplen el criterio estricto (> 2*std).")

            if n_selected_2_std > 0:
                print("Usando el umbral estricto (2*std) para la selección.")
                selected_mask = selected_mask_2_std
                threshold_used = "2*std"
            else:
                print("El umbral estricto (2*std) no seleccionó características. Probando umbral más permisivo (1*std).")
                # Criterio 2 (Permisivo): Importancia media > 1 * Desviación Estándar
                selected_mask_1_std = importances_mean > 1 * importances_std
                n_selected_1_std = np.sum(selected_mask_1_std)
                print(f"PFI: {n_selected_1_std} características cumplen el criterio permisivo (> 1*std).")
                
                if n_selected_1_std > 0:
                    selected_mask = selected_mask_1_std
                    threshold_used = "1*std"
                else:
                    # Fallback
                    print("Advertencia: PFI con umbral estadístico no seleccionó características. Usando fallback de las 5 mejores con importancia > 0.")
                    fallback_indices = importances_mean.argsort()[-5:]
                    positive_importance_indices = fallback_indices[importances_mean[fallback_indices] > 0]
                    
                    selected_mask = np.zeros_like(importances_mean, dtype=bool)
                    selected_mask[positive_importance_indices] = True
                    threshold_used = "Fallback (Top 5 con imp > 0)"

            new_feature_names = [name for i, name in enumerate(original_feature_names) if selected_mask[i]]
            
            # Final check in case fallback also yielded nothing
            if not new_feature_names and X_train.shape[1] > 0:
                 print("PFI no seleccionó características. Usando todas las características originales como fallback final.")
                 new_feature_names = original_feature_names
                 threshold_used += " -> Final Fallback (All features)"

            X_train_reduced_df = X_train[new_feature_names]
            X_val_reduced_df = X_val[new_feature_names]
            
            reducer_object = None 
            reduction_info["n_selected_features"] = len(new_feature_names)
            reduction_info["selected_features_names"] = new_feature_names
            reduction_info["pfi_importances"] = {name: imp for name, imp in zip(original_feature_names, result.importances_mean)}
            reduction_info["threshold_used"] = threshold_used

        else:
            print("Advertencia: No hay características para aplicar PFI.")
            reduction_info["status"] = "Omitido (0 características de entrada)"

    elif reduction_type == "none": pass 
    else: print(f"Error: Tipo de reducción desconocido '{reduction_type}'."); exit(1)

    reduction_info["n_features_after"] = X_train_reduced_df.shape[1]
    return X_train_reduced_df, X_val_reduced_df, reducer_object, reduction_info, new_feature_names

# --- Funciones de Entrenamiento de Modelos ---

class EarlyStoppingCallback:
    """
    Callback de Optuna para detener el estudio tempranamente si no hay mejora significativa.
    La mejora se mide en términos de RMSE.
    Limpia la salida de la consola, mostrando solo el mensaje final de detención si se activa.
    """
    def __init__(self, min_trials: int, patience: int, relative_improvement_threshold: float, direction: str = "minimize"):
        self.min_trials = min_trials
        self.patience = patience
        self.relative_improvement_threshold = relative_improvement_threshold
        self.direction = direction
        self._no_improvement_count = 0
        if direction == "minimize":
            self.best_value = float('inf')
        else: # pragma: no cover
            self.best_value = float('-inf')

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        current_trial_number = trial.number + 1
        current_value = trial.value

        if current_value is None: 
            if current_trial_number > self.min_trials:
                self._no_improvement_count += 1
        
        elif self.direction == "minimize": 
            is_significant_improvement = False
            if current_value < self.best_value:
                if self.best_value == float('inf'): 
                    is_significant_improvement = True 
                else:
                    denominator = abs(self.best_value) if abs(self.best_value) > 1e-9 else 1e-9
                    relative_diff = (self.best_value - current_value) / denominator
                    if relative_diff > self.relative_improvement_threshold:
                        is_significant_improvement = True
                self.best_value = current_value 
            
            if current_trial_number > self.min_trials:
                if is_significant_improvement:
                    self._no_improvement_count = 0
                else:
                    self._no_improvement_count += 1
        
        if current_trial_number > self.min_trials and self._no_improvement_count >= self.patience:
            print("\r" + " " * 120 + "\r", end="") 
            print(f"Detención temprana activada: No hubo mejora relativa del RMSE > {self.relative_improvement_threshold*100:.3f}% durante {self.patience} trials consecutivos.")
            print(f"Esto ocurrió después de {current_trial_number} trials totales.")
            study.stop()


def train_model_base(X_train_df, y_train_ravel, model_type, trial, current_feature_names, catb_train_dir_override=None):
    """Ayudante para crear una instancia de modelo basada en el tipo y el trial para Optuna."""
    model = None
    if model_type == "mlr": model = LinearRegression(n_jobs=1) 
    elif model_type == "svm":
        params = {'C': trial.suggest_float('C', 1e-3, 1e3, log=True), 'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid']), 'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']), 'epsilon': trial.suggest_float('epsilon', 1e-3, 1e1, log=True)}
        if params['kernel'] == 'poly': params['degree'] = trial.suggest_int('degree', 2, 5)
        model = SVR(**params, max_iter=100000) 
    elif model_type == "dt":
        params = {'max_depth': trial.suggest_int('max_depth', 2, 32, log=False), 'min_samples_split': trial.suggest_int('min_samples_split', 2, 20), 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20), 'criterion': trial.suggest_categorical('criterion', ['squared_error', 'friedman_mse', 'absolute_error'])}
        model = DecisionTreeRegressor(**params, random_state=RANDOM_STATE)
    elif model_type == "rf":
        params = {'n_estimators': trial.suggest_int('n_estimators', 50, 300), 'max_depth': trial.suggest_int('max_depth', 3, 32), 'min_samples_split': trial.suggest_int('min_samples_split', 2, 20), 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20), 'criterion': trial.suggest_categorical('criterion', ['squared_error', 'absolute_error', 'poisson'])}
        model = RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=1) 
    elif model_type == "xgb":
        params = {'n_estimators': trial.suggest_int('n_estimators', 100, 1000), 'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True), 'max_depth': trial.suggest_int('max_depth', 3, 15), 'min_child_weight': trial.suggest_int('min_child_weight', 1, 30), 'gamma': trial.suggest_float('gamma', 0, 20), 'subsample': trial.suggest_float('subsample', 0.5, 1.0), 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0), 'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0), 'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0), 'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 100.0, log=True), 'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 100.0, log=True)}
        model = xgb.XGBRegressor(**params, random_state=RANDOM_STATE, n_jobs=1, objective='reg:squarederror', booster='gbtree')
    elif model_type == "knn":
        max_k = X_train_df.shape[0] -1 if X_train_df.shape[0] > 1 else 1
        if max_k == 0: max_k = 1
        params = {'n_neighbors': trial.suggest_int('n_neighbors', 1, min(30, max_k)), 'weights': trial.suggest_categorical('weights', ['uniform', 'distance']), 'p': trial.suggest_int('p', 1, 2)}
        if params['n_neighbors'] == 0 : params['n_neighbors'] = 1 
        model = KNeighborsRegressor(**params, n_jobs=1) 
    elif model_type == "ann":
        n_layers = trial.suggest_int('n_layers', 1, 3)
        hidden_layer_sizes = [trial.suggest_int(f'n_units_l{i}', 16, 256, log=True) for i in range(n_layers)]
        params = {'hidden_layer_sizes': tuple(hidden_layer_sizes), 'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic']), 'solver': trial.suggest_categorical('solver', ['adam', 'sgd']), 'alpha': trial.suggest_float('alpha', 1e-6, 1e-1, log=True), 'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-5, 1e-2, log=True), 'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive'])}
        model = MLPRegressor(**params, random_state=RANDOM_STATE, max_iter=2000, early_stopping=True, n_iter_no_change=25, tol=1e-5) 
    elif model_type == "gam":
        n_splines_global = trial.suggest_int('n_splines', 5, 25)
        lam_global = trial.suggest_float('lam', 1e-3, 1e3, log=True)
        terms_generator = (s(i, n_splines=n_splines_global) for i in range(X_train_df.shape[1]))
        terms = sum(terms_generator, TermList()) 
        model = LinearGAM(terms, lam=lam_global, callbacks=[])
    elif model_type == "gpr":
        kernel_choice = trial.suggest_categorical('kernel_choice', ['RBF', 'Matern', 'RationalQuadratic'])
        base_kernel = None
        if kernel_choice == 'RBF':
            base_kernel = RBF(length_scale=trial.suggest_float('length_scale_rbf', 1e-2, 1e2, log=True))
        elif kernel_choice == 'Matern':
            base_kernel = Matern(length_scale=trial.suggest_float('length_scale_matern', 1e-2, 1e2, log=True),
                                 nu=trial.suggest_categorical('nu_matern', [0.5, 1.5, 2.5]))
        elif kernel_choice == 'RationalQuadratic': 
            base_kernel = RationalQuadratic(length_scale=trial.suggest_float('length_scale_rq', 1e-2, 1e2, log=True),
                                            alpha=trial.suggest_float('alpha_rq', 1e-2, 1e2, log=True))
        
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * base_kernel + \
                 WhiteKernel(noise_level=trial.suggest_float('noise_level_wk', 1e-5, 1e-1, log=True)) 
        
        alpha_gpr = trial.suggest_float('alpha_gpr', 1e-10, 1e-2, log=True) 
        model = GaussianProcessRegressor(kernel=kernel, alpha=alpha_gpr, 
                                         n_restarts_optimizer=trial.suggest_int('n_restarts_optimizer', 3, 10), 
                                         random_state=RANDOM_STATE)
    elif model_type == "lgbm" and LGBM_INSTALLED:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True), 
            'num_leaves': trial.suggest_int('num_leaves', 10, 150), 
            'max_depth': trial.suggest_int('max_depth', 3, 12), 
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True), 
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True), 
        }
        model = lgb.LGBMRegressor(**params, random_state=RANDOM_STATE, n_jobs=1, verbose=-1)
    elif model_type == "catb" and CATBOOST_INSTALLED:
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True), 
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True), 
            'border_count': trial.suggest_int('border_count', 32, 255) 
        }
        model = CatBoostRegressor(**params, random_state=RANDOM_STATE, verbose=0, thread_count=1, train_dir=catb_train_dir_override)
    elif model_type == "gbr":
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'loss': trial.suggest_categorical('loss', ['squared_error', 'absolute_error', 'huber', 'quantile'])
        }
        model = GradientBoostingRegressor(**params, random_state=RANDOM_STATE)
    elif model_type == "adaboost":
        base_estimator = DecisionTreeRegressor(max_depth=trial.suggest_int('max_depth_base', 3, 10))
        params = {
            'estimator': base_estimator,
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
            'loss': trial.suggest_categorical('loss', ['linear', 'square', 'exponential'])
        }
        model = AdaBoostRegressor(**params, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Error de configuración del tipo de modelo {model_type} en train_model_base.")
    return model

def train_model(X_train_df, y_train, model_type, n_trials_optuna_config, current_feature_names, output_dir_name, early_stopping_params=None): 
    """Función principal de despacho de entrenamiento con Optuna."""
    if X_train_df.shape[1] == 0 and model_type not in ["gam", "gpr"]: 
        print(f"Advertencia: No hay características disponibles para entrenar el modelo {model_type}. Omitiendo entrenamiento.")
        return None, {}, None 
    if model_type == "lgbm" and not LGBM_INSTALLED:
        print(f"LightGBM no instalado. Omitiendo entrenamiento de {model_type}.")
        return None, {}, None 
    if model_type == "catb" and not CATBOOST_INSTALLED:
        print(f"CatBoost no instalado. Omitiendo entrenamiento de {model_type}.")
        return None, {}, None 

    y_train_ravel = np.ravel(y_train)
    
    actual_n_trials_to_run = n_trials_optuna_config
    if early_stopping_params:
        actual_n_trials_to_run = 10000 
        print(f"Iniciando optimización con Optuna para {model_type} con DETECCIÓN TEMPRANA (hasta {actual_n_trials_to_run} trials).")
    else:
        print(f"Iniciando optimización con Optuna para {model_type} con {actual_n_trials_to_run} trials.")


    catboost_base_output_path = None
    if model_type == "catb":
        catboost_base_output_path = os.path.join(output_dir_name, "catboost_info")
        os.makedirs(catboost_base_output_path, exist_ok=True) 

    def progress_callback(study, trial):
        trial_num = trial.number + 1 
        max_trials_display = n_trials_optuna_config if not early_stopping_params else "auto"
        best_val_str = f"{study.best_value:.6f}" if study.best_value is not None and study.best_value != float('inf') else "N/A"
        print(f"Optimizando {model_type} - Trial: {trial_num}/{max_trials_display}. Mejor RMSE: {best_val_str}" + " " * 20, end='\r', flush=True)

    def objective(trial):
        n_samples = X_train_df.shape[0]
        cv_folds = min(5, n_samples)
        if cv_folds < 2: 
            if n_samples >= 2: cv_folds = 2
            else: return float('inf') 

        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        
        model = train_model_base(X_train_df, y_train_ravel, model_type, trial, current_feature_names, catb_train_dir_override=catboost_base_output_path)
        
        model_for_cv = model 
        if model_type == "ann":
            ann_cv_params_from_trial = {
                'hidden_layer_sizes': tuple([trial.params[f'n_units_l{i}'] for i in range(trial.params['n_layers'])]),
                'activation': trial.params['activation'],
                'solver': trial.params['solver'],
                'alpha': trial.params['alpha'],
                'learning_rate_init': trial.params['learning_rate_init'],
                'learning_rate': trial.params['learning_rate']
            }
            model_for_cv = MLPRegressor(**ann_cv_params_from_trial, 
                                        random_state=RANDOM_STATE, 
                                        max_iter=2000, 
                                        early_stopping=False 
                                        )
        
        if model_type == "gam": 
            gam_trial_model = model_for_cv 
            fold_scores = []
            if X_train_df.shape[1] == 0: 
                 try:
                    if y_train_ravel.size > 0: 
                        y_mean_pred = np.full_like(y_train_ravel, np.mean(y_train_ravel))
                        return rmse(y_train_ravel, y_mean_pred)
                    else: return float('inf') 
                 except Exception: return float('inf')

            for train_idx, val_idx in cv.split(X_train_df.values, y_train_ravel):
                X_fold_train, X_fold_val = X_train_df.values[train_idx], X_train_df.values[val_idx]
                y_fold_train, y_fold_val = y_train_ravel[train_idx], y_train_ravel[val_idx]
                
                try:
                    gam_trial_model.fit(X_fold_train, y_fold_train) 
                    preds = gam_trial_model.predict(X_fold_val)
                    fold_scores.append(rmse(y_fold_val, preds))
                except Exception: fold_scores.append(np.inf); break 
            if not fold_scores or np.all(np.isinf(fold_scores)): return float('inf')
            current_rmse_gam = np.nanmean(fold_scores)
            if np.isnan(current_rmse_gam): return float('inf') 
            return current_rmse_gam
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UndefinedMetricWarning)
                warnings.simplefilter("ignore", ConvergenceWarning) 
                score = cross_val_score(model_for_cv, X_train_df, y_train_ravel, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=1).mean() 
            if np.isnan(score): return float('inf') 
            return -score 
        except Exception as e:
            return float('inf') 

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction='minimize', sampler=sampler) 
    
    callbacks_list = [progress_callback]
    if early_stopping_params:
        es_callback = EarlyStoppingCallback(
            min_trials=early_stopping_params["min_trials"],
            patience=early_stopping_params["patience"],
            relative_improvement_threshold=early_stopping_params["relative_improvement_threshold"],
            direction="minimize" 
        )
        callbacks_list.append(es_callback)

    try:
        study.optimize(objective, n_trials=actual_n_trials_to_run, callbacks=callbacks_list, n_jobs=1, show_progress_bar=False) 
    except Exception as e:
        print(f"\nError durante la optimización con Optuna para {model_type}: {e}") 
        return None, {}, study 
    finally:
        print("\r" + " " * 120 + "\r", end="") 

    best_params = study.best_trial.params if study.best_trial is not None else {} 
    filtered_best_params = best_params.copy()
    if model_type == 'gpr': 
        kernel_choice = best_params.get('kernel_choice')
        if kernel_choice != 'RBF': filtered_best_params.pop('length_scale_rbf', None)
        if kernel_choice != 'Matern': 
            filtered_best_params.pop('length_scale_matern', None)
            filtered_best_params.pop('nu_matern', None)
        if kernel_choice != 'RationalQuadratic':
            filtered_best_params.pop('length_scale_rq', None)
            filtered_best_params.pop('alpha_rq', None)
    
    final_model = train_model_base(X_train_df, y_train_ravel, model_type, optuna.trial.FixedTrial(filtered_best_params), current_feature_names, catb_train_dir_override=catboost_base_output_path)
    
    if final_model is not None:
        try:
            final_model.fit(X_train_df, y_train_ravel)
        except Exception as e:
            print(f"Error ajustando el modelo final {model_type} con los mejores parámetros: {e}")
            final_model = None 
    
    return final_model, best_params, study

def evaluate_model_cv(model_params, X_df, y, model_type, cv_method_str, current_feature_names_for_gam):
    """Evalúa el modelo usando la validación cruzada especificada."""
    if X_df.shape[1] == 0 and model_type not in ["gam", "gpr"]:
        print(f"Advertencia: No hay características para evaluar el modelo {model_type} para {cv_method_str}. Devolviendo NaNs.")
        return np.nan, np.nan, np.nan

    y_ravel, n_samples = np.ravel(y), X_df.shape[0]
    cv_obj = None 
    model_cv = None 

    if model_type == "mlr": model_cv = LinearRegression(n_jobs=1)
    elif model_type == "svm": model_cv = SVR(**model_params, max_iter=100000) 
    elif model_type == "dt": model_cv = DecisionTreeRegressor(**model_params, random_state=RANDOM_STATE)
    elif model_type == "rf": model_cv = RandomForestRegressor(**model_params, random_state=RANDOM_STATE, n_jobs=1) 
    elif model_type == "xgb": model_cv = xgb.XGBRegressor(**model_params, random_state=RANDOM_STATE, n_jobs=1, objective='reg:squarederror') 
    elif model_type == "knn": model_cv = KNeighborsRegressor(**model_params, n_jobs=1) 
    elif model_type == "gbr": model_cv = GradientBoostingRegressor(**model_params, random_state=RANDOM_STATE)
    elif model_type == "adaboost":
        base_estimator_depth = model_params.get('max_depth_base', 5)
        base_estimator_cv = DecisionTreeRegressor(max_depth=base_estimator_depth)
        adaboost_cv_params = model_params.copy()
        adaboost_cv_params.pop('max_depth_base', None)
        adaboost_cv_params['estimator'] = base_estimator_cv
        model_cv = AdaBoostRegressor(**adaboost_cv_params, random_state=RANDOM_STATE)
    elif model_type == "ann":
        n_layers = model_params.get('n_layers',1) 
        hls_keys = [k for k in model_params if k.startswith('n_units_l')]
        hls = [model_params[k] for k in sorted(hls_keys, key=lambda x: int(x.split('l')[-1]))] if hls_keys else [64]*n_layers
        
        ann_best_params_for_cv = {
            'hidden_layer_sizes': tuple(hls),
            'activation': model_params.get('activation'),
            'solver': model_params.get('solver'),
            'alpha': model_params.get('alpha'),
            'learning_rate_init': model_params.get('learning_rate_init'),
            'learning_rate': model_params.get('learning_rate')
        }
        model_cv = MLPRegressor(**ann_best_params_for_cv, 
                                random_state=RANDOM_STATE, 
                                max_iter=2000, 
                                early_stopping=False 
                                )
    elif model_type == "gpr":
        kernel_choice_cv = model_params.get('kernel_choice', 'RBF')
        base_kernel_cv = None
        if kernel_choice_cv == 'RBF':
            base_kernel_cv = RBF(length_scale=model_params.get('length_scale_rbf', 1.0))
        elif kernel_choice_cv == 'Matern':
            base_kernel_cv = Matern(length_scale=model_params.get('length_scale_matern', 1.0), nu=model_params.get('nu_matern', 1.5))
        elif kernel_choice_cv == 'RationalQuadratic':
            base_kernel_cv = RationalQuadratic(length_scale=model_params.get('length_scale_rq', 1.0), alpha=model_params.get('alpha_rq', 1.0))
        else: base_kernel_cv = RBF(1.0) 
        kernel_cv = ConstantKernel(1.0, (1e-3, 1e3)) * base_kernel_cv + \
                    WhiteKernel(noise_level=model_params.get('noise_level_wk', 1e-5)) 
        alpha_gpr_cv = model_params.get('alpha_gpr', 1e-10)
        n_restarts_cv = model_params.get('n_restarts_optimizer',0) 
        model_cv = GaussianProcessRegressor(kernel=kernel_cv, alpha=alpha_gpr_cv, n_restarts_optimizer=n_restarts_cv, random_state=RANDOM_STATE)
    elif model_type == "lgbm" and LGBM_INSTALLED:
        model_cv = lgb.LGBMRegressor(**model_params, random_state=RANDOM_STATE, n_jobs=1, verbose=-1)
    elif model_type == "catb" and CATBOOST_INSTALLED:
        eval_catb_params = model_params.copy()
        eval_catb_params.pop('train_dir', None) 
        model_cv = CatBoostRegressor(**eval_catb_params, random_state=RANDOM_STATE, verbose=0, thread_count=1, allow_writing_files=False) 


    if model_type == "gam": 
        n_splines_cv, lam_cv = model_params.get('n_splines', 10), model_params.get('lam', 0.6)
        X_cv_data_np = X_df.values
        if cv_method_str == "loocv":
            if n_samples < 2: return np.nan, np.nan, np.nan
            cv_obj = LeaveOneOut()
        elif cv_method_str == "5-fold":
            n_splits_kfold = min(5, n_samples)
            if n_splits_kfold < 2: return np.nan, np.nan, np.nan
            cv_obj = KFold(n_splits=n_splits_kfold, shuffle=True, random_state=RANDOM_STATE)
        else: raise ValueError("cv_method_str debe ser 'loocv' o '5-fold'")

        if X_cv_data_np.shape[1] == 0 : 
            y_pred_cv = np.full_like(y_ravel, np.mean(y_ravel), dtype=float)
        else:
            y_pred_cv = np.zeros_like(y_ravel, dtype=float) 
            for train_idx, val_idx in cv_obj.split(X_cv_data_np, y_ravel):
                X_fold_train, X_fold_val = X_cv_data_np[train_idx], X_cv_data_np[val_idx]
                y_fold_train = y_ravel[train_idx]
                terms_cv_generator = (s(i, n_splines=n_splines_cv) for i in range(X_fold_train.shape[1]))
                terms_cv_fold = sum(terms_cv_generator, TermList())
                model_fold = LinearGAM(terms_cv_fold, lam=lam_cv, callbacks=[])
                try:
                    model_fold.fit(X_fold_train, y_fold_train)
                    y_pred_cv[val_idx] = model_fold.predict(X_fold_val)
                except Exception: y_pred_cv[val_idx] = np.nan 
        if np.all(np.isnan(y_pred_cv)): return np.nan, np.nan, np.nan
        with warnings.catch_warnings(): 
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            r2_cv = r2_score(y_ravel[~np.isnan(y_pred_cv)], y_pred_cv[~np.isnan(y_pred_cv)])
        return r2_cv, rmse(y_ravel, y_pred_cv), mape(y_ravel, y_pred_cv)

    if model_cv is None: 
        print(f"Error: No se pudo reconstruir el modelo {model_type} para CV o el modelo no está disponible.")
        return np.nan, np.nan, np.nan

    if cv_method_str == "loocv":
        if n_samples < 2: print("Advertencia: LOOCV requiere al menos 2 muestras. Omitiendo."); return np.nan, np.nan, np.nan
        cv_obj = LeaveOneOut()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UndefinedMetricWarning); warnings.simplefilter("ignore", ConvergenceWarning)
                y_pred_loocv = cross_val_predict(model_cv, X_df, y_ravel, cv=cv_obj, n_jobs=1) 
            return r2_score(y_ravel, y_pred_loocv), rmse(y_ravel, y_pred_loocv), mape(y_ravel, y_pred_loocv)
        except Exception as e: print(f"Error durante LOOCV para {model_type}: {e}"); return np.nan, np.nan, np.nan
    elif cv_method_str == "5-fold":
        n_splits_kfold = min(5, n_samples)
        if n_splits_kfold < 2: print(f"Advertencia: {cv_method_str} requiere al menos 2 muestras. Omitiendo."); return np.nan, np.nan, np.nan
        cv_obj = KFold(n_splits=n_splits_kfold, shuffle=True, random_state=RANDOM_STATE)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UndefinedMetricWarning); warnings.simplefilter("ignore", ConvergenceWarning)
                r2_cv = cross_val_score(model_cv, X_df, y_ravel, cv=cv_obj, scoring='r2', n_jobs=1).mean() 
                rmse_cv_scores = cross_val_score(model_cv, X_df, y_ravel, cv=cv_obj, scoring='neg_root_mean_squared_error', n_jobs=1)
                rmse_cv = -rmse_cv_scores.mean() if not np.all(np.isnan(rmse_cv_scores)) else np.nan
                mape_cv = -cross_val_score(model_cv, X_df, y_ravel, cv=cv_obj, scoring=mape_scorer, n_jobs=1).mean() 
            return r2_cv, rmse_cv, mape_cv
        except Exception as e: print(f"Error durante 5-Fold CV para {model_type}: {e}"); return np.nan, np.nan, np.nan
    else: raise ValueError("cv_method_str debe ser 'loocv' o '5-fold'")

# --- Nuevas funciones de graficación ---

def plot_y_distribution(y_series, output_path):
    """
    Grafica la distribución de la variable objetivo (Y) con cuartiles.
    """
    if y_series.empty or y_series.isnull().all():
        print("Advertencia: La serie de la variable objetivo está vacía o solo contiene NaNs. No se puede generar el gráfico de distribución.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))

    n = len(y_series.dropna())
    num_bins = int(1 + 3.322 * np.log(n)) if n > 1 else 10 # Regla de Sturges

    counts, bin_edges, _ = plt.hist(y_series.dropna(), bins=num_bins, color='cornflowerblue', edgecolor='black')

    q1, q2, q3 = y_series.quantile([0.25, 0.5, 0.75])

    plt.axvline(q1, color='red', linestyle='--', lw=1.5, label='Cuartiles (Q1, Q2, Q3)')
    plt.axvline(q2, color='red', linestyle='--', lw=1.5)
    plt.axvline(q3, color='red', linestyle='--', lw=1.5)

    y_max = counts.max()
    text_y_pos = y_max * 0.95

    # Contar puntos en cada cuartil
    count_q1 = (y_series <= q1).sum()
    count_q2 = ((y_series > q1) & (y_series <= q2)).sum()
    count_q3 = ((y_series > q2) & (y_series <= q3)).sum()
    count_q4 = (y_series > q3).sum()

    # Anotaciones
    plt.text((bin_edges[0] + q1) / 2, text_y_pos, f'Q1\n{count_q1} puntos', ha='center', va='top', fontsize=9)
    plt.text((q1 + q2) / 2, text_y_pos, f'Q2\n{count_q2} puntos', ha='center', va='top', fontsize=9)
    plt.text((q2 + q3) / 2, text_y_pos, f'Q3\n{count_q3} puntos', ha='center', va='top', fontsize=9)
    plt.text((q3 + bin_edges[-1]) / 2, text_y_pos, f'Q4\n{count_q4} puntos', ha='center', va='top', fontsize=9)

    plt.title('Distribución de la Variable Objetivo (Y)')
    plt.xlabel('Valor de Y')
    plt.ylabel('Frecuencia')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_sanity_check(model_rmse, y_true, output_path):
    """
    Grafica una comparación del RMSE del modelo contra líneas base (media y mediana).
    """
    if y_true.empty or y_true.isnull().all() or np.isnan(model_rmse):
        print("Advertencia: Datos insuficientes o RMSE inválido para generar el gráfico de Sanity Check.")
        return
        
    y_true_clean = y_true.dropna()
    if y_true_clean.empty:
        print("Advertencia: y_true no tiene valores válidos. No se puede generar el gráfico de Sanity Check.")
        return

    mean_pred = np.full_like(y_true_clean, y_true_clean.mean())
    median_pred = np.full_like(y_true_clean, y_true_clean.median())

    rmse_mean = rmse(y_true_clean, mean_pred)
    rmse_median = rmse(y_true_clean, median_pred)

    labels = ['Modelo Actual', 'Predicción Media', 'Predicción Mediana']
    values = [model_rmse, rmse_mean, rmse_median]
    colors = ['cornflowerblue', 'lightcoral', 'lightgreen']

    plt.style.use('default')
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=colors)
    
    for bar in bars:
        yval = bar.get_height()
        if not np.isnan(yval):
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center', weight='bold')

    plt.ylabel('RMSE')
    plt.title('Comparación de RMSE: Modelo vs. Líneas Base')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.box(False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_permutation_importance(model, X, y, feature_names, output_path, model_type):
    """
    Calcula y grafica la importancia de características por permutación.
    """
    if X.empty or X.shape[1] == 0:
        print("Advertencia: No hay características para calcular la importancia por permutación.")
        return
    
    X_pfi = X.values if model_type == 'gam' else X
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            result = permutation_importance(
                model, X_pfi, y, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
                scoring='neg_root_mean_squared_error'
            )
    except Exception as e:
        print(f"Error calculando la importancia por permutación: {e}")
        return

    perm_sorted_idx = result.importances_mean.argsort()
    
    top_n = min(20, len(perm_sorted_idx))
    top_indices = perm_sorted_idx[-top_n:]

    plt.style.use('seaborn-v0_8-deep')
    plt.figure(figsize=(10, 8))
    plt.barh(
        np.array(feature_names)[top_indices],
        result.importances_mean[top_indices],
    )
    plt.xlabel("PFI (Incremento en RMSE)")
    plt.title("Importancia de Características por Permutación\n(en conjunto de validación)", pad=20, weight='bold')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_shap_summary(model, X_train, output_path, model_type):
    """
    Calcula y grafica un resumen de valores SHAP.
    """
    if X_train.empty or X_train.shape[1] == 0:
        print("Advertencia: No hay características para calcular los valores SHAP.")
        return

    print("Calculando valores SHAP... Esto puede tardar un momento.")
    explainer = None
    shap_values = None

    try:
        if model_type in ['xgb', 'rf', 'dt', 'lgbm', 'catb', 'gbr', 'adaboost']:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train)
        elif model_type in ['mlr', 'plsr_as_model']: # PLSR no es un modelo final aquí, pero si lo fuera
             explainer = shap.LinearExplainer(model, X_train)
             shap_values = explainer.shap_values(X_train)
        else: # Fallback para SVM, KNN, ANN, GPR, GAM
            X_train_summary = shap.sample(X_train, min(100, X_train.shape[0]), random_state=RANDOM_STATE)
            predict_func = model.predict
            if model_type == "gam":
                 # GAM necesita entrada numpy
                 X_train_summary = X_train_summary.values
                 predict_func = lambda x: model.predict(x)

            explainer = shap.KernelExplainer(predict_func, X_train_summary)
            
            X_shap = X_train.values if model_type == 'gam' else X_train
            shap_values = explainer.shap_values(X_shap)

        plt.figure()
        shap.summary_plot(shap_values, X_train, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"No se pudo generar el gráfico de resumen de SHAP para el modelo {model_type}: {e}")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Error al generar el gráfico SHAP:\n{str(e)}", 
                ha='center', va='center', wrap=True, color='red')
        ax.set_xticks([]); ax.set_yticks([])
        plt.title("Error de SHAP"); plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

# --- Fin de nuevas funciones de graficación ---

def plot_training_learning_curve(model, model_type, X_train_df, y_train, output_path, sklearn_scoring_metric, score_display_name, best_hyperparams_for_gam=None):
    """Grafica las curvas de entrenamiento/aprendizaje usando K-Fold CV."""
    plt.figure(figsize=(10, 6))
    y_train_ravel = np.ravel(y_train)
    X_train_plot_data = X_train_df.values if model_type == "gam" else X_train_df
    
    estimator_to_use_for_lc = model 
    
    if model_type == "ann":
        ann_lc_params = {
            'hidden_layer_sizes': model.get_params()['hidden_layer_sizes'],
            'activation': model.get_params()['activation'],
            'solver': model.get_params()['solver'],
            'alpha': model.get_params()['alpha'],
            'learning_rate_init': model.get_params()['learning_rate_init'],
            'learning_rate': model.get_params()['learning_rate']
        }
        estimator_to_use_for_lc = MLPRegressor(**ann_lc_params, random_state=RANDOM_STATE, max_iter=2000, early_stopping=False)

    elif model_type == "gam": 
        n_splines_lc, lam_lc = 10, 0.6      
        if best_hyperparams_for_gam: 
            n_splines_lc = best_hyperparams_for_gam.get('n_splines', n_splines_lc)
            lam_lc = best_hyperparams_for_gam.get('lam', lam_lc)
        if X_train_plot_data.shape[1] > 0:
            terms_lc = sum((s(i, n_splines=int(n_splines_lc)) for i in range(X_train_plot_data.shape[1])), TermList())
        else: terms_lc = TermList() 
        estimator_to_use_for_lc = LinearGAM(terms_lc, lam=float(lam_lc), callbacks=[])
        

    if X_train_plot_data.shape[1] == 0 and model_type not in ["gam", "gpr"]: 
        plt.text(0.5, 0.5, "Sin características para la curva de entrenamiento.", ha='center', va='center')
        plt.title(f"Curva de Entrenamiento para {model_type} (Sin Características)")
    elif model_type == "ann" and hasattr(model, 'loss_curve_') and sklearn_scoring_metric == 'neg_mean_squared_error' and model.get_params()['early_stopping']:
        plt.plot(model.loss_curve_, label="Pérdida de Entrenamiento (ANN)")
        plt.xlabel("Épocas"); plt.ylabel("Pérdida"); plt.title(f"Curva de Pérdida de Entrenamiento para {model_type}"); plt.legend()
    elif X_train_plot_data.shape[0] >= 2 : 
        n_samples_lc = X_train_plot_data.shape[0] 
        
        k_folds_for_curve = n_samples_lc if n_samples_lc < 5 and n_samples_lc >=2 else 5
        if n_samples_lc < 2: k_folds_for_curve = 0

        if k_folds_for_curve < 2:
            plt.text(0.5, 0.5, f"No hay suficientes muestras (N={n_samples_lc}) para la curva de aprendizaje K-Fold (k={k_folds_for_curve}).", ha='center', va='center', wrap=True)
            plt.title(f"Curva de Aprendizaje para {model_type} (Error de Configuración K-Fold)")
            plt.grid(True); plt.tight_layout(); plt.savefig(output_path); plt.close()
            return

        cv_obj_for_lc = KFold(n_splits=k_folds_for_curve, shuffle=True, random_state=RANDOM_STATE)
        cv_method_label = f"{k_folds_for_curve}-Fold CV"
        
        num_train_samples_in_fold = n_samples_lc - (n_samples_lc // k_folds_for_curve)
        min_abs_train_size = k_folds_for_curve 
        max_abs_train_size = num_train_samples_in_fold
        train_sizes_abs = np.array([])

        if max_abs_train_size < min_abs_train_size:
            print("\r" + " " * 120 + "\r", end="")
            print(f"Advertencia: Para la curva {cv_method_label} con n_samples={n_samples_lc}, el tamaño máximo de entrenamiento es {max_abs_train_size}, el mínimo requerido es {min_abs_train_size}. No se generaron puntos.")
        else:
            num_points = min(5, max_abs_train_size - min_abs_train_size + 1 if max_abs_train_size >= min_abs_train_size else 1)
            if num_points <=0: num_points = 1 
            if max_abs_train_size == min_abs_train_size : train_sizes_abs = np.array([min_abs_train_size])
            else: train_sizes_abs = np.unique(np.linspace(min_abs_train_size, max_abs_train_size, num_points).astype(int))
            if len(train_sizes_abs) > 0: train_sizes_abs = train_sizes_abs[train_sizes_abs >= min_abs_train_size]
        
        if len(train_sizes_abs) == 0:
             plt.text(0.5, 0.5, f"No hay suficientes muestras o tamaños de entrenamiento adecuados para la curva de aprendizaje {cv_method_label} (N={n_samples_lc}).", ha='center', va='center', wrap=True)
             plt.title(f"Curva de Aprendizaje para {model_type} (Error de Configuración {cv_method_label})")
        else:
            print(f"Generando curva de aprendizaje {cv_method_label} para {model_type} (Métrica: {score_display_name}) con train_sizes: {train_sizes_abs}")
            try:
                with warnings.catch_warnings():
                    if model_type == "knn": 
                        warnings.simplefilter('ignore', UserWarning)
                    warnings.simplefilter("ignore", UndefinedMetricWarning)
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    
                    actual_train_sizes, train_scores, test_scores = learning_curve(
                        estimator_to_use_for_lc, X_train_plot_data, y_train_ravel, 
                        cv=cv_obj_for_lc, n_jobs=1, train_sizes=train_sizes_abs, 
                        scoring=sklearn_scoring_metric, random_state=RANDOM_STATE
                    )
                
                if sklearn_scoring_metric == 'neg_mean_squared_error':
                    train_scores_mean = np.sqrt(np.abs(np.mean(train_scores, axis=1)))
                    test_scores_mean = np.sqrt(np.abs(np.mean(test_scores, axis=1)))
                elif sklearn_scoring_metric == 'neg_root_mean_squared_error':
                    train_scores_mean = np.abs(np.mean(train_scores, axis=1))
                    test_scores_mean = np.abs(np.mean(test_scores, axis=1))
                else:
                    train_scores_mean = np.mean(train_scores, axis=1)
                    test_scores_mean = np.mean(test_scores, axis=1)
                
                plt.plot(actual_train_sizes, train_scores_mean, 'o-', color="r", label=f"Puntuación de Entrenamiento ({score_display_name})")
                if not np.all(np.isnan(test_scores_mean)): plt.plot(actual_train_sizes, test_scores_mean, 'o-', color="g", label=f"Puntuación de Validación Cruzada ({cv_method_label}, {score_display_name})")
                else: 
                    print("\r" + " " * 120 + "\r", end="")
                    print(f"ADVERTENCIA: Todas las puntuaciones de CV para {score_display_name} con {cv_method_label} son NaN. La línea de puntuación de CV no se trazó.")

                if sklearn_scoring_metric == 'r2':
                    all_r2_values = []
                    if not np.all(np.isnan(train_scores_mean)): all_r2_values.extend(train_scores_mean[~np.isnan(train_scores_mean)])
                    if not np.all(np.isnan(test_scores_mean)): all_r2_values.extend(test_scores_mean[~np.isnan(test_scores_mean)])
                    if all_r2_values: 
                        min_val_plot, max_val_plot = min(all_r2_values), max(all_r2_values)
                        padding = (max_val_plot - min_val_plot) * 0.1 if (max_val_plot - min_val_plot) > 0 else 0.1
                        y_min_limit = min(min_val_plot - padding, 0 if min_val_plot > -padding else min_val_plot - padding) 
                        plt.ylim(y_min_limit, max(1.05, max_val_plot + padding))
                    else: plt.ylim(-1.05, 1.05) 
                
                plt.xlabel("Ejemplos de Entrenamiento"); plt.ylabel(score_display_name); plt.title(f"Curva de Aprendizaje para {model_type} (CV: {cv_method_label})"); plt.legend(loc="best")
            
            except ValueError as ve: 
                if model_type == "knn" and "Expected n_neighbors <= n_samples_fit" in str(ve):
                    print("\r" + " " * 120 + "\r", end="") 
                    print(f"ADVERTENCIA: Curva de aprendizaje para KNN no se pudo generar ({score_display_name}); n_neighbors es muy grande para algunos tamaños de muestra de la curva.")
                    plt.text(0.5, 0.5, f"Curva de aprendizaje para {model_type} ({score_display_name}) no generada:\n'n_neighbors' del modelo es demasiado grande\npara algunos tamaños de muestra usados en la curva.",
                             ha='center', va='center', wrap=True, color='orange', fontsize=9)
                    plt.title(f"Curva de Aprendizaje para {model_type} (Error de Configuración)")
                else:
                    plt.text(0.5, 0.5, f"No se pudo generar la curva de aprendizaje {cv_method_label} ({score_display_name}):\n{ve}", ha='center', va='center', wrap=True)
                    plt.title(f"Curva de Aprendizaje para {model_type} (Error {cv_method_label})")
            except Exception as e: 
                plt.text(0.5, 0.5, f"No se pudo generar la curva de aprendizaje {cv_method_label} ({score_display_name}):\n{e}", ha='center', va='center', wrap=True)
                plt.title(f"Curva de Aprendizaje para {model_type} (Error {cv_method_label})")

    else: 
        plt.text(0.5, 0.5, "No hay suficientes muestras para la curva de aprendizaje (requiere al menos 2).", ha='center', va='center')
        plt.title(f"Curva de Entrenamiento/Aprendizaje para {model_type}")
    
    plt.grid(True); plt.tight_layout(); plt.savefig(output_path); plt.close()

def plot_predictions_scatter(y_train_true, y_train_pred, y_val_true, y_val_pred, output_path):
    """Grafica los valores observados vs. predichos para los conjuntos de entrenamiento y validación."""
    plt.figure(figsize=(8, 8))
    y_train_true_np, y_train_pred_np = np.ravel(y_train_true), np.ravel(y_train_pred)
    y_val_true_np, y_val_pred_np = np.ravel(y_val_true), np.ravel(y_val_pred)
    plt.scatter(y_train_true_np, y_train_pred_np, c='gray', marker='o', alpha=0.6, s=50, label="Entrenamiento: Observado vs. Predicho")
    plt.scatter(y_val_true_np, y_val_pred_np, c='blue', marker='x', s=70, label="Validación: Observado vs. Predicho")
    
    all_true_valid = y_train_true_np[~np.isnan(y_train_true_np) & ~np.isnan(y_train_pred_np)]
    all_pred_valid_train = y_train_pred_np[~np.isnan(y_train_true_np) & ~np.isnan(y_train_pred_np)]
    all_val_true_valid = y_val_true_np[~np.isnan(y_val_true_np) & ~np.isnan(y_val_pred_np)]
    all_pred_valid_val = y_val_pred_np[~np.isnan(y_val_true_np) & ~np.isnan(y_val_pred_np)]

    all_values_for_plot = np.concatenate((all_true_valid, all_pred_valid_train, all_val_true_valid, all_pred_valid_val))

    min_val, max_val = (0,1)
    if len(all_values_for_plot) > 0:
        min_val_data, max_val_data = np.min(all_values_for_plot), np.max(all_values_for_plot)
        padding = (max_val_data - min_val_data) * 0.05 if (max_val_data - min_val_data) > 0 else 0.1
        min_val, max_val = min_val_data - padding, max_val_data + padding
    
    if min_val == max_val: min_val -= 0.5; max_val += 0.5
    
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Ajuste Ideal (y=x)") 
    plt.xlabel("Valores Observados"); plt.ylabel("Valores Predichos"); plt.title("Observado vs. Predicho (Entrenamiento y Validación)")
    plt.legend(); plt.grid(True); plt.xlim(min_val, max_val); plt.ylim(min_val, max_val)
    plt.tight_layout(); plt.savefig(output_path); plt.close()

def main():
    team_logo = "aquí irá el logo cuando esté terminado"
    
    parser = argparse.ArgumentParser(description="Script Automatizado de Predicción con Aprendizaje Automático")
    parser.add_argument("--training", required=True, help="Ruta al conjunto de entrenamiento (.xlsx)")
    parser.add_argument("--validation", required=True, help="Ruta al conjunto de validación (.xlsx)")
    parser.add_argument("--scaler", required=True, choices=["none", "zscore", "minmax", "decimal", "robust", "unit"], help="Tipo de escalado")
    parser.add_argument("--reduction", required=True, choices=["none", "pca", "efa", "plsr", "lasso", "rfe", "sparsepca", "mihr", "pfi"], help="Tipo de reducción/selección de dimensionalidad")
    model_choices = ["mlr", "svm", "dt", "rf", "xgb", "knn", "ann", "gam", "gpr", "gbr", "adaboost"]
    if LGBM_INSTALLED: model_choices.append("lgbm")
    if CATBOOST_INSTALLED: model_choices.append("catb")
    parser.add_argument("--model", required=True, choices=model_choices, help="Modelo a entrenar")
    parser.add_argument("--trials", required=True, type=str,
                        help="Número de trials de Optuna o 'auto' para detención temprana.")
    parser.add_argument("--shap", default='off', choices=['on', 'off'], help="Activa la generación de gráficos SHAP ('on' para activar). Por defecto: 'off'.")
    parser.add_argument("--no_intro", action="store_true", help="Salta la introducción para ejecución automatizada.") 
    
    args = parser.parse_args()

    n_trials_optuna_config = 0
    early_stopping_settings = None

    if args.trials.lower() == "auto":
        print("\nConfigurando detención temprana automática para Optuna...")
        while True:
            try:
                min_trials_str = input("  min_trials = ")
                args.min_trials_auto = int(min_trials_str)
                if args.min_trials_auto <= 0:
                    print("    min_trials debe ser un entero positivo.")
                    continue
                break
            except ValueError:
                print("    Entrada inválida. Por favor, ingrese un número entero para min_trials.")
        
        while True:
            try:
                patience_str = input("  patience = ")
                args.patience_auto = int(patience_str)
                if args.patience_auto <= 0:
                    print("    patience debe ser un entero positivo.")
                    continue
                break
            except ValueError:
                print("    Entrada inválida. Por favor, ingrese un número entero para patience.")

        while True:
            try:
                relative_improvement_str = input("  relative_improvement (%) = ")
                args.relative_improvement_auto = float(relative_improvement_str)
                if args.relative_improvement_auto < 0: 
                    print("    relative_improvement no puede ser negativo (0 es aceptable para ninguna mejora).")
                    continue
                break
            except ValueError:
                print("    Entrada inválida. Por favor, ingrese un número para relative_improvement (ej: 0.1 para 0.1%).")

        print("\nINFO: La optimización automática de trials se basará en la mejora relativa del RMSE.")
        print(f"Se ejecutarán al menos {args.min_trials_auto} trials.")
        print(f"La optimización se detendrá si no hay una mejora relativa del RMSE de al menos {args.relative_improvement_auto}% durante {args.patience_auto} trials consecutivos.\n")
        
        early_stopping_settings = {
            "min_trials": args.min_trials_auto,
            "patience": args.patience_auto,
            "relative_improvement_threshold": args.relative_improvement_auto / 100.0 
        }
        n_trials_optuna_config = 10000 
    else:
        try:
            n_trials_optuna_config = int(args.trials)
            if n_trials_optuna_config <= 0:
                print(f"Error: El número de trials debe ser un entero positivo, o 'auto'. Recibido: {args.trials}")
                exit(1)
        except ValueError:
            print(f"Error: --trials debe ser un número entero o la palabra 'auto'. Recibido: {args.trials}")
            exit(1)


    if not args.no_intro:
        print(team_logo)
        print("\nPresiona enter para unirte al lado oscuro......") 
        input() 
        print("-" * 50) 

    print(f"Iniciando ejecución del script a las {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.trials.lower() == "auto":
        print(f"Parámetros: Escalador={args.scaler}, Reducción={args.reduction}, Modelo={args.model}, Trials=auto (min={args.min_trials_auto}, patience={args.patience_auto}, rel_imp={args.relative_improvement_auto}%)")
    else:
        print(f"Parámetros: Escalador={args.scaler}, Reducción={args.reduction}, Modelo={args.model}, Trials={n_trials_optuna_config}")
    
    output_dir_name = os.path.abspath(f"{args.scaler}_{args.reduction}_{args.model}") 
    os.makedirs(output_dir_name, exist_ok=True)
    print(f"Los resultados se guardarán en: {output_dir_name}")

    print("Cargando datos...")
    X_train_orig_df, y_train_series, ids_train_series, original_feature_names = load_data(args.training)
    X_val_orig_df, y_val_series, ids_val_series, val_original_feature_names = load_data(args.validation) 

    # --- INICIO DE NUEVOS GRÁFICOS ---
    # Gráfico de distribución de Y
    y_dist_path = os.path.join(output_dir_name, "y_distribution.png")
    plot_y_distribution(y_train_series, y_dist_path)
    print(f"Gráfico de distribución de Y guardado en {y_dist_path}")
    # --- FIN DE NUEVOS GRÁFICOS ---
    
    y_train_mean = y_train_series.mean()
    y_train_std = y_train_series.std()

    if not X_train_orig_df.columns.equals(X_val_orig_df.columns):
        print("Advertencia: Los conjuntos de entrenamiento y validación tienen nombres/orden de características diferentes. Alineando validación con entrenamiento.")
        try: X_val_orig_df = X_val_orig_df[original_feature_names]
        except KeyError as e:
            missing_cols = list(set(original_feature_names) - set(X_val_orig_df.columns))
            print(f"Error: Al conjunto de validación le faltan columnas: {missing_cols}. Detalles: {e}"); exit(1)

    print(f"Aplicando escalador: {args.scaler}...")
    X_train_scaled_df, X_val_scaled_df, scaler_obj = apply_scaler(X_train_orig_df, X_val_orig_df, args.scaler)

    df_train_std = X_train_scaled_df.copy()
    id_col_name_train = ids_train_series.name if ids_train_series.name is not None else 'Identifier'
    target_col_name_train = y_train_series.name if y_train_series.name is not None else 'Target'
    df_train_std.insert(0, id_col_name_train, ids_train_series.values) 
    df_train_std[target_col_name_train] = y_train_series.values
    train_std_path = os.path.join(output_dir_name, "training_set_standardized.xlsx")
    df_train_std.to_excel(train_std_path, index=False)
    print(f"Datos de entrenamiento estandarizados guardados en {train_std_path}")

    df_val_std = X_val_scaled_df.copy()
    id_col_name_val = ids_val_series.name if ids_val_series.name is not None else 'Identifier' 
    target_col_name_val = y_val_series.name if y_val_series.name is not None else 'Target'
    df_val_std.insert(0, id_col_name_val, ids_val_series.values) 
    df_val_std[target_col_name_val] = y_val_series.values
    val_std_path = os.path.join(output_dir_name, "validation_set_standardized.xlsx")
    df_val_std.to_excel(val_std_path, index=False)
    print(f"Datos de validación estandarizados guardados en {val_std_path}")


    print(f"Aplicando reducción de dimensionalidad: {args.reduction}...")
    X_train_reduced_df, X_val_reduced_df, reducer_obj, reduction_info, final_feature_names_raw = apply_reduction(X_train_scaled_df, y_train_series, X_val_scaled_df, args.reduction, X_train_scaled_df.columns.tolist())
    
    final_feature_names = sanitize_feature_names(final_feature_names_raw)
    if X_train_reduced_df.shape[1] == len(final_feature_names): 
        X_train_reduced_df.columns = final_feature_names
        if X_val_reduced_df.shape[1] == len(final_feature_names): X_val_reduced_df.columns = final_feature_names
        elif X_val_reduced_df.shape[1] > 0 : 
             print(f"Advertencia: El conjunto de validación tiene {X_val_reduced_df.shape[1]} características, pero {len(final_feature_names)} nombres saneados. Usando nombres genéricos para validación.")
             X_val_reduced_df.columns = [f"feature_val_{j}" for j in range(X_val_reduced_df.shape[1])]
    elif X_train_reduced_df.shape[1] > 0 : 
        print(f"Advertencia: El número de características finales ({X_train_reduced_df.shape[1]}) != nombres saneados ({len(final_feature_names)}). Usando nombres genéricos.")
        X_train_reduced_df.columns = [f"feature_train_{j}" for j in range(X_train_reduced_df.shape[1])]
        if X_val_reduced_df.shape[1] == X_train_reduced_df.shape[1]: X_val_reduced_df.columns = X_train_reduced_df.columns
        elif X_val_reduced_df.shape[1] > 0: X_val_reduced_df.columns = [f"feature_val_{j}" for j in range(X_val_reduced_df.shape[1])]
        final_feature_names = X_train_reduced_df.columns.tolist() 
    
    print(f"La reducción resultó en {X_train_reduced_df.shape[1]} características. Nombres finales usados: {X_train_reduced_df.columns.tolist()}")

    df_train_used = X_train_reduced_df.copy()
    df_train_used.insert(0, id_col_name_train, ids_train_series.values)
    df_train_used[target_col_name_train] = y_train_series.values
    train_used_path = os.path.join(output_dir_name, "training_used.xlsx")
    df_train_used.to_excel(train_used_path, index=False)
    print(f"Datos usados para entrenamiento guardados en {train_used_path}")

    df_val_used = X_val_reduced_df.copy()
    df_val_used.insert(0, id_col_name_val, ids_val_series.values) 
    df_val_used[target_col_name_val] = y_val_series.values
    val_used_path = os.path.join(output_dir_name, "validation_used.xlsx")
    df_val_used.to_excel(val_used_path, index=False)
    print(f"Datos usados para validación guardados en {val_used_path}")

    print(f"Entrenando modelo: {args.model}...") 
    trained_model, best_hyperparams, optuna_study = train_model(
        X_train_reduced_df, y_train_series, args.model, 
        n_trials_optuna_config,
        final_feature_names, output_dir_name, 
        early_stopping_settings
    )

    if trained_model is None:
        print(f"El entrenamiento del modelo falló para {args.model}. Saliendo.")
        stats_path = os.path.join(output_dir_name, "Statistics.txt")
        with open(stats_path, "w", encoding="utf-8") as f:
            f.write(f"Ejecución del script: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Parámetros: Escalador={args.scaler}, Reducción={args.reduction}, Modelo={args.model}, Trials={args.trials}\n\n")
            f.write("FALLÓ EL ENTRENAMIENTO DEL MODELO.\nRazón: Sin características o error de ajuste de Optuna/modelo.\n")
        exit(1)

    print("Evaluando modelo con Validación Cruzada...")
    r2_loocv, rmse_loocv, mape_loocv = evaluate_model_cv(best_hyperparams, X_train_reduced_df, y_train_series, args.model, "loocv", final_feature_names)
    r2_kfold, rmse_kfold, mape_kfold = evaluate_model_cv(best_hyperparams, X_train_reduced_df, y_train_series, args.model, "5-fold", final_feature_names)

    X_train_pred_input = X_train_reduced_df.values if args.model == "gam" else X_train_reduced_df
    y_train_pred_final = trained_model.predict(X_train_pred_input)

    print("Realizando predicciones en el conjunto de validación...")
    X_val_pred_input = X_val_reduced_df.values if args.model == "gam" else X_val_reduced_df
    y_val_pred = None 
    if X_val_pred_input.shape[1] == 0 and X_train_reduced_df.shape[1] > 0 :
        print("Advertencia: El conjunto de validación tiene 0 características. Prediciendo con la media de entrenamiento.")
        y_val_pred = np.full(y_val_series.shape, y_train_series.mean())
    elif hasattr(trained_model, 'n_features_in_') and X_val_pred_input.shape[1] != trained_model.n_features_in_ and args.model not in ["gam", "gpr"]: 
        print(f"Advertencia: Desajuste en el recuento de características para validación. El modelo espera {trained_model.n_features_in_}, obtuvo {X_val_pred_input.shape[1]}. Prediciendo con la media de entrenamiento.")
        y_val_pred = np.full(y_val_series.shape, y_train_series.mean())
    elif args.model == "gam" and hasattr(trained_model, 'terms') and trained_model.terms: 
        try: y_val_pred = trained_model.predict(X_val_pred_input)
        except Exception as e: print(f"Error prediciendo en el conjunto de validación para GAM: {e}. Usando la media de entrenamiento."); y_val_pred = np.full(y_val_series.shape, y_train_series.mean())
    elif args.model == "gam" and hasattr(trained_model, 'terms') and not trained_model.terms: 
        print("Advertencia: El modelo GAM no tiene términos. Prediciendo con la media/intercepto de entrenamiento.")
        try: y_val_pred = trained_model.predict(X_val_pred_input) 
        except Exception as e: print(f"Error prediciendo en validación para GAM solo con intercepto: {e}. Usando la media de entrenamiento."); y_val_pred = np.full(y_val_series.shape, y_train_series.mean())
    else: 
        try: y_val_pred = trained_model.predict(X_val_pred_input)
        except Exception as e: print(f"Error prediciendo en el conjunto de validación: {e}. Usando la media de entrenamiento."); y_val_pred = np.full(y_val_series.shape, y_train_series.mean())

    val_r2 = r2_score(y_val_series, y_val_pred) if len(y_val_series) > 0 and not np.all(np.isnan(y_val_pred)) else np.nan
    val_mae = mean_absolute_error(y_val_series, y_val_pred) if len(y_val_series) > 0 and not np.all(np.isnan(y_val_pred)) else np.nan
    val_rmse = rmse(y_val_series, y_val_pred) if len(y_val_series) > 0 and not np.all(np.isnan(y_val_pred)) else np.nan
    val_mape = mape(y_val_series, y_val_pred) if len(y_val_series) > 0 and not np.all(np.isnan(y_val_pred)) else np.nan
    
    val_obs_vs_pred_r2 = np.nan
    if len(y_val_series) > 1 and len(y_val_pred) == len(y_val_series) and not np.all(np.isnan(y_val_series)) and not np.all(np.isnan(y_val_pred)):
        mask = ~np.isnan(y_val_series) & ~np.isnan(y_val_pred)
        if np.sum(mask) > 1:
            if np.var(y_val_series[mask]) > 1e-9 and np.var(y_val_pred[mask]) > 1e-9:
                slope, intercept, r_value, p_value, std_err = linregress(y_val_series[mask], y_val_pred[mask])
                val_obs_vs_pred_r2 = r_value**2
            else:
                print("Advertencia: No se pudo calcular R2 (Obs vs Pred) por varianza cero en los datos de validación.")
        else:
            print("Advertencia: No hay suficientes puntos no-NaN para calcular el R2 (Observado vs. Predicho) en el conjunto de validación.")
    else:
        print("Advertencia: No se pudo calcular el R2 (Observado vs. Predicho) en el conjunto de validación debido a datos insuficientes o NaNs.")

    print("Generando gráficos de análisis del modelo...")

    # Gráfico de Sanity Check
    sanity_check_path = os.path.join(output_dir_name, "sanity_check.png")
    plot_sanity_check(val_rmse, y_val_series, sanity_check_path)
    print(f"Gráfico de Sanity Check guardado en {sanity_check_path}")
    
    # Importancia de Características por Permutación
    if X_val_reduced_df.shape[1] > 0:
        pfi_path = os.path.join(output_dir_name, "permutation_feature_importance.png")
        plot_permutation_importance(trained_model, X_val_reduced_df, y_val_series, final_feature_names, pfi_path, model_type=args.model)
        print(f"Gráfico de Importancia por Permutación guardado en {pfi_path}")
    else:
        print("Omitiendo gráfico de Importancia por Permutación (0 características).")

    # Gráfico de Resumen de SHAP
    if args.shap == 'on':
        if SHAP_INSTALLED and X_train_reduced_df.shape[1] > 0:
            shap_summary_path = os.path.join(output_dir_name, "shap_summary.png")
            plot_shap_summary(trained_model, X_train_reduced_df, shap_summary_path, model_type=args.model)
            print(f"Gráfico de Resumen SHAP guardado en {shap_summary_path}")
        elif not SHAP_INSTALLED:
            print("Omitiendo gráfico SHAP (biblioteca 'shap' no instalada).")
        else: # No features
            print("Omitiendo gráfico SHAP (0 características).")
    else:
        print("Generación de gráficos SHAP omitida por el usuario (--shap off).")

    print("Guardando resultados...")
    results_df_val = pd.DataFrame({
        id_col_name_val : ids_val_series.values, 
        target_col_name_val: y_val_series.values, 
        'Predicted_Target': np.ravel(y_val_pred)
    })
    results_val_path = os.path.join(output_dir_name, f"results-{args.scaler}-{args.reduction}-{args.model}.xlsx")
    results_df_val.to_excel(results_val_path, index=False)
    print(f"Resultados de predicción de validación guardados en {results_val_path}")

    results_df_train = pd.DataFrame({
        id_col_name_train : ids_train_series.values,
        target_col_name_train: y_train_series.values,
        'Predicted_Target_Train': np.ravel(y_train_pred_final)
    })
    results_train_path = os.path.join(output_dir_name, f"results-training-{args.scaler}-{args.reduction}-{args.model}.xlsx")
    results_df_train.to_excel(results_train_path, index=False)
    print(f"Resultados de predicción de entrenamiento guardados en {results_train_path}")


    training_curve_rmse_path = os.path.join(output_dir_name, "training_curve_rmse.png") 
    plot_training_learning_curve(trained_model, args.model, X_train_reduced_df, y_train_series, training_curve_rmse_path, sklearn_scoring_metric='neg_root_mean_squared_error', score_display_name='RMSE', best_hyperparams_for_gam=best_hyperparams if args.model == "gam" else None)
    print(f"Curva de entrenamiento RMSE (K-Fold CV) guardada en {training_curve_rmse_path}")

    training_curve_r2_path = os.path.join(output_dir_name, "training_curve_r2.png") 
    plot_training_learning_curve(trained_model, args.model, X_train_reduced_df, y_train_series, training_curve_r2_path, sklearn_scoring_metric='r2', score_display_name='R2 Score', best_hyperparams_for_gam=best_hyperparams if args.model == "gam" else None)
    print(f"Curva de entrenamiento R2 (K-Fold CV) guardada en {training_curve_r2_path}")

    prediction_plot_path = os.path.join(output_dir_name, "prediction.png")
    plot_predictions_scatter(y_train_series, y_train_pred_final, y_val_series, y_val_pred, prediction_plot_path)
    print(f"Gráfico de predicción guardado en {prediction_plot_path}")

    stats_path = os.path.join(output_dir_name, "Statistics.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"Resumen de Ejecución del Script\n--------------------------\n")
        f.write(f"Marca de tiempo: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Datos de Entrenamiento: {args.training}\nDatos de Validación: {args.validation}\n")
        f.write(f"Escalador: {args.scaler}\n")
        try: reduction_info_str = json.dumps(reduction_info, indent=2, default=lambda o: str(o))
        except TypeError: reduction_info_str = str(reduction_info) 
        f.write(f"Reducción: {args.reduction} (Detalles: {reduction_info_str})\n")
        
        if args.reduction == 'plsr' and 'plsr_loadings' in reduction_info:
            f.write("\nComposición de los Componentes PLS (Loadings):\n")
            loadings_dict = reduction_info['plsr_loadings']
            loadings_df = pd.DataFrame(loadings_dict)
            f.write(loadings_df.to_string(float_format="%.4f"))
            f.write("\n\n")

        f.write(f"Modelo: {args.model}\n")
        
        f.write("\nEstadísticas Descriptivas del Target de Entrenamiento (y_train):\n")
        f.write(f"  Media de y_train: {y_train_mean:.4f}\n")
        f.write(f"  Desviación Estándar de y_train: {y_train_std:.4f}\n\n")

        if args.trials.lower() == 'auto':
            f.write(f"Trials de Optuna: 'auto' (Detención Temprana Activada)\n")
            f.write(f"  Configuración de Detención Temprana:\n")
            f.write(f"    min_trials: {args.min_trials_auto}\n")
            f.write(f"    patience: {args.patience_auto}\n")
            f.write(f"    relative_improvement_threshold: {args.relative_improvement_auto:.3f}%\n")
            if optuna_study:
                 f.write(f"  Trials efectivos ejecutados: {len(optuna_study.trials)}\n")
        else:
            f.write(f"Trials de Optuna: {args.trials}\n")
            if optuna_study:
                 f.write(f"  Trials efectivos ejecutados: {len(optuna_study.trials)}\n")

        if optuna_study and optuna_study.best_trial: 
            f.write(f"Mejor número de trial: {optuna_study.best_trial.number + 1}\n")
            best_value_display = optuna_study.best_value if optuna_study.best_value is not None else np.nan
            f.write(f"Mejor RMSE (valor objetivo): {best_value_display:.6f}\n\n")
        else:
            f.write("Mejor trial: No disponible\n\n")
            
        if args.model == "mlr":
            f.write("Ecuación del Modelo de Regresión Lineal Múltiple (MLR):\n")
            if hasattr(trained_model, 'intercept_') and hasattr(trained_model, 'coef_'):
                intercept_val = trained_model.intercept_
                coef_vals = trained_model.coef_
                feature_names_for_eq = X_train_reduced_df.columns.tolist() 

                equation_str = f"  Y = {intercept_val:.4f}"
                if len(coef_vals) == len(feature_names_for_eq) and len(coef_vals) > 0 :
                    for coef, name in zip(coef_vals, feature_names_for_eq):
                        if coef >= 0:
                            equation_str += f" + {coef:.4f} * {name}"
                        else:
                            equation_str += f" - {abs(coef):.4f} * {name}"
                elif len(coef_vals) == 0 and X_train_reduced_df.shape[1] == 0: 
                     equation_str += " (Modelo solo con intercepto, sin características)"
                else:
                    equation_str += " (No se pudieron determinar los coeficientes para las características)"
                f.write(equation_str + "\n\n")
            else:
                f.write("  No se pudo extraer la ecuación del modelo MLR.\n\n")
        else:
            f.write(f"Mejores Hiperparámetros para {args.model}:\n")
            if best_hyperparams:
                 for k, v_param in best_hyperparams.items(): f.write(f"  {k}: {v_param}\n")
            else:
                 f.write("  N/A (El modelo no tiene hiperparámetros optimizables por este script o la optimización falló).\n")
            f.write("\n")


        f.write("\nMétricas de Validación Cruzada (sobre Datos de Entrenamiento):\n")
        f.write(f"  LOOCV (Evaluación Global):\n    R2: {r2_loocv:.4f}\n    RMSE: {rmse_loocv:.4f}\n    MAPE: {mape_loocv:.4f}%\n")
        f.write(f"  5-Fold CV (Evaluación Global):\n    R2: {r2_kfold:.4f}\n    RMSE: {rmse_kfold:.4f}\n    MAPE: {mape_kfold:.4f}%\n\n")
        f.write(f"Nota: Las curvas de aprendizaje de RMSE y R2 usan K-Fold CV (k determinado dinámicamente, hasta 5) internamente para estabilidad.\n\n")
        f.write("Rendimiento del Conjunto de Validación:\n")
        f.write(f"  R2 (del modelo): {val_r2:.4f}\n") 
        f.write(f"  R2 (Observado vs. Predicho): {val_obs_vs_pred_r2:.4f}\n")
        f.write(f"    MAE: {val_mae:.4f}\n") 
        f.write(f"    RMSE: {val_rmse:.4f}\n") 
        f.write(f"    MAPE: {val_mape:.4f}%\n") 
    print(f"Estadísticas guardadas en {stats_path}")
    print(f"Ejecución del script finalizada a las {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()

# Ejemplo de uso:
# python predictor_V8.py --training training_set.xlsx --validation validation_set.xlsx --scaler minmax --reduction lasso --model xgb --trials auto --no_intro --shap on