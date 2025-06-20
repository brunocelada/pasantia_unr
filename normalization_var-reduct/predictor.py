#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import pandas as pd
import numpy as np
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
from sklearn.ensemble import RandomForestRegressor
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
    print("Warning: LightGBM not installed. LGBM model will not be available.")

# CatBoost (Optional)
try:
    from catboost import CatBoostRegressor
    CATBOOST_INSTALLED = True
except ImportError:
    CATBOOST_INSTALLED = False
    print("Warning: CatBoost not installed. CatBoost model will not be available.")


# Model selection and metrics
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, cross_val_score, GridSearchCV, learning_curve, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error, make_scorer
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning 

# Factor Analyzer for EFA
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

# Global random state for reproducibility
RANDOM_STATE = 42
# Suppress Optuna trial logging messages
optuna.logging.set_verbosity(optuna.logging.WARNING)


def rmse(y_true, y_pred):
    """Calculates Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    """Calculates Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    if np.sum(non_zero_mask) == 0: 
        return np.nan if np.sum(y_pred != 0) > 0 else 0.0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

rmse_scorer = make_scorer(rmse, greater_is_better=False)
mape_scorer = make_scorer(mape, greater_is_better=False)

def sanitize_feature_names(feature_names):
    """Sanitizes feature names for compatibility with models like XGBoost."""
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
    """Loads data from an Excel file."""
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        exit(1)
    except Exception as e:
        print(f"Error loading Excel file {file_path}: {e}")
        exit(1)

    if df.shape[1] < 2:
        print(f"Error: Data in {file_path} must have at least 2 columns (identifier/features + target).")
        exit(1)
        
    identifiers = df.iloc[:, 0].copy()
    X = df.iloc[:, 1:-1].copy()
    y = df.iloc[:, -1].copy()
    
    if X.ndim == 1: 
        X = X.to_frame()
        
    original_feature_names = X.columns.tolist()
    
    return X, y, identifiers, original_feature_names

def apply_scaler(X_train, X_val, scaler_type):
    """Applies the selected scaling method."""
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
        print(f"Error: Unknown scaler type '{scaler_type}'.")
        exit(1)
    
    X_train_scaled = pd.DataFrame(X_train_scaled_np, columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled_np, columns=X_val.columns, index=X_val.index)
            
    return X_train_scaled, X_val_scaled, scaler

def apply_reduction(X_train, y_train, X_val, reduction_type, original_feature_names):
    """Applies the selected dimensionality reduction method."""
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
             print("Warning: PCA cannot be applied with 0 samples or 0 features. Skipping PCA.")
             reduction_info["status"] = "Skipped (0 samples/features)"
        else:
            min_samples_in_cv_train_split = n_samples_total - (n_samples_total // cv_folds_for_gridsearch) if cv_folds_for_gridsearch > 1 else n_samples_total
            max_n_comp = min(n_features, min_samples_in_cv_train_split)
            if max_n_comp < 1: max_n_comp = 1 
            pca_n_components_options = list(range(1, int(max_n_comp) + 1))
            if not pca_n_components_options and max_n_comp >=1 : pca_n_components_options = [1]
            if not pca_n_components_options:
                print(f"Warning: PCA cannot determine valid n_components range (max_n_comp={max_n_comp}). Using fallback.")
                reduction_info["status"] = "Skipped (invalid n_components range for GridSearchCV)"
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
                    print(f"Warning: PCA GridSearchCV failed ({e}). Using PCA to explain 95% variance as fallback.")
                    try:
                        reducer_object = PCA(n_components=0.95, random_state=RANDOM_STATE) 
                        X_train_transformed, X_val_transformed = reducer_object.fit_transform(X_train.values), reducer_object.transform(X_val.values)
                        new_feature_names = [f"PC{i+1}" for i in range(reducer_object.n_components_)]
                        X_train_reduced_df = pd.DataFrame(X_train_transformed, columns=new_feature_names, index=X_train.index)
                        X_val_reduced_df = pd.DataFrame(X_val_transformed, columns=new_feature_names, index=X_val.index)
                        reduction_info["best_n_components"] = reducer_object.n_components_
                        reduction_info["status"] = "Fallback to 95% explained variance"
                    except Exception as e_fallback:
                        print(f"PCA fallback also failed: {e_fallback}. Using original features.")
                        new_feature_names = original_feature_names 
                        reduction_info["status"] = "PCA failed, using original features"
    
    elif reduction_type == "efa":
        if X_train.shape[1] < 2:
            print("Warning: EFA requires at least 2 features. Skipping EFA.")
            reduction_info["status"] = "Skipped (too few features)"
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning); warnings.simplefilter("ignore", RuntimeWarning) 
                    chi_square_value, p_value = calculate_bartlett_sphericity(X_train.values) 
                    kmo_all, kmo_model = calculate_kmo(X_train.values) 
                reduction_info["bartlett_sphericity"] = {"chi_square": chi_square_value, "p_value": p_value}
                reduction_info["kmo"] = {"kmo_model": kmo_model}
                if p_value > 0.05 or kmo_model < 0.6: print("Warning: Data may not be suitable for EFA (Bartlett's p > 0.05 or KMO < 0.6).")
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
                    print(f"Warning: EFA Kaiser criterion. Defaulting to {best_n_factors} factors.")
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
                    print("Warning: EFA 0 factors. Using original features.")
                    new_feature_names = original_feature_names 
                    reduction_info["status"] = "Skipped (0 factors found)"
            except Exception as e:
                print(f"Error EFA: {e}. Using original features.")
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
            except ValueError as e:
                print(f"Warning: PLSR GridSearchCV failed ({e}).")
                n_c = min(X_train.shape[0], X_train.shape[1])
                if n_c > 0:
                    reducer_object = PLSRegression(n_components=n_c, scale=False)
                    reducer_object.fit(X_train.values, y_train_ravel)
                    X_train_transformed, X_val_transformed = reducer_object.transform(X_train.values), reducer_object.transform(X_val.values)
                    new_feature_names = [f"PLSC{i+1}" for i in range(reducer_object.n_components)]
                    X_train_reduced_df = pd.DataFrame(X_train_transformed, columns=new_feature_names, index=X_train.index)
                    X_val_reduced_df = pd.DataFrame(X_val_transformed, columns=new_feature_names, index=X_val.index)
                    reduction_info["best_n_components"] = reducer_object.n_components
                else: # Should not happen if max_comps > 0
                    new_feature_names, reduction_info["best_n_components"] = [], 0
                    X_train_reduced_df = pd.DataFrame(np.array([]).reshape(X_train.shape[0], 0), columns=new_feature_names, index=X_train.index)
                    X_val_reduced_df = pd.DataFrame(np.array([]).reshape(X_val.shape[0], 0), columns=new_feature_names, index=X_val.index)
        else: print("Warning: Not enough features/samples for PLSR.")

    elif reduction_type == "lasso": # Identical to previous correct version
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
            if not new_feature_names and X_train.shape[1] > 0: print("Warning: LASSO selected 0 features.")
            X_train_reduced_df = pd.DataFrame(X_train_transformed_np, columns=new_feature_names, index=X_train.index)
            X_val_reduced_df = pd.DataFrame(X_val_transformed_np, columns=new_feature_names, index=X_val.index)
            reduction_info["n_selected_features"], reduction_info["selected_features_names"] = len(new_feature_names), new_feature_names
        else: print("Warning: No features to apply LASSO."); reduction_info["status"] = "Skipped (0 input features)"

    elif reduction_type == "rfe": # Identical
        if X_train.shape[1] > 0:
            estimator_for_rfe = LinearRegression(n_jobs=1) 
            cv_rfe = cv_folds_for_gridsearch
            if X_train.shape[0] < 2: print("Warning: Not enough samples for RFE CV."); reduction_info["status"] = "Skipped (too few samples for CV)"
            else:
                try:
                    reducer_object = RFECV(estimator_for_rfe, step=1, cv=cv_rfe, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0) 
                    reducer_object.fit(X_train.values, y_train_ravel) 
                    X_train_transformed_np, X_val_transformed_np = reducer_object.transform(X_train.values), reducer_object.transform(X_val.values)
                    reduction_info["n_selected_features"] = reducer_object.n_features_
                    selected_mask = reducer_object.get_support()
                    new_feature_names = [name for i, name in enumerate(original_feature_names) if selected_mask[i]]
                    if not new_feature_names and X_train.shape[1] > 0: print("Warning: RFE selected 0 features.")
                    X_train_reduced_df = pd.DataFrame(X_train_transformed_np, columns=new_feature_names, index=X_train.index)
                    X_val_reduced_df = pd.DataFrame(X_val_transformed_np, columns=new_feature_names, index=X_val.index)
                    reduction_info["selected_features_names"] = new_feature_names
                except Exception as e: print(f"Error during RFE: {e}. Using original features."); new_feature_names, reduction_info["status"] = original_feature_names, f"Error: {e}"
        else: print("Warning: No features to apply RFE."); reduction_info["status"] = "Skipped (0 input features)"

    elif reduction_type == "sparsepca": # Identical
        pipe = Pipeline([('sparsepca', SparsePCA(random_state=RANDOM_STATE, max_iter=100000)), ('regressor', dummy_regressor)])
        n_features, n_samples_total = X_train.shape[1], X_train.shape[0]
        if n_samples_total == 0 or n_features == 0: print("Warning: SparsePCA cannot be applied with 0 samples or 0 features. Skipping."); reduction_info["status"] = "Skipped (0 samples/features)"
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
            if not spca_n_components_options: print(f"Warning: SparsePCA cannot determine valid n_components range (max_n_comp={max_n_comp}). Using fallback."); reduction_info["status"] = "Skipped (invalid n_components range for GridSearchCV)"
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
                    print(f"Warning: SparsePCA GridSearchCV failed ({e}). Using SparsePCA with default n_components and alpha=1 as fallback.")
                    n_c = min(X_train.shape[1], X_train.shape[0], 3) 
                    if n_c > 0:
                        try:
                            reducer_object = SparsePCA(n_components=n_c, alpha=1, random_state=RANDOM_STATE, max_iter=100000)
                            X_train_transformed, X_val_transformed = reducer_object.fit_transform(X_train.values), reducer_object.transform(X_val.values)
                            new_feature_names = [f"SPC{i+1}" for i in range(reducer_object.n_components_)]
                            X_train_reduced_df, X_val_reduced_df = pd.DataFrame(X_train_transformed, columns=new_feature_names, index=X_train.index), pd.DataFrame(X_val_transformed, columns=new_feature_names, index=X_val.index)
                            reduction_info["best_params"], reduction_info["status"] = {"n_components": n_c, "alpha": 1}, "Fallback to default n_components/alpha"
                        except Exception as e_fallback: print(f"SparsePCA fallback also failed: {e_fallback}. Using original features."); new_feature_names, reduction_info["status"] = original_feature_names, "SparsePCA failed, using original features"
                    else: new_feature_names, reduction_info["status"] = original_feature_names, "SparsePCA skipped (not enough samples/features for fallback)"
    
    elif reduction_type == "mihr": # Identical
        print("Applying MIHR (Made In Home Reductor)...")
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
        print(f"MIHR - After collinearity removal (threshold > {collinearity_threshold}): {len(features_after_collinearity_removal)} features remaining.")
        if X_temp_mihr.empty or X_temp_mihr.shape[1] == 0: print("MIHR - No features left. Using original features (if any)."); new_feature_names, reduction_info["status"] = original_feature_names, "No features after collinearity removal"
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
                print(f"MIHR - SFS failed: {e_sfs}. Using features after collinearity removal (if any).")
                new_feature_names = features_after_collinearity_removal
                X_train_reduced_df = X_train[new_feature_names] if new_feature_names else pd.DataFrame(index=X_train.index)
                X_val_reduced_df = X_val[new_feature_names] if new_feature_names else pd.DataFrame(index=X_val.index)
                reduction_info["status"] = f"SFS failed: {e_sfs}"
        if not new_feature_names: print("MIHR - Resulted in 0 features. Using original features fallback."); new_feature_names, X_train_reduced_df, X_val_reduced_df, reduction_info["status"] = original_feature_names, X_train.copy(), X_val.copy(), "MIHR resulted in 0 features, using original"

    elif reduction_type == "none": pass 
    else: print(f"Error: Unknown reduction type '{reduction_type}'."); exit(1)

    reduction_info["n_features_after"] = X_train_reduced_df.shape[1]
    return X_train_reduced_df, X_val_reduced_df, reducer_object, reduction_info, new_feature_names

# --- Model Training Functions ---

def train_model_base(X_train_df, y_train_ravel, model_type, trial, current_feature_names, catb_train_dir_override=None):
    """Helper to create model instance based on type and trial for Optuna."""
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
        # train_dir is passed here for Optuna trials and final model fitting
        model = CatBoostRegressor(**params, random_state=RANDOM_STATE, verbose=0, thread_count=1, train_dir=catb_train_dir_override)
    else:
        raise ValueError(f"Model type {model_type} configuration error in train_model_base.")
    return model

def train_model(X_train_df, y_train, model_type, n_trials_optuna, current_feature_names, output_dir_name): 
    """Main training function dispatcher with Optuna."""
    if X_train_df.shape[1] == 0 and model_type not in ["gam", "gpr"]: 
        print(f"Warning: No features available for training model {model_type}. Skipping training.")
        return None, {}, None 
    if model_type == "lgbm" and not LGBM_INSTALLED:
        print(f"LightGBM not installed. Skipping {model_type} training.")
        return None, {}, None
    if model_type == "catb" and not CATBOOST_INSTALLED:
        print(f"CatBoost not installed. Skipping {model_type} training.")
        return None, {}, None

    y_train_ravel = np.ravel(y_train)
    
    print(f"Starting Optuna optimization for {model_type} with {n_trials_optuna} trials...")

    # This is the directory where CatBoost will create its 'catboost_info' and 'tmp' subdirectories.
    catboost_base_output_path = None
    if model_type == "catb":
        # Define the specific subdirectory for CatBoost files
        catboost_base_output_path = os.path.join(output_dir_name, "catboost_info")
        os.makedirs(catboost_base_output_path, exist_ok=True) 
        print(f"CatBoost files (including its own 'catboost_info' and 'tmp' subdirs) will be in: {catboost_base_output_path}")


    def progress_callback(study, trial):
        trial_num = trial.number + 1 
        print(f"Optimizing {model_type} - Optuna Trial: {trial_num}/{n_trials_optuna} completed.", end='\r', flush=True)

    def objective(trial):
        n_samples = X_train_df.shape[0]
        cv_folds = min(5, n_samples)
        if cv_folds < 2: 
            if n_samples >= 2: cv_folds = 2
            else: return float('inf') 

        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        
        # Pass the designated base path for CatBoost files to each trial
        model = train_model_base(X_train_df, y_train_ravel, model_type, trial, current_feature_names, catb_train_dir_override=catboost_base_output_path)

        if model_type == "gam": 
            fold_scores = []
            if X_train_df.shape[1] == 0: 
                 current_lam = trial.suggest_float('lam', 1e-3, 1e3, log=True) 
                 gam_model_fold = LinearGAM(lam=current_lam, callbacks=[])
                 try:
                    if y_train_ravel.size > 0: return mean_squared_error(y_train_ravel, np.full_like(y_train_ravel, np.mean(y_train_ravel)))
                    else: return float('inf') 
                 except Exception: return float('inf')

            for train_idx, val_idx in cv.split(X_train_df.values, y_train_ravel):
                X_fold_train, X_fold_val = X_train_df.values[train_idx], X_train_df.values[val_idx]
                y_fold_train, y_fold_val = y_train_ravel[train_idx], y_train_ravel[val_idx]
                
                current_n_splines = trial.params.get('n_splines', 10) 
                current_lam = trial.params.get('lam', 0.6)        
                terms_gen_fold = (s(i, n_splines=current_n_splines) for i in range(X_fold_train.shape[1]))
                terms_fold = sum(terms_gen_fold, TermList())
                gam_model_fold = LinearGAM(terms_fold, lam=current_lam, callbacks=[])
                try:
                    gam_model_fold.fit(X_fold_train, y_fold_train)
                    preds = gam_model_fold.predict(X_fold_val)
                    fold_scores.append(mean_squared_error(y_fold_val, preds))
                except Exception: fold_scores.append(np.inf); break 
            if not fold_scores or np.all(np.isinf(fold_scores)): return float('inf')
            return np.nanmean(fold_scores) 
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UndefinedMetricWarning)
                warnings.simplefilter("ignore", ConvergenceWarning) 
                score = cross_val_score(model, X_train_df, y_train_ravel, cv=cv, scoring='neg_mean_squared_error', n_jobs=1).mean() 
            return -score 
        except Exception as e:
            return float('inf') 

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction='minimize', sampler=sampler) 
    try:
        study.optimize(objective, n_trials=n_trials_optuna, callbacks=[progress_callback], n_jobs=1, show_progress_bar=False) 
    except Exception as e:
        print(f"\nError during Optuna optimization for {model_type}: {e}") 
        return None, {}, study
    finally: print() 

    best_params = study.best_params
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
            print(f"Error fitting final model {model_type} with best params: {e}")
            final_model = None 
    
    return final_model, best_params, study

# ... (evaluate_model_cv, plot_training_learning_curve, plot_predictions_scatter remain the same)
def evaluate_model_cv(model_params, X_df, y, model_type, cv_method_str, current_feature_names_for_gam):
    """Evaluates the model using specified cross-validation."""
    if X_df.shape[1] == 0 and model_type not in ["gam", "gpr"]:
        print(f"Warning: No features to evaluate model {model_type} for {cv_method_str}. Returning NaNs.")
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
    elif model_type == "ann":
        n_layers = model_params.get('n_layers',1)
        hls_keys = [k for k in model_params if k.startswith('n_units_l')]
        hls = [model_params[k] for k in sorted(hls_keys, key=lambda x: int(x.split('l')[-1]))] if hls_keys else [64]*n_layers
        mlp_eval_params = {k: v for k, v in model_params.items() if not k.startswith('n_units_l') and k != 'n_layers'}
        mlp_eval_params['hidden_layer_sizes'] = tuple(hls)
        model_cv = MLPRegressor(**mlp_eval_params, random_state=RANDOM_STATE, max_iter=2000, early_stopping=True, n_iter_no_change=25)
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
        else: raise ValueError("cv_method_str must be 'loocv' or '5-fold'")

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
        return r2_cv, rmse(y_ravel[~np.isnan(y_pred_cv)], y_pred_cv[~np.isnan(y_pred_cv)]), mape(y_ravel[~np.isnan(y_pred_cv)], y_pred_cv[~np.isnan(y_pred_cv)])

    if model_cv is None: 
        print(f"Error: Could not reconstruct model {model_type} for CV or model not available.")
        return np.nan, np.nan, np.nan

    if cv_method_str == "loocv":
        if n_samples < 2: print("Warning: LOOCV requires at least 2 samples. Skipping."); return np.nan, np.nan, np.nan
        cv_obj = LeaveOneOut()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UndefinedMetricWarning); warnings.simplefilter("ignore", ConvergenceWarning)
                y_pred_loocv = cross_val_predict(model_cv, X_df, y_ravel, cv=cv_obj, n_jobs=1) 
            return r2_score(y_ravel, y_pred_loocv), rmse(y_ravel, y_pred_loocv), mape(y_ravel, y_pred_loocv)
        except Exception as e: print(f"Error during LOOCV for {model_type}: {e}"); return np.nan, np.nan, np.nan
    elif cv_method_str == "5-fold":
        n_splits_kfold = min(5, n_samples)
        if n_splits_kfold < 2: print(f"Warning: {cv_method_str} requires at least 2 samples. Skipping."); return np.nan, np.nan, np.nan
        cv_obj = KFold(n_splits=n_splits_kfold, shuffle=True, random_state=RANDOM_STATE)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UndefinedMetricWarning); warnings.simplefilter("ignore", ConvergenceWarning)
                r2_cv = cross_val_score(model_cv, X_df, y_ravel, cv=cv_obj, scoring='r2', n_jobs=1).mean() 
                rmse_cv = -cross_val_score(model_cv, X_df, y_ravel, cv=cv_obj, scoring=rmse_scorer, n_jobs=1).mean() 
                mape_cv = -cross_val_score(model_cv, X_df, y_ravel, cv=cv_obj, scoring=mape_scorer, n_jobs=1).mean() 
            return r2_cv, rmse_cv, mape_cv
        except Exception as e: print(f"Error during 5-Fold CV for {model_type}: {e}"); return np.nan, np.nan, np.nan
    else: raise ValueError("cv_method_str must be 'loocv' or '5-fold'")

def plot_training_learning_curve(model, model_type, X_train_df, y_train, output_path, sklearn_scoring_metric, score_display_name, best_hyperparams_for_gam=None):
    """Plots training/learning curves using K-Fold CV."""
    plt.figure(figsize=(10, 6))
    y_train_ravel = np.ravel(y_train)
    X_train_plot_data = X_train_df.values if model_type == "gam" else X_train_df
    estimator_to_use_for_lc = model 

    if model_type == "gam": 
        n_splines_lc, lam_lc = 10, 0.6      
        if best_hyperparams_for_gam: 
            n_splines_lc = best_hyperparams_for_gam.get('n_splines', n_splines_lc)
            lam_lc = best_hyperparams_for_gam.get('lam', lam_lc)
        if X_train_plot_data.shape[1] > 0:
            terms_lc = sum((s(i, n_splines=int(n_splines_lc)) for i in range(X_train_plot_data.shape[1])), TermList())
        else: terms_lc = TermList() 
        estimator_to_use_for_lc = LinearGAM(terms_lc, lam=float(lam_lc), callbacks=[])
    elif model_type == "gpr":
        pass # Using the fitted model directly for learning curve
        

    if X_train_plot_data.shape[1] == 0 and model_type not in ["gam", "gpr"]: 
        plt.text(0.5, 0.5, "No features for training curve.", ha='center', va='center')
        plt.title(f"Training Curve for {model_type} (No Features)")
    elif model_type == "ann" and hasattr(model, 'loss_curve_') and sklearn_scoring_metric == 'neg_mean_squared_error': 
        plt.plot(model.loss_curve_, label="Training Loss (ANN)")
        plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.title(f"Training Loss Curve for {model_type}"); plt.legend()
    elif X_train_plot_data.shape[0] >= 2 : 
        n_samples_lc = X_train_plot_data.shape[0] 
        
        k_folds_for_curve = n_samples_lc if n_samples_lc < 5 and n_samples_lc >=2 else 5
        if n_samples_lc < 2: k_folds_for_curve = 0

        if k_folds_for_curve < 2:
            plt.text(0.5, 0.5, f"Not enough samples (N={n_samples_lc}) for K-Fold learning curve (k={k_folds_for_curve}).", ha='center', va='center', wrap=True)
            plt.title(f"Learning Curve for {model_type} (K-Fold Setup Error)")
            plt.grid(True); plt.tight_layout(); plt.savefig(output_path); plt.close()
            return

        cv_obj_for_lc = KFold(n_splits=k_folds_for_curve, shuffle=True, random_state=RANDOM_STATE)
        cv_method_label = f"{k_folds_for_curve}-Fold CV"
        
        num_train_samples_in_fold = n_samples_lc - (n_samples_lc // k_folds_for_curve)
        min_abs_train_size = k_folds_for_curve 
        max_abs_train_size = num_train_samples_in_fold
        train_sizes_abs = np.array([])

        if max_abs_train_size < min_abs_train_size:
             print(f"Warning: For {cv_method_label} curve with n_samples={n_samples_lc}, max train size is {max_abs_train_size}, min required is {min_abs_train_size}. No points generated.")
        else:
            num_points = min(5, max_abs_train_size - min_abs_train_size + 1 if max_abs_train_size >= min_abs_train_size else 1)
            if num_points <=0: num_points = 1 
            if max_abs_train_size == min_abs_train_size : train_sizes_abs = np.array([min_abs_train_size])
            else: train_sizes_abs = np.unique(np.linspace(min_abs_train_size, max_abs_train_size, num_points).astype(int))
            if len(train_sizes_abs) > 0: train_sizes_abs = train_sizes_abs[train_sizes_abs >= min_abs_train_size]
        
        if len(train_sizes_abs) == 0:
             plt.text(0.5, 0.5, f"Not enough samples or suitable train_sizes for {cv_method_label} learning curve (N={n_samples_lc}).", ha='center', va='center', wrap=True)
             plt.title(f"Learning Curve for {model_type} ({cv_method_label} Setup Error)")
        else:
            print(f"Generating {cv_method_label} learning curve for {model_type} (Metric: {score_display_name}) with train_sizes: {train_sizes_abs}")
            try:
                with warnings.catch_warnings(): 
                    warnings.simplefilter("ignore", UndefinedMetricWarning); warnings.simplefilter("ignore", ConvergenceWarning)
                    actual_train_sizes, train_scores, test_scores = learning_curve(estimator_to_use_for_lc, X_train_plot_data, y_train_ravel, cv=cv_obj_for_lc, n_jobs=1, train_sizes=train_sizes_abs, scoring=sklearn_scoring_metric, random_state=RANDOM_STATE)
                train_scores_mean, test_scores_mean = np.mean(train_scores, axis=1), np.mean(test_scores, axis=1)   

                # Removed DEBUG print statements for R2 scores

                if sklearn_scoring_metric == 'neg_mean_squared_error':
                    train_scores_mean, test_scores_mean = np.sqrt(np.abs(-train_scores_mean)), np.sqrt(np.abs(-test_scores_mean))  
                
                plt.plot(actual_train_sizes, train_scores_mean, 'o-', color="r", label=f"Training score ({score_display_name})")
                if not np.all(np.isnan(test_scores_mean)): plt.plot(actual_train_sizes, test_scores_mean, 'o-', color="g", label=f"Cross-validation score ({cv_method_label}, {score_display_name})")
                else: print(f"Warning: All CV scores for {score_display_name} with {cv_method_label} are NaN. CV score line not plotted.")

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
                
                plt.xlabel("Training examples"); plt.ylabel(score_display_name); plt.title(f"Learning Curve for {model_type} (CV: {cv_method_label})"); plt.legend(loc="best")
            except Exception as e:
                plt.text(0.5, 0.5, f"Could not generate {cv_method_label} learning curve: {e}", ha='center', va='center', wrap=True)
                plt.title(f"Learning Curve for {model_type} ({cv_method_label} Error)")
    else: 
        plt.text(0.5, 0.5, "Not enough samples for learning curve (requires at least 2).", ha='center', va='center')
        plt.title(f"Training/Learning Curve for {model_type}")
    plt.grid(True); plt.tight_layout(); plt.savefig(output_path); plt.close()

# ... (plot_predictions_scatter remains the same) ...
def plot_predictions_scatter(y_train_true, y_train_pred, y_val_true, y_val_pred, output_path):
    """Plots observed vs. predicted values for training and validation sets."""
    plt.figure(figsize=(8, 8))
    y_train_true_np, y_train_pred_np = np.ravel(y_train_true), np.ravel(y_train_pred)
    y_val_true_np, y_val_pred_np = np.ravel(y_val_true), np.ravel(y_val_pred)
    plt.scatter(y_train_true_np, y_train_pred_np, c='gray', marker='o', alpha=0.6, s=50, label="Entrenamiento: Observado vs. Predicho")
    plt.scatter(y_val_true_np, y_val_pred_np, c='blue', marker='x', s=70, label="Validación: Observado vs. Predicho")
    all_true = np.concatenate((y_train_true_np[~np.isnan(y_train_true_np)], y_val_true_np[~np.isnan(y_val_true_np)]))
    all_pred = np.concatenate((y_train_pred_np[~np.isnan(y_train_pred_np)], y_val_pred_np[~np.isnan(y_val_pred_np)]))
    min_val, max_val = (0,1)
    if len(all_true) > 0 and len(all_pred) > 0:
        min_val_data, max_val_data = min(np.min(all_true), np.min(all_pred)), max(np.max(all_true), np.max(all_pred))
        padding = (max_val_data - min_val_data) * 0.05 if (max_val_data - min_val_data) > 0 else 0.1
        min_val, max_val = min_val_data - padding, max_val_data + padding
    elif len(all_true) > 0: 
        min_val_data, max_val_data = np.min(all_true), np.max(all_true)
        padding = (max_val_data - min_val_data) * 0.05 if (max_val_data - min_val_data) > 0 else 0.1
        min_val, max_val = min_val_data - padding, max_val_data + padding
    elif len(all_pred) > 0: 
        min_val_data, max_val_data = np.min(all_pred), np.max(all_pred)
        padding = (max_val_data - min_val_data) * 0.05 if (max_val_data - min_val_data) > 0 else 0.1
        min_val, max_val = min_val_data - padding, max_val_data + padding
    if min_val == max_val: min_val -= 0.5; max_val += 0.5
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Ajuste Ideal (y=x)") 
    plt.xlabel("Valores Observados"); plt.ylabel("Valores Predichos"); plt.title("Observado vs. Predicho (Entrenamiento y Validación)")
    plt.legend(); plt.grid(True); plt.xlim(min_val, max_val); plt.ylim(min_val, max_val)
    plt.tight_layout(); plt.savefig(output_path); plt.close()

def main():
    vader_art = [
        "                                      			@@@@@@",
        "                                      		@@@@@@@@@@@@@@@@",
        "                                           @@@@@@@@@@@@@@@@@@@@@@@@",
        "                            		@@@@@       @@  @@       @@@@@",
        "                            	      @@@@          @@  @@          @@@@",
        "                          	     @@@   @@       @@  @@            @@@",
        "                                    @@@             @@  @@             @@@",
        "                                   @@@ @@@          @@  @@              @@@",
        "                                   @@               @@  @@               @@",
        "                                  @@@               @@  @@   @@@         @@@",
        "                                  @@@               @@  @@ @@@       @@@ @@@",
        "                                   @@    @@@         @@  @@           @@@  @@",
        "                                  @@ @@@@   @@@@@@@ @@  @@ @@@@@@@   @@@@@@@",
        "                                  @@@@             @@@  @@@             @@@@",
        "                                 @@@@     @@@@@@        @@@@  @@@@@@@@@@  @@@",
        "                                @@@@   @@@@@@@@@@@@@      @@@@@@@@@@@@@@@@ @@@",
        "                               @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  @@@@",
        "                              @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  @@@@",
        "                              @@@@@ @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  @@@@",
        "                             @@@@@@@@@@@@@@@@@@@@@@@@@  @@@@@@@@@@@@@@@@@@@@  @@@",
        "                            @@@   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@   @@@",
        "                           @@@    @@@@@@  @@@@     @@@@@@@@     @@@@  @@@@@@@   @@@",
        "                           @@@    @@@@@@@@@        @@@@  @@@@@       @@@@@@@@@@    @@",
        "                         @@@    @@@@@@@@@@     @@@@@@@@@@@@@@@@@     @@@@@@@@@@  @@@@@",
        "                        @@@@@@ @@@@@@@@@@@@  @@@@@@ @@@@@@@ @@@@@@  @@@@@@@@@@@   @@@@@",
        "                       @@@@@@  @@@@@@@@@@@@@@@@ @ @@@@@@@@@@  @ @@@@@@@@@@@@@@@@ @@@@@@@",
        "                        @@@@@  @@@@@@@@@@@@ @@@@@ @@@@@@@@@@@@@ @@@@@@@@@@@@@@@@@  @@@@@@",
        "                       @@@@   @@@@@@@@@@@@@@@  @@@@@@@@@@@@@@@@@  @@@@@@@@@@@@@@@  @@@@",
        "                        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",
        "                            @@@     @@@@@@@@@@@@@@@         @@@@@@@@@ @@@@     @@@@",
        "                                  @@@@@@@@ @@@   @@@@@@@@@@@@@  @@@  @@@@@@@",
        "                                 @@@ @@@   @@@@   @@@@  @@@@    @@      @@ @@@",
        "                                @@@@         @@@   @@@@  @@@@   @@@          @@@",
        "                                @@@@          @@@    @@@  @@@@   @@@    @@@@@@@@@@",
        "                              @@@  @@@@@@@@@@@@@@   @@@  @@@@   @@@@@@@@@@@@@@@@@@",
        "                            @@@@@@@@@@  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ @@@  @@@@@",
        "                            @@@@@@@@@  @@@ @@    @@@@@@@@@@@@   @@@ @@@  @  @@@@@@",
        "                           @@@@@@@@@@ @@@@ @@    @@@@@@@@@@@@   @@@  @@@ @@  @@ @@@",
        "                          @@@@@@@ @@  @@@@@@@@@@ @@@@@@@@@@@@  @@@@@@@@@@ @@  @@ @@@",
        "                          @@@@@@@ @   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  @ @@@@@@@",
        "                          @@@@@@@@@@@@@@@@@@@@@@  @@@@ @@@@@  @@@@@@@@@@@@@@@     @@@",
        "                          @@@ @@@       @@@@@@@@@@@@@@@ @@@@@@@@@@@@@@@@@@         @@@",
        "                         @@@ @@@@      @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@   @     @@@",
        "                         @@@ @@@@ @@@@@@@@@@  @@ @  @@@@@@  @@@@  @@@@@@@@@@@@@@@@@",
        "                          @@@ @@@@@@@@@@@@@ @  @@ @@@     @@@@ @@  @@ @@@@@@@@@@@@@",
        "                         @@@@ @@@@@@@@@@@@ @  @@   @@@@@@@@   @@  @@ @@@@@@ @@  @@",
        "                          @@@ @ @@@@@@@@@@ @  @@   @@@@@@@@   @@  @@ @@@@@@ @@  @@",
        "                          @@@    @@@@@@@@@@@@@@@@  @@@@@@@@  @@@@@@@@@@@@@@     @@",
        "                           @@@    @@@@@@@@      @@@@@@@@@@@@@@@     @@@@@@@    @@@",
        "                            @@@     @@@@@          @@@@@@@@@           @@@@@@@@@@@",
        "                             @@@@@@@@@@@ @@         @@@ @@@@@@@@       @@@@@@@@@@",
        "                              @@@@@  @@@@@@@@@@@@@@@@@  @@@@@@@@@@@@@@@@@@",
        "                                            @@@                 @@"
    ] 
    
    parser = argparse.ArgumentParser(description="Automated Machine Learning Prediction Script")
    parser.add_argument("--training", required=True, help="Path to the training set (.xlsx)")
    parser.add_argument("--validation", required=True, help="Path to the validation set (.xlsx)")
    parser.add_argument("--scaler", required=True, choices=["none", "zscore", "minmax", "decimal", "robust", "unit"], help="Type of scaling")
    parser.add_argument("--reduction", required=True, choices=["none", "pca", "efa", "plsr", "lasso", "rfe", "sparsepca", "mihr"], help="Type of dimensionality reduction")
    model_choices = ["mlr", "svm", "dt", "rf", "xgb", "knn", "ann", "gam", "gpr"]
    if LGBM_INSTALLED: model_choices.append("lgbm")
    if CATBOOST_INSTALLED: model_choices.append("catb")
    parser.add_argument("--model", required=True, choices=model_choices, help="Model to train")
    parser.add_argument("--trials", required=True, type=int, help="Number of Optuna trials for hyperparameter optimization")
    parser.add_argument("--no_intro", action="store_true", help="Salta la introducción para ejecución automatizada.") 
    
    args = parser.parse_args()

    if not args.no_intro:
        print("\n")
        for line in vader_art: print(line)
        print("\nPress enter to join the dark side......") 
        input() 
        print("-" * 50) 

    print(f"Starting script execution at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Parameters: Scaler={args.scaler}, Reduction={args.reduction}, Model={args.model}, Trials={args.trials}")
    output_dir_name = os.path.abspath(f"{args.scaler}_{args.reduction}_{args.model}") # Ensure absolute path
    os.makedirs(output_dir_name, exist_ok=True)
    print(f"Output will be saved in: {output_dir_name}")

    print("Loading data...")
    X_train_orig_df, y_train_series, ids_train_series, original_feature_names = load_data(args.training)
    X_val_orig_df, y_val_series, ids_val_series, val_original_feature_names = load_data(args.validation) 
    if not X_train_orig_df.columns.equals(X_val_orig_df.columns):
        print("Warning: Training and validation sets have different feature names/order. Aligning validation to training.")
        try: X_val_orig_df = X_val_orig_df[original_feature_names]
        except KeyError as e:
            missing_cols = list(set(original_feature_names) - set(X_val_orig_df.columns))
            print(f"Error: Validation set is missing columns: {missing_cols}. Details: {e}"); exit(1)

    print(f"Applying scaler: {args.scaler}...")
    X_train_scaled_df, X_val_scaled_df, scaler_obj = apply_scaler(X_train_orig_df, X_val_orig_df, args.scaler)

    # ... (Saving standardized data - code omitted for brevity but is unchanged) ...
    df_train_std = X_train_scaled_df.copy()
    id_col_name_train = ids_train_series.name if ids_train_series.name is not None else 'Identifier'
    target_col_name_train = y_train_series.name if y_train_series.name is not None else 'Target'
    df_train_std.insert(0, id_col_name_train, ids_train_series.values) 
    df_train_std[target_col_name_train] = y_train_series.values
    train_std_path = os.path.join(output_dir_name, "training_set_standardized.xlsx")
    df_train_std.to_excel(train_std_path, index=False)
    print(f"Saved standardized training data to {train_std_path}")

    df_val_std = X_val_scaled_df.copy()
    id_col_name_val = ids_val_series.name if ids_val_series.name is not None else 'Identifier' 
    target_col_name_val = y_val_series.name if y_val_series.name is not None else 'Target'
    df_val_std.insert(0, id_col_name_val, ids_val_series.values) 
    df_val_std[target_col_name_val] = y_val_series.values
    val_std_path = os.path.join(output_dir_name, "validation_set_standardized.xlsx")
    df_val_std.to_excel(val_std_path, index=False)
    print(f"Saved standardized validation data to {val_std_path}")


    print(f"Applying dimensionality reduction: {args.reduction}...")
    X_train_reduced_df, X_val_reduced_df, reducer_obj, reduction_info, final_feature_names_raw = apply_reduction(X_train_scaled_df, y_train_series, X_val_scaled_df, args.reduction, X_train_scaled_df.columns.tolist())
    
    final_feature_names = sanitize_feature_names(final_feature_names_raw)
    # ... (Feature name sanitization and assignment - code omitted for brevity but is unchanged) ...
    if X_train_reduced_df.shape[1] == len(final_feature_names): 
        X_train_reduced_df.columns = final_feature_names
        if X_val_reduced_df.shape[1] == len(final_feature_names): X_val_reduced_df.columns = final_feature_names
        elif X_val_reduced_df.shape[1] > 0 : 
             print(f"Warning: Validation set has {X_val_reduced_df.shape[1]} features, but {len(final_feature_names)} sanitized names. Using generic names for validation.")
             X_val_reduced_df.columns = [f"feature_val_{j}" for j in range(X_val_reduced_df.shape[1])]
    elif X_train_reduced_df.shape[1] > 0 : 
        print(f"Warning: Number of final features ({X_train_reduced_df.shape[1]}) != sanitized names ({len(final_feature_names)}). Using generic names.")
        X_train_reduced_df.columns = [f"feature_train_{j}" for j in range(X_train_reduced_df.shape[1])]
        if X_val_reduced_df.shape[1] == X_train_reduced_df.shape[1]: X_val_reduced_df.columns = X_train_reduced_df.columns
        elif X_val_reduced_df.shape[1] > 0: X_val_reduced_df.columns = [f"feature_val_{j}" for j in range(X_val_reduced_df.shape[1])]
        final_feature_names = X_train_reduced_df.columns.tolist() 
    
    print(f"Reduction resulted in {X_train_reduced_df.shape[1]} features. Final names used: {X_train_reduced_df.columns.tolist()}")

    # ... (Saving used data - code omitted for brevity but is unchanged) ...
    df_train_used = X_train_reduced_df.copy()
    df_train_used.insert(0, id_col_name_train, ids_train_series.values)
    df_train_used[target_col_name_train] = y_train_series.values
    train_used_path = os.path.join(output_dir_name, "training_used.xlsx")
    df_train_used.to_excel(train_used_path, index=False)
    print(f"Saved data used for training to {train_used_path}")

    df_val_used = X_val_reduced_df.copy()
    df_val_used.insert(0, id_col_name_val, ids_val_series.values) 
    df_val_used[target_col_name_val] = y_val_series.values
    val_used_path = os.path.join(output_dir_name, "validation_used.xlsx")
    df_val_used.to_excel(val_used_path, index=False)
    print(f"Saved data used for validation to {val_used_path}")

    print(f"Training model: {args.model}...") 
    # Pass output_dir_name for CatBoost train_dir handling
    trained_model, best_hyperparams, optuna_study = train_model(
        X_train_reduced_df, y_train_series, args.model, args.trials, 
        final_feature_names, output_dir_name 
    )

    if trained_model is None:
        print(f"Model training failed for {args.model}. Exiting.")
        stats_path = os.path.join(output_dir_name, "Statistics.txt")
        with open(stats_path, "w", encoding="utf-8") as f:
            f.write(f"Script execution: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Parameters: Scaler={args.scaler}, Reduction={args.reduction}, Model={args.model}, Trials={args.trials}\n\n")
            f.write("MODEL TRAINING FAILED.\nReason: No features or Optuna/model fitting error.\n")
        exit(1)

    print("Evaluating model with Cross-Validation...")
    r2_loocv, rmse_loocv, mape_loocv = evaluate_model_cv(best_hyperparams, X_train_reduced_df, y_train_series, args.model, "loocv", final_feature_names)
    r2_kfold, rmse_kfold, mape_kfold = evaluate_model_cv(best_hyperparams, X_train_reduced_df, y_train_series, args.model, "5-fold", final_feature_names)

    X_train_pred_input = X_train_reduced_df.values if args.model == "gam" else X_train_reduced_df
    y_train_pred_for_plot = trained_model.predict(X_train_pred_input)

    print("Making predictions on validation set...")
    X_val_pred_input = X_val_reduced_df.values if args.model == "gam" else X_val_reduced_df
    y_val_pred = None 
    if X_val_pred_input.shape[1] == 0 and X_train_reduced_df.shape[1] > 0 :
        print("Warning: Validation set has 0 features. Predicting with training mean.")
        y_val_pred = np.full(y_val_series.shape, y_train_series.mean())
    elif hasattr(trained_model, 'n_features_in_') and X_val_pred_input.shape[1] != trained_model.n_features_in_ and args.model not in ["gam", "gpr"]: 
        print(f"Warning: Mismatch in feature count for validation. Model expects {trained_model.n_features_in_}, got {X_val_pred_input.shape[1]}. Predicting with training mean.")
        y_val_pred = np.full(y_val_series.shape, y_train_series.mean())
    elif args.model == "gam" and hasattr(trained_model, 'terms') and trained_model.terms: 
        try: y_val_pred = trained_model.predict(X_val_pred_input)
        except Exception as e: print(f"Error predicting on validation set for GAM: {e}. Using training mean."); y_val_pred = np.full(y_val_series.shape, y_train_series.mean())
    elif args.model == "gam" and hasattr(trained_model, 'terms') and not trained_model.terms: 
        print("Warning: GAM model has no terms. Predicting with training mean/intercept.")
        try: y_val_pred = trained_model.predict(X_val_pred_input) 
        except Exception as e: print(f"Error predicting on validation for intercept-only GAM: {e}. Using training mean."); y_val_pred = np.full(y_val_series.shape, y_train_series.mean())
    else: 
        try: y_val_pred = trained_model.predict(X_val_pred_input)
        except Exception as e: print(f"Error predicting on validation set: {e}. Using training mean."); y_val_pred = np.full(y_val_series.shape, y_train_series.mean())

    val_r2 = r2_score(y_val_series, y_val_pred) if len(y_val_series) > 0 and not np.all(np.isnan(y_val_pred)) else np.nan
    val_rmse = rmse(y_val_series, y_val_pred) if len(y_val_series) > 0 and not np.all(np.isnan(y_val_pred)) else np.nan
    val_mape = mape(y_val_series, y_val_pred) if len(y_val_series) > 0 and not np.all(np.isnan(y_val_pred)) else np.nan

    print("Saving results...")
    results_df = pd.DataFrame({id_col_name_val : ids_val_series.values, target_col_name_val: y_val_series.values, 'Predicted_Target': np.ravel(y_val_pred)})
    results_path = os.path.join(output_dir_name, f"results-{args.scaler}-{args.reduction}-{args.model}.xlsx") #<----------------------------------------------------------------
    results_df.to_excel(results_path, index=False)
    print(f"Saved prediction results to {results_path}")

    training_curve_rmse_path = os.path.join(output_dir_name, "training_curve_rmse.png") 
    plot_training_learning_curve(trained_model, args.model, X_train_reduced_df, y_train_series, training_curve_rmse_path, sklearn_scoring_metric='neg_mean_squared_error', score_display_name='RMSE', best_hyperparams_for_gam=best_hyperparams if args.model == "gam" else None)
    print(f"Saved RMSE training curve (K-Fold CV) to {training_curve_rmse_path}")

    training_curve_r2_path = os.path.join(output_dir_name, "training_curve_r2.png") 
    plot_training_learning_curve(trained_model, args.model, X_train_reduced_df, y_train_series, training_curve_r2_path, sklearn_scoring_metric='r2', score_display_name='R2 Score', best_hyperparams_for_gam=best_hyperparams if args.model == "gam" else None)
    print(f"Saved R2 training curve (K-Fold CV) to {training_curve_r2_path}")

    prediction_plot_path = os.path.join(output_dir_name, "prediction.png")
    plot_predictions_scatter(y_train_series, y_train_pred_for_plot, y_val_series, y_val_pred, prediction_plot_path)
    print(f"Saved prediction plot to {prediction_plot_path}")

    stats_path = os.path.join(output_dir_name, "Statistics.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"Script Execution Summary\n--------------------------\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training Data: {args.training}\nValidation Data: {args.validation}\n")
        f.write(f"Scaler: {args.scaler}\n")
        try: reduction_info_str = json.dumps(reduction_info, indent=2, default=lambda o: str(o))
        except TypeError: reduction_info_str = str(reduction_info) 
        f.write(f"Reduction: {args.reduction} (Details: {reduction_info_str})\n")
        f.write(f"Model: {args.model}\nOptuna Trials: {args.trials}\n\n")
        f.write(f"Best Hyperparameters for {args.model}:\n")
        for k, v_param in best_hyperparams.items(): f.write(f"  {k}: {v_param}\n")
        f.write("\nCross-Validation Metrics (on Training Data):\n")
        f.write(f"  LOOCV (Global Evaluation):\n    R2: {r2_loocv:.4f}\n    RMSE: {rmse_loocv:.4f}\n    MAPE: {mape_loocv:.4f}%\n")
        f.write(f"  5-Fold CV (Global Evaluation):\n    R2: {r2_kfold:.4f}\n    RMSE: {rmse_kfold:.4f}\n    MAPE: {mape_kfold:.4f}%\n\n")
        f.write(f"Note: Both RMSE and R2 learning curves use K-Fold CV (k determined dynamically, up to 5) internally for stability.\n\n")
        f.write("Validation Set Performance:\n")
        f.write(f"  R2: {val_r2:.4f}\n") 
        f.write(f"    RMSE: {val_rmse:.4f}\n") 
        f.write(f"    MAPE: {val_mape:.4f}%\n") 
    print(f"Saved statistics to {stats_path}")
    print(f"Script execution finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
