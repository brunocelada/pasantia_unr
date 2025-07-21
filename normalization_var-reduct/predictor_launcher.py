#!/usr/bin/env python
# coding: utf-8

import subprocess
import os
import itertools
from datetime import datetime

# --- Configuración ---
# ¡IMPORTANTE! Configuración.

# TRAINING_FILE = "training_set.xlsx"
TRAINING_FILE = "training_set.xlsx"

# VALIDATION_FILE = "validation_set.xlsx"
VALIDATION_FILE = "validation_set.xlsx"

# Script a ejecutar, asegúrate de que sea el correcto (predictor.py)
PREDICTOR_SCRIPT_NAME = "predictor.py" 
TRIALS = 500 # Número de trials de Optuna para cada combinación
SHAP = "off" # Por defecto, SHAP está desactivado. Cambia a "on" si quieres activarlo.

# Nombres de las carpetas de logs
LOGS_SUBFOLDER = "logs_new_tests"
ERROR_LOGS_FOLDER = "error" # Nueva carpeta para los logs de errores

# Definición de las opciones para cada argumento de predictor.py
SCALER_OPTIONS = ["zscore", "minmax", "decimal", "robust", "unit", "none"]
REDUCTION_OPTIONS = ["pca", "pca_2", "efa", "plsr", "lasso", "rfe", "rfe_2", "sparsepca", "sparsepca_2", "mihr", "pfi", "pfi_2", "none"]
MODEL_OPTIONS = ["mlr", "svm", "dt", "rf", "xgb", "knn", "ann", "gam", "gpr", "lgbm", "catb", "gbr", "adaboost"]

def run_test():
    """
    Ejecuta todas las combinaciones posibles de escaladores, reductores y modelos
    usando predictor.py, guarda los logs y estima el tiempo restante.
    Los logs de errores se guardan en una carpeta separada.
    """
    # Crear las carpetas de logs si no existen
    if not os.path.exists(LOGS_SUBFOLDER):
        os.makedirs(LOGS_SUBFOLDER)
        print(f"Subcarpeta '{LOGS_SUBFOLDER}' creada para guardar los logs de éxito.")
    if not os.path.exists(ERROR_LOGS_FOLDER):
        os.makedirs(ERROR_LOGS_FOLDER)
        print(f"Subcarpeta '{ERROR_LOGS_FOLDER}' creada para guardar los logs de error.")

    combinations = list(itertools.product(SCALER_OPTIONS, REDUCTION_OPTIONS, MODEL_OPTIONS))
    total_combinations = len(combinations)
    
    print(f"\nSe ejecutarán un total de {total_combinations} combinaciones.")
    print(f"Usando --training '{TRAINING_FILE}', --validation '{VALIDATION_FILE}', y --trials {TRIALS} para todas las pruebas.\n")

    start_time_total = datetime.now()

    for i, (scaler, reduction, model) in enumerate(combinations):
        test_name = f"{scaler}_{reduction}_{model}"
        base_log_filename = f"{test_name}.txt" # Solo el nombre del archivo, la ruta se decide después
        
        command = [
            "python", PREDICTOR_SCRIPT_NAME,
            "--training", TRAINING_FILE,
            "--validation", VALIDATION_FILE,
            "--scaler", scaler,
            "--reduction", reduction,
            "--model", model,
            "--trials", str(TRIALS),
            "--shap", SHAP,
            "--no_intro" 
        ]
        
        print(f"--- Ejecutando Combinación {i+1}/{total_combinations}: {test_name} ---")
        start_time_combination = datetime.now()

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False, encoding='utf-8', errors='replace')
            
            # Decidir la carpeta de destino en función del resultado
            if result.returncode != 0:
                log_filename = os.path.join(ERROR_LOGS_FOLDER, base_log_filename)
                print(f"¡ERROR! La combinación {test_name} finalizó con errores. Log guardado en: {log_filename}")
            else:
                log_filename = os.path.join(LOGS_SUBFOLDER, base_log_filename)
                print(f"Combinación {test_name} completada. Log guardado en: {log_filename}")

            # Escribir el log en la ruta determinada
            with open(log_filename, "w", encoding="utf-8") as log_file:
                log_file.write(f"Comando Ejecutado: {' '.join(command)}\n")
                log_file.write(f"Hora de Inicio: {start_time_combination.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                log_file.write("--- Salida Estándar (stdout) ---\n")
                log_file.write(result.stdout if result.stdout else "No hubo salida estándar.\n")
                log_file.write("\n--- Salida de Error (stderr) ---\n")
                log_file.write(result.stderr if result.stderr else "No hubo salida de error.\n")
                
                if result.returncode != 0:
                    log_file.write(f"\n--- El script {PREDICTOR_SCRIPT_NAME} finalizó con errores (código de salida: {result.returncode}) ---\n")

        except (FileNotFoundError, Exception) as e:
            # Si hay un error al ejecutar el script (ej. no se encuentra), también se guarda en la carpeta de error
            log_filename = os.path.join(ERROR_LOGS_FOLDER, base_log_filename)
            error_message = f"Error crítico al intentar ejecutar {test_name}: {e}"
            print(error_message)
            with open(log_filename, "w", encoding="utf-8") as log_file:
                log_file.write(error_message)
        
        end_time_combination = datetime.now()
        total_accumulated_time = end_time_combination - start_time_total
        
        print(f"Tiempo para esta combinación: {end_time_combination - start_time_combination}")
        print(f"Tiempo total acumulado:     {total_accumulated_time}")

        # --- CÁLCULO DE TIEMPO ESTIMADO RESTANTE (ETR) ---
        combinations_completed = i + 1
        average_time_per_combination = total_accumulated_time / combinations_completed
        combinations_remaining = total_combinations - combinations_completed

        if combinations_remaining >= 0: # Mostrar ETR hasta el final
            estimated_time_remaining_delta = average_time_per_combination * combinations_remaining
            etr_seconds = estimated_time_remaining_delta.total_seconds()
            etr_days = etr_seconds / 86400
            etr_hours = etr_seconds / 3600
            etr_minutes = etr_seconds / 60
            
            print("--------------------------------------------------")
            print(">>> Estimación de Tiempo Restante (ETR):")
            print(f"    {etr_days:.2f} días")
            print(f"    {etr_hours:.2f} horas")
            print(f"    {etr_minutes:.2f} minutos")
            print("--------------------------------------------------\n")

    end_time_total = datetime.now()
    print(f"--- Todas las {total_combinations} combinaciones han sido procesadas. ---")
    print(f"Tiempo total de ejecución: {end_time_total - start_time_total}")

if __name__ == "__main__":
    run_test()