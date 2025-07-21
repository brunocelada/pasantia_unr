import os
import logging

# -------------------- CONFIGURACIÓN DE LOGGING --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# -------------------- LISTAS --------------------
NORMALIZACIONES = ["zscore", "minmax", "decimal", "robust", "unit", "none"]
REDUCCIONES = ["pca", "pca_2", "efa", "plsr", "lasso", "rfe", "rfe_2", "sparsepca", "sparsepca_2", "mihr", "pfi", "pfi_2", "none"]
MODELOS = ["mlr", "svm", "dt", "rf", "xgb", "knn", "ann", "gam", "gpr", "lgbm", "catb", "gbr", "adaboost"]

def main():
    base_folder = input("Indica la ruta de la carpeta base que contiene los modelos: ").strip()
    output_folder = os.path.join(base_folder, "0.1_Model_Lists")
    os.makedirs(output_folder, exist_ok=True)

    completed_models = []
    
    # -------------------- Buscar archivos existentes --------------------
    logging.info("Buscando modelos completados...")
    for subfolder in os.listdir(base_folder):
        subfolder_path = os.path.join(base_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        excel_found = False
        for filename in os.listdir(subfolder_path):
            if filename.startswith("results-") and filename.endswith(".xlsx"):
                excel_found = True
                break

        if excel_found:
            completed_models.append(subfolder)

    # Guardar completed_models.txt
    completed_path = os.path.join(output_folder, "completed_models.txt")
    with open(completed_path, "w", encoding="utf-8") as f:
        for model in completed_models:
            f.write(model + "\n")
    logging.info(f"Guardado: {completed_path}")

    # -------------------- Generar todas las combinaciones --------------------
    full_model_list = []
    for norm in NORMALIZACIONES:
        for red in REDUCCIONES:
            for model in MODELOS:
                name = f"{norm}_{red}_{model}"
                full_model_list.append(name)

    # Guardar full_model_list.txt
    full_list_path = os.path.join(output_folder, "full_model_list.txt")
    with open(full_list_path, "w", encoding="utf-8") as f:
        for model in full_model_list:
            f.write(model + "\n")
    logging.info(f"Guardado: {full_list_path}")

    # -------------------- Detectar los que faltan --------------------
    missing_models = [m for m in full_model_list if m not in completed_models]

    # Guardar missing_models.txt
    missing_path = os.path.join(output_folder, "missing_models.txt")
    with open(missing_path, "w", encoding="utf-8") as f:
        for model in missing_models:
            f.write(model + "\n")
    logging.info(f"Guardado: {missing_path}")

    # Mostrar solo missing en terminal
    print("\nModelos que faltan calcular:\n" + "-"*40)
    for model in missing_models:
        print(model)
    print(f"\nTotal faltantes: {len(missing_models)} de {len(full_model_list)}")

if __name__ == "__main__":
    main()
