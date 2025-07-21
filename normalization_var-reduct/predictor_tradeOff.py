import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import os
import logging

"""
----------------------------------------------------------
Análisis de Trade-offs
----------------------------------------------------------

Compara métricas entre sí (por ej: R² vs RMSE) para descubrir si hay compensaciones entre precisión y robustez.
"""

# ----------------------------------------
# Configuración básica del logging
# ----------------------------------------
def setup_logging():
    """
    Configura el logging para registrar mensajes en consola con nivel INFO.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ----------------------------------------
# Función genérica: prompt_input
# Descripción: Solicita al usuario un valor con validación y valor por defecto.
# ----------------------------------------
def prompt_input(message, default=None, validator=None, error_msg="Entrada inválida"):
    """
    Pide al usuario un input con un mensaje personalizado.
    - message: texto que se muestra al usuario.
    - default: valor por defecto si el usuario no ingresa nada.
    - validator: función que recibe el input y retorna True si es válido.
    - error_msg: mensaje de error al reintentar.
    """
    while True:
        try:
            prompt_msg = message
            if default is not None:
                prompt_msg += f" (por defecto '{default}')"
            prompt_msg += ": "
            user_input = input(prompt_msg).strip()
            # Aplica valor por defecto
            result = user_input or default
            # Valida si se proporcionó validador
            if validator and not validator(result):
                raise ValueError(error_msg)
            logger.info(f"Usuario ingresó: {result}")
            return result
        except Exception as e:
            logger.warning(f"{e}. Por favor, intente de nuevo.")

# ----------------------------------------
# Función: prompt_excel_path (usa prompt_input)
# Descripción: Solicita al usuario la ruta del archivo Excel de manera genérica.
# ----------------------------------------
def prompt_excel_path():
    """
    Pide al usuario la ruta del archivo Excel.
    Utiliza prompt_input para manejar validación y valor por defecto.
    """
    return prompt_input(
        message="Ruta del archivo Excel",
        default='C:\\Linux\\Master_Table.xlsx',
        validator=lambda p: bool(p and os.path.splitext(p)[1] in ['.xlsx', '.xls']),
        error_msg="Debe ingresar un nombre de archivo válido con extensión .xlsx o .xls"
    )

def load_master_table(filepath: str, sheet_name: str = None) -> pd.DataFrame:
    """
    Carga el archivo Master_Table desde Excel.
    Si no se especifica sheet_name, usa la hoja activa por defecto.
    """
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    return df

def create_output_folder(filepath: str, folder_name: str = "2.1_TradeOff") -> str:
    """
    Crea la carpeta donde se guardarán los gráficos.
    Usa la misma ubicación que el archivo Excel.
    """
    base_dir = os.path.dirname(filepath)
    output_dir = os.path.join(base_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def plot_tradeoff(df: pd.DataFrame, x_metric: str, y_metric: str, label_col: str, output_dir: str):
    """
    Genera y guarda un gráfico de dispersión para visualizar trade-offs.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_metric, y=y_metric)

    if label_col:
        for _, row in df.iterrows():
            plt.text(row[x_metric], row[y_metric], "",# Agregar , str(row[label_col]) para ver los nombres de los modelos
                     fontsize=8, alpha=0.7)

    plt.title(f"Trade-off: {x_metric} vs {y_metric}")
    plt.xlabel(x_metric)
    plt.ylabel(y_metric)
    plt.tight_layout()

    filename = f"{x_metric}_vs_{y_metric}.png".replace("/", "_").replace("\\", "_")
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()  # Cierra la figura para liberar memoria

    print(f"Guardado: {save_path}")

def main():
    # === CONFIG ===
    filepath = prompt_excel_path()
    sheet_name = "Resumen"
    label_col = "Modelo"

    metrics = [
        "R2_LOOCV",
        "R2_5FoldCV",
        "R2_ObsVsPred",
        "MAE",
        "RMSE",
        "MAPE",
        "Outlier_∆∆G-0.3",
        "Outlier_%ee-30%"
    ]

    df = load_master_table(filepath, sheet_name)

    output_dir = create_output_folder(filepath, "2.1_TradeOff")

    pairs = list(itertools.combinations(metrics, 2))
    print(f"Generando y guardando {len(pairs)} gráficos de trade-offs en: {output_dir}")

    for x_metric, y_metric in pairs:
        plot_tradeoff(df, x_metric, y_metric, label_col, output_dir)

if __name__ == "__main__":
    main()
