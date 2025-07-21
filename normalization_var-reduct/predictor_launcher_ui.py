import tkinter as tk
from tkinter import filedialog, messagebox
import predictor_launcher as launcher

# agregar SHAP


def seleccionar_archivo(entry):
    archivo = filedialog.askopenfilename()
    if archivo:
        entry.delete(0, tk.END)
        entry.insert(0, archivo)


def validar_y_ejecutar():
    try:
        # Lectura de valores
        training_file = entry_training.get()
        validation_file = entry_validation.get()
        predictor_script = entry_script.get()
        trials = int(entry_trials.get())
        logs_folder = entry_logs.get()
        error_folder = entry_errors.get()

        if trials <= 0:
            raise ValueError("TRIALS debe ser mayor que 0")

        scaler_selected = [opt for opt, var in scaler_vars.items() if var.get()]
        reduction_selected = [opt for opt, var in reduction_vars.items() if var.get()]
        model_selected = [opt for opt, var in model_vars.items() if var.get()]

        if not (scaler_selected and reduction_selected and model_selected):
            raise ValueError("Debe seleccionar al menos una opción en cada lista.")

        # Cerrar interfaz
        root.destroy()

        # Sobrescribir variables en el launcher
        launcher.TRAINING_FILE = training_file
        launcher.VALIDATION_FILE = validation_file
        launcher.PREDICTOR_SCRIPT_NAME = predictor_script
        launcher.TRIALS = trials
        launcher.LOGS_SUBFOLDER = logs_folder
        launcher.ERROR_LOGS_FOLDER = error_folder
        launcher.SCALER_OPTIONS = scaler_selected
        launcher.REDUCTION_OPTIONS = reduction_selected
        launcher.MODEL_OPTIONS = model_selected

        # Ejecutar pruebas
        launcher.run_test()

    except Exception as e:
        messagebox.showerror("Error", str(e))


def crear_checkboxes(frame, opciones):
    variables = {}
    for opcion in opciones:
        var = tk.BooleanVar(value=True)
        tk.Checkbutton(frame, text=opcion, variable=var).pack(anchor='w', padx=2, pady=1)
        variables[opcion] = var
    return variables

# ----- Interfaz principal -----
root = tk.Tk()
root.title("Configurador de Predictor")

# Frame 1: Carga de datos y parámetros
data_frame = tk.LabelFrame(root, text="Carga de Archivos y Parámetros", padx=10, pady=10)

data_frame.pack(fill='x', padx=10, pady=5)

# Entradas de archivo
tk.Label(data_frame, text="Archivo Training:").grid(row=0, column=0, sticky='e')
entry_training = tk.Entry(data_frame, width=40)
entry_training.grid(row=0, column=1, padx=5, pady=2)
btn_training = tk.Button(data_frame, text="Seleccionar", command=lambda: seleccionar_archivo(entry_training))
btn_training.grid(row=0, column=2, padx=5)

tk.Label(data_frame, text="Archivo Validation:").grid(row=1, column=0, sticky='e')
entry_validation = tk.Entry(data_frame, width=40)
entry_validation.grid(row=1, column=1, padx=5, pady=2)
btn_validation = tk.Button(data_frame, text="Seleccionar", command=lambda: seleccionar_archivo(entry_validation))
btn_validation.grid(row=1, column=2, padx=5)

tk.Label(data_frame, text="Script Predictor:").grid(row=2, column=0, sticky='e')
entry_script = tk.Entry(data_frame, width=40)
entry_script.grid(row=2, column=1, padx=5, pady=2)
btn_script = tk.Button(data_frame, text="Seleccionar", command=lambda: seleccionar_archivo(entry_script))
btn_script.grid(row=2, column=2, padx=5)

# Parámetros simples
tk.Label(data_frame, text="TRIALS:").grid(row=3, column=0, sticky='e', pady=5)
entry_trials = tk.Entry(data_frame, width=10)
entry_trials.insert(0, str(launcher.TRIALS))
entry_trials.grid(row=3, column=1, sticky='w')

tk.Label(data_frame, text="Logs Folder:").grid(row=4, column=0, sticky='e')
entry_logs = tk.Entry(data_frame, width=20)
entry_logs.insert(0, launcher.LOGS_SUBFOLDER)
entry_logs.grid(row=4, column=1, sticky='w')

tk.Label(data_frame, text="Error Logs Folder:").grid(row=5, column=0, sticky='e')
entry_errors = tk.Entry(data_frame, width=20)
entry_errors.insert(0, launcher.ERROR_LOGS_FOLDER)
entry_errors.grid(row=5, column=1, sticky='w')

# Frame 2: Selección de opciones
options_frame = tk.LabelFrame(root, text="Opciones de Modelado", padx=10, pady=10)
options_frame.pack(fill='both', expand=True, padx=10, pady=5)

# Subframes para checkbox
overview_frame = tk.Frame(options_frame)
overview_frame.pack(fill='both', expand=True)

scaler_frame = tk.LabelFrame(overview_frame, text="SCALER")
scaler_frame.pack(side='left', fill='y', expand=True, padx=5, pady=5)
scaler_vars = crear_checkboxes(scaler_frame, launcher.SCALER_OPTIONS)

reduction_frame = tk.LabelFrame(overview_frame, text="REDUCTION")
reduction_frame.pack(side='left', fill='y', expand=True, padx=5, pady=5)
reduction_vars = crear_checkboxes(reduction_frame, launcher.REDUCTION_OPTIONS)

model_frame = tk.LabelFrame(overview_frame, text="MODEL")
model_frame.pack(side='left', fill='y', expand=True, padx=5, pady=5)
model_vars = crear_checkboxes(model_frame, launcher.MODEL_OPTIONS)

# Botón de ejecución
btn_run = tk.Button(root, text="Ejecutar", command=validar_y_ejecutar, bg="green", fg="white")
btn_run.pack(pady=10)

root.mainloop()
