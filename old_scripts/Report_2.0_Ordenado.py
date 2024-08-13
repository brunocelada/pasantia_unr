import glob
import os

# Obtener la lista de carpetas en el directorio principal
carpetas = glob.glob("C:\\linux\\*")

# Iterar sobre cada carpeta
for carpeta in carpetas:
    # Obtener el nombre de la carpeta actual
    nombre_carpeta = os.path.basename(carpeta)

    # Cambiar al directorio de la carpeta actual
    os.chdir(carpeta)

    # Listar los archivos .out en la carpeta actual
    listing = glob.glob("*.out")

    # Renombrar cada archivo .out a .log
    for file in listing:
        os.rename(file, (file.rsplit(".", 1)[0]) + ".log")

    # Contar el número de archivos .log en la carpeta actual
    counter = len(glob.glob("*.log"))

    # Abrir o crear el archivo report.csv en modo de escritura
    with open(nombre_carpeta + ".csv", "w") as new:
        # Escribir el encabezado en el archivo CSV
        print("Compound_Name,SCF,ZPE,Enthalpie,Gibbs", file=new)

        # Iterar sobre cada archivo .log en la carpeta actual
        for file in glob.glob("*.log"):
            # Abrir el archivo .log actual en modo lectura
            with open(file, "r") as old:
                rline = old.readlines()

                scf = 0
                gibbs = 0

                # Buscar la línea que contiene "SCF Done:"
                for i in range(len(rline)):
                    if "SCF Done:" in rline[i]:
                        scf = i

                # Buscar la línea que contiene "Sum of electronic and thermal Free Energies="
                for j in range(len(rline)):
                    if "Sum of electronic and thermal Free Energies=" in rline[j]:
                        gibbs = j

                # Extraer los valores necesarios
                if gibbs == 0:
                    print((file.rsplit(".", 1)[0]) + "," + rline[scf].split()[4] + ",-,-,-", file=new)
                else:
                    wzpe = rline[gibbs - 3].split()
                    wentalpie = rline[gibbs - 1].split()
                    wgibbs = rline[gibbs].split()
                    print((file.rsplit(".", 1)[0]) + "," + rline[scf].split()[4] + "," + wzpe[6] + "," + wentalpie[6] + "," + wgibbs[7], file=new)

print("Terminó")