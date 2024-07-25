import os

# Obtener la lista de archivos .gjc en la carpeta C:\Linux
files = [f for f in os.listdir('C:\\Linux') if f.endswith('.gjc')]

# Pregunta el valor que deben tomar "a1", "a2" y "a3"
valor_a1 = input('nprocshared= ')
valor_a2 = input('mem=: ')
valor_a3 = input('command line ')

# Recorre cada archivo .gjc
for file in files:
  # Abre el archivo en modo de lectura
  with open(f'C:\\Linux\\{file}', 'r') as f:
    # Lee las líneas del archivo
    lines = f.readlines()

  # Si el archivo tiene al menos dos líneas, reemplaza las dos primeras
  if len(lines) >= 2:
    lines[0] = f'%nprocshared={valor_a1}\n'
    lines[1] = f'%mem={valor_a2}\n'
    # Inserta una nueva línea después de las dos primeras
    lines.insert(2, f'{valor_a3}\n')
    lines.insert(3, '\n')
    lines.insert(4, 'comment\n')
    lines.insert(5, '\n')

  # Recorre las líneas del archivo desde la segunda
  for i in range(1, len(lines)):
    # Si encuentra la palabra "ENDCART", elimina la línea y borra todo el contenido que esté por debajo
    if 'ENDCART' in lines[i]:
      lines = lines[:i]  # Elimina la línea que contiene "ENDCART"
      # Agrega tres líneas vacías
      lines.append('\n')
      lines.append('\n')
      lines.append('\n')
      break

  # Abre el archivo en modo de escritura
  with open(f'C:\\Linux\\{file}', 'w') as f:
    # Escribe las líneas modificadas en el archivo
    f.writelines(lines)

print('Done')
