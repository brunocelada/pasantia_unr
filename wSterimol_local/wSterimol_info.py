'''
Tener instalado y actualizado "pip"
    Para actualizar pip: "python.exe -m pip install --upgrade pip"
    Para instalar pip:  "py -m ensurepip --upgrade" o "python.exe -m ensurepip --upgrade"
librerias necesarias:
- numpy
- dbstep
- sterimol

En pymol ejecutar lo siguiente:
run C:\Users\bruno\OneDrive - Church of Jesus Christ\UNR\Pasantias\programas\pasantia_unr\wSterimol\wsterimol/visualize.py
run C:\Users\bruno\OneDrive - Church of Jesus Christ\UNR\Pasantias\programas\pasantia_unr\wSterimol\wsterimol/setup.py
run C:\Users\bruno\OneDrive - Church of Jesus Christ\UNR\Pasantias\programas\pasantia_unr\wSterimol\wsterimol/generate.py
run C:\Users\bruno\OneDrive - Church of Jesus Christ\UNR\Pasantias\programas\pasantia_unr\wSterimol\wsterimol/filter_gen.py
run C:\Users\bruno\OneDrive - Church of Jesus Christ\UNR\Pasantias\programas\pasantia_unr\wSterimol\wsterimol/prepare_file.py
run C:\Users\bruno\OneDrive - Church of Jesus Christ\UNR\Pasantias\programas\pasantia_unr\wSterimol\wsterimol/optimisation.py
run C:\Users\bruno\OneDrive - Church of Jesus Christ\UNR\Pasantias\programas\pasantia_unr\wSterimol\wsterimol/filter_opt.py
run C:\Users\bruno\OneDrive - Church of Jesus Christ\UNR\Pasantias\programas\pasantia_unr\wSterimol\wsterimol/sterimoltools.py
run C:\Users\bruno\OneDrive - Church of Jesus Christ\UNR\Pasantias\programas\pasantia_unr\wSterimol\wsterimol/sterimol.py
run C:\Users\bruno\OneDrive - Church of Jesus Christ\UNR\Pasantias\programas\pasantia_unr\wSterimol\wsterimol/weight.py
run C:\Users\bruno\OneDrive - Church of Jesus Christ\UNR\Pasantias\programas\pasantia_unr\wSterimol\wsterimol/wSterimol.py
'''

import subprocess
import sys

try:
    import numpy
    print("numpy ya está instalado.")
except ImportError:
    print("numpy no está instalado. Instalando...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    import numpy