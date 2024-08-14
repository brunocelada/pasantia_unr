import subprocess
import sys
import datetime

try:
    import pywhatkit as kit
except ImportError:
    print("pywhatkit no está instalado. Instalando...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pywhatkit"])
    import pywhatkit as kit




# Datos del mensaje
phone_number = "+543415556323"  # Número de teléfono en formato internacional
message = "Hola brenda wacha"
now = datetime.datetime.now()
hour = now.hour
minute = now.minute + 1

# Enviar mensaje
kit.sendwhatmsg(phone_number, message, hour, minute)