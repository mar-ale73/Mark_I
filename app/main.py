import MetaTrader5 as mt5
import pandas as pd
from conexion.easy_Trading import Basic_funcs
from procesamiento.features import aplicar_todos_los_indicadores

# === Parámetros de conexión ===
login = 68238343
clave = 'Colombia123*'
servidor = 'RoboForex-PRO'
path = r'C:\Program Files\MetaTrader 5\terminal64.exe'

# === Inicializar conexión ===
if not mt5.initialize(login=login, password=clave, server=servidor, path=path):
    print("❌ Error al conectar a MT5:", mt5.last_error())
    quit()
else:
    print("✅ Conexión establecida con MetaTrader 5")

# === Crear instancia del manejador ===
BF = Basic_funcs(login, clave, servidor, path)

# === Extraer datos de EURUSD timeframe M15 ===
simbolo = 'EURUSD'
timeframe = mt5.TIMEFRAME_M15
cantidad = 500

print("⏳ Extrayendo datos de MT5...")
df = BF.get_data_for_bt(timeframe, simbolo, cantidad)

# === Aplicar indicadores técnicos ===
print("📈 Calculando indicadores técnicos...")
df_indicadores = aplicar_todos_los_indicadores(df)

# === Mostrar resultados ===
print("✅ Datos enriquecidos con indicadores:")
print(df_indicadores.tail())

# === Finalizar conexión ===
mt5.shutdown()
print("🛑 Conexión cerrada")
