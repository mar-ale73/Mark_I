import MetaTrader5 as mt5
import pandas as pd
import yaml
from conexion.easy_Trading import Basic_funcs
from procesamiento.features import aplicar_todos_los_indicadores
from modelos.prophet_model import entrenar_modelo_prophet, predecir_precio
from agentes.agente_analisis import generar_senal_operativa
from agentes.agente_portafolio import asignar_capital
from agentes.agente_ejecucion import ejecutar_operacion, generar_reporte_excel

# === Cargar configuración desde YAML ===
with open("utils/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# === Parámetros de configuración ===
simbolo = config["simbolo"]
timeframe_str = config["timeframe"]
cantidad = config["cantidad_datos"]
modelo_str = config["modelo"]
pasos_pred = config["pasos_prediccion"]
frecuencia_pred = config["frecuencia_prediccion"]
umbral_senal = config["umbral_senal"]

# Mapear timeframe
timeframes = {
    'M1': mt5.TIMEFRAME_M1,
    'M5': mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15,
    'H1': mt5.TIMEFRAME_H1,
    'D1': mt5.TIMEFRAME_D1,
}
timeframe = timeframes[timeframe_str]

# === Conexión MT5 ===
login = 68238343
clave = 'Colombia123*'
servidor = 'RoboForex-Pro'
path = r'C:\Program Files\RoboForex - MetaTrader 5\terminal64.exe'

if not mt5.initialize(login=login, password=clave, server=servidor, path=path):
    print("❌ Error al conectar a MT5:", mt5.last_error())
    quit()
else:
    print("✅ Conexión establecida con MetaTrader 5")

BF = Basic_funcs(login, clave, servidor, path)

print("⏳ Extrayendo datos de MT5...")
df = BF.get_data_for_bt(timeframe, simbolo, cantidad)

print("📈 Calculando indicadores técnicos...")
df_indicadores = aplicar_todos_los_indicadores(df)

if modelo_str == "prophet":
    print("🤖 Entrenando modelo Prophet...")
    modelo = entrenar_modelo_prophet(df_indicadores)
    print("🔮 Generando predicción futura...")
    predicciones = predecir_precio(modelo, pasos=pasos_pred, frecuencia=frecuencia_pred)
    print(predicciones)
else:
    raise ValueError(f"Modelo '{modelo_str}' no implementado aún.")

senal = generar_senal_operativa(predicciones, umbral=umbral_senal)
print(f"📢 Señal generada: {senal}")

balance, _, _, _ = BF.info_account()
capital = asignar_capital(balance, senal)
print(f"💰 Capital asignado según la señal: ${capital}")

precio_actual = df_indicadores['Close'].iloc[-1]
operacion = ejecutar_operacion(simbolo, senal, capital, precio_actual)
print(f"🧾 Operación simulada: {operacion}")

generar_reporte_excel(predicciones, senal, capital, operacion, umbral=umbral_senal)

mt5.shutdown()
print("🛑 Conexión cerrada")
