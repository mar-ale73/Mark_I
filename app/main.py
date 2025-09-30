import os
import sys
import argparse
import yaml
import pandas as pd
import MetaTrader5 as mt5
# Los imports “pesados” (features, prophet, agentes, reportes) se hacen dentro de los bloques donde se usan.

def obtener_df_desde_mt5(symbol: str, timeframe, n_barras: int) -> pd.DataFrame:
    """Devuelve DataFrame con columnas: timestamp, Open, High, Low, Close, Volume (UTC)."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, int(n_barras))
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No se obtuvieron datos de {symbol} desde MT5.")
    df = pd.DataFrame(rates)
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close", "tick_volume": "Volume"
    })
    return df[["timestamp", "Open", "High", "Low", "Close", "Volume"]]


# =========================
# 1) CARGA DE CONFIGURACIÓN
# =========================
with open("utils/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Parámetros base desde YAML (con defaults para evitar KeyError en modo benchmark)
simbolo          = config.get("simbolo", "EURUSD")
timeframe_str    = config.get("timeframe", "M5")
cantidad         = int(config.get("cantidad_datos", 2000))

# Estos 4 son del flujo NORMAL; en benchmark pueden no existir. Ponemos defaults seguros.
modelo_str       = str(config.get("modelo", "prophet")).lower()
pasos_pred       = int(config.get("pasos_prediccion", 12))
frecuencia_pred  = str(config.get("frecuencia_prediccion", "H"))
umbral_senal     = float(config.get("umbral_senal", 0.0))

# Nuevos parámetros YAML (trading / reportes)
riesgo_por_trade = float(config.get("riesgo_por_trade", 0.02))
volumen_minimo   = float(config.get("volumen_minimo", 0.01))
stop_loss_pips   = float(config.get("stop_loss_pips", 10))
take_profit_pips = float(config.get("take_profit_pips", 20))
ruta_reporte     = config.get("ruta_reporte", "outputs/reporte_inversion.xlsx")
pip_size_cfg     = config.get("pip_size", None)

# Mapear timeframe
timeframes = {
    'M1': mt5.TIMEFRAME_M1,
    'M5': mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15,
    'H1': mt5.TIMEFRAME_H1,
    'D1': mt5.TIMEFRAME_D1,
}
if timeframe_str not in timeframes:
    raise ValueError(f"Timeframe '{timeframe_str}' no soportado. Usa uno de {list(timeframes.keys())}.")
timeframe = timeframes[timeframe_str]

# --- CLI (incluye 'benchmark') ---
parser = argparse.ArgumentParser()
parser.add_argument(
    "--modo",
    choices=["normal", "eda", "benchmark"],
    default="normal",
    help="Ejecuta el flujo normal, solo el EDA, o el modo comparativo Benchmark."
)
parser.add_argument(
    "--freq",
    default=None,
    help="Frecuencia de resampleo para EDA (ej. D, H, 15T). Si se pasa, sobreescribe config.yaml."
)
args, _ = parser.parse_known_args()
if args.freq:
    config.setdefault("eda", {})["frecuencia_resampleo"] = args.freq

# =========================
# 1.1) RESOLVER MODO EFECTIVO Y DESVIAR A BENCHMARK SI APLICA
# =========================
yaml_modo = str(config.get("modo", "")).strip().lower()
modo_efectivo = yaml_modo if yaml_modo in {"normal", "eda", "benchmark"} else args.modo
# El CLI tiene prioridad si pasas --modo explícitamente
if args.modo in {"normal", "eda", "benchmark"}:
    modo_efectivo = args.modo
if modo_efectivo == "benchmark":
    print("🔎 Modo 'benchmark' ACTIVADO (por YAML o CLI).")
    try:
        # Import perezoso: solo si realmente corres benchmark
        from modelos.evaluacion_modelos import ejecutar_benchmark
        # 👉 IMPORTA LA FUNCIÓN PARA ESCRIBIR EN TU REPORTE PRINCIPAL
        from reportes.reportes_excel import write_benchmark_sheet
    except Exception as e:
        print("⚠️ Falta implementar el runner de benchmark (paso 3).")
        print("   Debes definir 'ejecutar_benchmark(cfg)' en modelos/evaluacion_modelos.py")
        print(f"   Detalle del import: {e}")
        sys.exit(1)

    try:
        # Ejecuta el benchmark → df con comparativa de modelos
        df_resultados = ejecutar_benchmark(config)

        # (1) Excel dedicado a benchmark (aparte)
        from pathlib import Path
        ruta_bench = config.get("ruta_reporte_benchmark", "outputs/benchmark_resultados.xlsx")
        p = Path(ruta_bench)
        p.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(p, engine="openpyxl", mode="w") as writer:
            df_resultados.to_excel(writer, sheet_name="Comparación_Modelos", index=False)
        print(f"✅ Benchmark finalizado. Resultado escrito en: {p}")

        # (2) TAMBIÉN lo agregamos como hoja al reporte principal
        ruta_reporte_principal = config.get("ruta_reporte", "outputs/reporte_inversion.xlsx")
        write_benchmark_sheet(ruta_reporte_principal, df_resultados, sheet_name="Comparación_Modelos")
        print(f"📄 Hoja 'Comparación_Modelos' agregada a: {ruta_reporte_principal}")

        # Importante: salimos para no correr el flujo normal
        sys.exit(0)

    except NotImplementedError as nie:
        print("⚠️ 'ejecutar_benchmark(cfg)' lanzó NotImplementedError (pendiente conectar loader/modelos).")
        print(f"   Detalle: {nie}")
        sys.exit(1)
    except Exception as e:
        print("❌ Error al ejecutar/escribir el benchmark.")
        print(f"   Detalle: {e}")
        sys.exit(1)


# =========================
# 2) CONEXIÓN MT5
# =========================
# Recomendado: mover credenciales a variables de entorno o a config.yaml (sección 'mt5')
mt5_cfg = config.get("mt5", {})
login = int(os.getenv("MT5_LOGIN", mt5_cfg.get("login", 68238343)))
clave = os.getenv("MT5_PASSWORD", mt5_cfg.get("password", "Colombia123*"))
servidor = os.getenv("MT5_SERVER", mt5_cfg.get("server", "RoboForex-Pro"))
path = os.getenv("MT5_PATH", mt5_cfg.get("path", r"C:\Program Files\RoboForex - MetaTrader 5\terminal64.exe"))

if not mt5.initialize(login=login, password=clave, server=servidor, path=path):
    print("❌ Error al conectar a MT5:", mt5.last_error())
    sys.exit(1)
print("✅ Conexión establecida con MetaTrader 5")

try:
    # === EDA de dos activos con --modo eda ===
    if args.modo == "eda":
        # Import perezoso de EDA SOLO si se usa
        from procesamiento.eda_crispdm import ejecutar_eda

        # EURUSD (usa tu 'simbolo' actual o 'simbolo_eurusd' si está en config)
        simbolo_eur = config.get("simbolo_eurusd", simbolo)
        df_eur = obtener_df_desde_mt5(simbolo_eur, timeframe, cantidad)

        # SPY: intenta MT5 si tienes el símbolo; si no, usa CSV (ruta en config["spy_csv"])
        df_spy = None
        simbolo_spy = config.get("simbolo_spy")
        ruta_spy_csv = config.get("spy_csv")

        if simbolo_spy:
            try:
                df_spy = obtener_df_desde_mt5(simbolo_spy, timeframe, cantidad)
            except Exception as e:
                print(f"⚠️ No se pudo obtener SPY desde MT5: {e}")
        if df_spy is None and ruta_spy_csv:
            df_spy = pd.read_csv(ruta_spy_csv)
            if "timestamp" not in df_spy.columns:
                for c in ["Date", "Datetime", "date", "datetime"]:
                    if c in df_spy.columns:
                        df_spy = df_spy.rename(columns={c: "timestamp"})
                        break
            if "Close" not in df_spy.columns and "close" in df_spy.columns:
                df_spy = df_spy.rename(columns={"close": "Close"})

        # Ejecuta EDA (CRISP-DM: Understanding & Preparation)
        ejecutar_eda(df_eurusd=df_eur, df_spy=df_spy, cfg=config)
        print("✅ EDA completado (ver outputs/eda).")
        sys.exit(0)

    # Instancia de utilidades MT5 (import perezoso)
    from conexion.easy_Trading import Basic_funcs
    BF = Basic_funcs(login, clave, servidor, path)

    # Tamaño de pip robusto
    info = mt5.symbol_info(simbolo)
    if info is None:
        raise RuntimeError(f"No se pudo obtener symbol_info de {simbolo}")
    point = info.point
    if pip_size_cfg is not None:
        pip = float(pip_size_cfg)
    elif info.digits in (3, 5):
        pip = point * 10  # FX con 3 ó 5 dígitos → 1 pip = 10 * point
    else:
        pip = point       # Acciones/índices u otros
    print(f"ℹ️ Símbolo={simbolo}, digits={info.digits}, point={point}, pip={pip}")

    # =========================
    # 3) EXTRACCIÓN DE DATOS
    # =========================
    print("⏳ Extrayendo datos de MT5...")
    df = BF.get_data_for_bt(timeframe, simbolo, cantidad)
    print("Última fecha en datos extraídos:", df.index.max())

    print("📈 Calculando indicadores técnicos...")
    # Import perezoso de features
    from procesamiento.features import aplicar_todos_los_indicadores
    df_indicadores = aplicar_todos_los_indicadores(df)

    # =========================
    # 3.1) EDA CRISP-DM (opcional por YAML)
    # =========================
    try:
        if config.get("eda", {}).get("habilitar", False):
            # Import aquí también, porque este branch puede ejecutarse en modo normal
            from procesamiento.eda_crispdm import ejecutar_eda

            # Pasamos el DF con un 'timestamp' explícito para que el EDA sea robusto
            df_eur_eda = (
                df_indicadores
                .reset_index()
                .rename(columns={(df_indicadores.index.name or "index"): "timestamp"})
            )
            # Si no tienes SPY en este flujo, pasa None
            ejecutar_eda(df_eurusd=df_eur_eda, df_spy=None, cfg=config)
    except Exception as e:
        print(f"⚠️ EDA opcional no se ejecutó: {e}")

    # =========================
    # 4) ENTRENAR / PREDECIR
    # =========================
    if modelo_str == "prophet":
        print("🤖 Entrenando modelo Prophet...")
        # Import perezoso de Prophet
        from modelos.prophet_model import entrenar_modelo_prophet, predecir_precio

        modelo = entrenar_modelo_prophet(df_indicadores)

        print("🔮 Generando predicción futura...")
        predicciones = predecir_precio(modelo, pasos=pasos_pred, frecuencia=frecuencia_pred)
        print(predicciones)
    else:
        raise ValueError(f"Modelo '{modelo_str}' no implementado aún.")

    # =========================
    # 5) SEÑAL + ASIGNACIÓN DE CAPITAL
    # =========================
    # Imports perezosos de agentes
    from agentes.agente_analisis import generar_senal_operativa
    from agentes.agente_portafolio import asignar_capital

    senal = generar_senal_operativa(predicciones, umbral=umbral_senal)
    print(f"📢 Señal generada: {senal}")

    balance, _, _, _ = BF.info_account()
    capital = asignar_capital(balance, senal)
    print(f"💰 Capital asignado según la señal: ${capital:.2f}")

    # Simulación (para registro en reporte)
    precio_actual = (df_indicadores.get('Close', df_indicadores.get('close'))).iloc[-1]

    from agentes.agente_ejecucion import ejecutar_operacion, generar_reporte_excel
    operacion = ejecutar_operacion(simbolo, senal, capital, precio_actual)
    print(f"🧾 Operación simulada: {operacion}")

    # =========================
    # 6) REPORTE BASE (asegura archivo)
    # =========================
    ruta_excel = ruta_reporte
    generar_reporte_excel(predicciones, senal, capital, operacion, umbral=umbral_senal)

    # =========================
    # 7) EVALUACIÓN (métricas + horizonte) MODULAR
    # =========================
    try:
        # Imports perezosos para métricas y escritura
        from modelos.evaluacion_modelos import compute_metrics_prophet
        from reportes.reportes_excel import write_metrics_sheet, append_history

        metrics = compute_metrics_prophet(
            df_indicadores=df_indicadores,
            predicciones_live=predicciones,
            pasos_pred=pasos_pred,
            frecuencia_pred=frecuencia_pred,
            simbolo=simbolo,
            timeframe_str=timeframe_str,
            modelo_str=modelo_str,
            entrenar_fn=lambda df_: (__import__("modelos.prophet_model", fromlist=["entrenar_modelo_prophet"]).entrenar_modelo_prophet)(df_),
            predecir_fn=lambda m_, pasos, frecuencia: (__import__("modelos.prophet_model", fromlist=["predecir_precio"]).predecir_precio)(m_, pasos=pasos, frecuencia=frecuencia)
        )

        print(
            f"📏 MAE={metrics['MAE']:.6f}  RMSE={metrics['RMSE']:.6f}  "
            f"MAPE={metrics['MAPE_%']:.2f}%  R²={metrics['R2']:.4f}  "
            f"Sortino={metrics['Sortino']:.4f}  AccDir={metrics['Accuracy_direccional']:.2%}  "
            f"Horiz≈{metrics['Horizonte_horas_totales']:.1f}h"
        )

        # Guarda hoja de métricas de la corrida actual y actualiza histórico
        write_metrics_sheet(ruta_excel, metrics, sheet_name='Métricas Modelo')
        append_history(ruta_excel, metrics, hist_sheet='Historico Métricas')

    except Exception as e:
        print(f"⚠️ No se pudieron calcular/guardar métricas: {e}")

    # =========================
    # 8) EJECUCIÓN REAL (si aplica)
    # =========================
    if senal in ['comprar', 'vender'] and capital > 0:
        tipo_mt5 = mt5.ORDER_TYPE_BUY if senal == 'comprar' else mt5.ORDER_TYPE_SELL

        # Tamaño de posición por riesgo fijo del YAML
        distancia_sl_precio = stop_loss_pips * pip  # SL en precio (de pips a precio)
        volumen = BF.calculate_position_size(
            simbolo,
            tradeinfo=distancia_sl_precio,
            per_to_risk=riesgo_por_trade
        )

        # Volumen mínimo desde YAML
        if volumen < volumen_minimo:
            print(f"⚠️ Volumen calculado ({volumen}) < mínimo ({volumen_minimo}); ajustado.")
            volumen = volumen_minimo

        # SL / TP por señal (usando pips del YAML)
        if senal == 'comprar':
            sl = round(precio_actual - stop_loss_pips * pip, info.digits)
            tp = round(precio_actual + take_profit_pips * pip, info.digits)
        else:  # vender
            sl = round(precio_actual + stop_loss_pips * pip, info.digits)
            tp = round(precio_actual - take_profit_pips * pip, info.digits)

        print(
            f"📌 SL: {sl} | TP: {tp} | Vol: {volumen}  "
            f"(riesgo {riesgo_por_trade*100:.1f}% ; SL {stop_loss_pips} pips / TP {take_profit_pips} pips)"
        )

        BF.open_operations(
            par=simbolo,
            volumen=volumen,
            tipo_operacion=tipo_mt5,
            nombre_bot=f'Sistema Prophet - {senal.upper()}',
            sl=sl,
            tp=tp
        )
        print(f"🚀 Orden enviada a MT5: {senal.upper()} {simbolo} con {volumen} lotes")

    else:
        print("❎ No se envió operación real (señal fue 'mantener' o capital = 0)")

finally:
    # =========================
    # 9) CIERRE
    # =========================
    mt5.shutdown()
    print("🛑 Conexión cerrada")
