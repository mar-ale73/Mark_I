# modelos/evaluacion_modelos.py
import warnings
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import MetaTrader5 as mt5

# -------------------------------------------------------------------
# 1) Deja tu función base tal cual (flujo NORMAL)
# -------------------------------------------------------------------
def compute_metrics_prophet(
    df_indicadores: pd.DataFrame,
    predicciones_live: pd.DataFrame,
    pasos_pred: int,
    frecuencia_pred: str,
    simbolo: str,
    timeframe_str: str,
    modelo_str: str,
    entrenar_fn,      # callable(df_train) -> modelo
    predecir_fn       # callable(modelo, pasos, frecuencia) -> df_pred
) -> dict:
    """
    Backtest simple: entrena con todo menos los últimos 'pasos_pred' y predice esos pasos
    para comparar vs valores reales. Calcula métricas + horizonte (desde predicciones live).
    """
    # Split
    df_train = df_indicadores.iloc[:-pasos_pred].copy()
    df_test  = df_indicadores.iloc[-pasos_pred:].copy()

    # Entrenar & predecir backtest
    modelo_bt = entrenar_fn(df_train)
    preds_bt  = predecir_fn(modelo_bt, pasos=pasos_pred, frecuencia=frecuencia_pred)

    # Alinear tamaños
    y_true = df_test['Close'].values
    y_pred = preds_bt['precio_estimado'].values[:len(y_true)]

    # Errores
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))) * 100)
    r2   = r2_score(y_true, y_pred)

    # Sortino (retornos de predicción vs real)
    returns = (y_pred - y_true) / np.where(y_true == 0, np.nan, y_true)
    returns = returns[~np.isnan(returns)]
    downside = returns[returns < 0]
    downside_std = float(np.std(downside)) if downside.size > 0 else np.nan
    risk_free = 0.0
    sortino = float((np.mean(returns) - risk_free) / downside_std) if downside_std not in [0.0, np.nan] else np.nan

    # Accuracy direccional
    dir_real = np.sign(np.diff(y_true))
    dir_pred = np.sign(np.diff(y_pred))
    m = min(len(dir_real), len(dir_pred))
    aciertos = int(np.sum(dir_real[:m] == dir_pred[:m]))
    total    = int(m)
    accuracy_dir = float(aciertos / total) if total > 0 else np.nan

    # Horizonte (desde predicciones live)
    try:
        horizonte = predicciones_live['timestamp_prediccion'].max() - predicciones_live['timestamp_prediccion'].min()
        horizonte_dias = int(horizonte.days)
        horizonte_horas_totales = float(horizonte.total_seconds() / 3600)
    except Exception:
        horizonte_dias = 0
        horizonte_horas_totales = 0.0

    return {
        'Fecha': pd.Timestamp.now(),
        'Simbolo': simbolo,
        'Timeframe': timeframe_str,
        'Modelo': modelo_str,
        'Pasos_pred': int(pasos_pred),
        'MAE': float(mae),
        'RMSE': float(rmse),
        'MAPE_%': float(mape),
        'R2': float(r2),
        'Sortino': float(sortino) if not np.isnan(sortino) else np.nan,
        'Accuracy_direccional': float(accuracy_dir) if not np.isnan(accuracy_dir) else np.nan,
        'Horizonte_dias': int(horizonte_dias),
        'Horizonte_horas_totales': float(horizonte_horas_totales)
    }


# -------------------------------------------------------------------
# 2) Utilidades Benchmark
# -------------------------------------------------------------------
_TIMEFRAMES = {
    'M1': mt5.TIMEFRAME_M1,
    'M5': mt5.TIMEFRAME_M5,
    'M15': mt5.TIMEFRAME_M15,
    'H1': mt5.TIMEFRAME_H1,
    'D1': mt5.TIMEFRAME_D1,
}

def _freq_from_timeframe_str(timeframe_str: str) -> str:
    tf = timeframe_str.upper()
    return {"M1": "1min", "M5": "5min", "M15": "15min", "H1": "h", "D1": "D"}.get(tf, "h")



def _init_mt5(mt5_cfg: Dict[str, Any]) -> None:
    login = int(mt5_cfg.get("login", 0))
    password = mt5_cfg.get("password", "")
    server = mt5_cfg.get("server", "")
    path = mt5_cfg.get("path", None)
    ok = mt5.initialize(login=login, password=password, server=server, path=path)
    if not ok:
        raise RuntimeError(f"Error al conectar a MT5: {mt5.last_error()}")

def _obtener_df_desde_mt5(symbol: str, timeframe, n_barras: int) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, int(n_barras))
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No se obtuvieron datos de {symbol} desde MT5.")
    df = pd.DataFrame(rates)
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                            "close": "Close", "tick_volume": "Volume"})
    df = df[["timestamp", "Open", "High", "Low", "Close", "Volume"]].set_index("timestamp")
    # 👇 añade esto:
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df


def _get_features_fn():
    # 1) ruta histórica del proyecto
    try:
        from procesamiento.features import aplicar_todos_los_indicadores as f
        return f
    except Exception as e1:
        warnings.warn(f"No se pudo importar procesamiento.features.aplicar_todos_los_indicadores ({e1}). Intentando 'features'…")
        # 2) alternativa por si el módulo vive en la raíz
        try:
            from features import aplicar_todos_los_indicadores as f
            return f
        except Exception as e2:
            warnings.warn(
                f"Tampoco se pudo importar features.aplicar_todos_los_indicadores ({e2}). "
                f"Se usará fallback (identidad). Verifica la ruta/nombre del módulo."
            )
            def _identity(df: pd.DataFrame) -> pd.DataFrame:
                out = df.copy()
                if "Close" not in out.columns and "close" in out.columns:
                    out = out.rename(columns={"close": "Close"})
                return out
            return _identity


def _profit_factor(returns: np.ndarray) -> float:
    gains = returns[returns > 0].sum()
    losses = returns[returns < 0].sum()
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return float(gains / abs(losses))

def _max_drawdown(equity: np.ndarray) -> float:
    peak = -np.inf
    max_dd = 0.0
    for x in equity:
        peak = max(peak, x)
        dd = (x / peak) - 1.0
        max_dd = min(max_dd, dd)
    return float(max_dd * 100.0)

def _apply_costs(gross_returns: np.ndarray, roundtrip_bps: float) -> np.ndarray:
    cost_frac = roundtrip_bps / 1e4
    return gross_returns - cost_frac

def _walk_forward_metrics(
    df_ind: pd.DataFrame,
    entrenar_fn,
    predecir_fn,
    horizonte: int,
    frecuencia_pred: str,
    costos_bps: float,
    slippage_ticks: float
) -> Dict[str, Any]:
    """
    Walk-forward causal:
      - En cada i, entrena con datos [:i] y predice a i+hor.
      - Señal = sign(pred_price_{i+hor} - price_i)
      - Retorno bruto = señal * (real_{i+hor}/price_i - 1)
      - Costos: roundtrip_bps = 2*(costos_bps + slippage_ticks)
        (tratamos slippage_ticks como bps; luego lo refinamos).
    """
    closes = df_ind["Close"].values.astype(float)
    n = len(closes)
    if n <= horizonte + 5:
        raise ValueError("Serie demasiado corta para el horizonte solicitado.")

    min_train = max(200, 5 * horizonte)
    min_train = min(min_train, n - horizonte - 1)

    y_true_list, y_pred_list = [], []
    gross_rets, dirs_true, dirs_pred = [], [], []

    for i in range(min_train, n - horizonte):
        df_train = df_ind.iloc[:i].copy()
        model = entrenar_fn(df_train)
        pred_df = predecir_fn(model, pasos=int(horizonte), frecuencia=frecuencia_pred)
        try:
            pred_price = float(pred_df["precio_estimado"].iloc[-1])
        except Exception:
            if "yhat" in pred_df.columns:
                pred_price = float(pred_df["yhat"].iloc[-1])
            else:
                raise

        price_t   = float(closes[i-1])
        price_fut = float(closes[i-1 + horizonte])

        pred_dir = int(np.sign(pred_price - price_t))
        true_dir = int(np.sign(price_fut - price_t))
        gross_ret = pred_dir * ((price_fut / price_t) - 1.0)

        y_true_list.append(price_fut)
        y_pred_list.append(pred_price)
        gross_rets.append(gross_ret)
        dirs_true.append(true_dir)
        dirs_pred.append(pred_dir)

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    gross_rets = np.array(gross_rets)
    dirs_true = np.array(dirs_true)
    dirs_pred = np.array(dirs_pred)

    # Predictivas
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))) * 100)
    r2 = r2_score(y_true, y_pred)

    # Decisionales pos-costos
    roundtrip_bps = 2.0 * (float(costos_bps) + float(slippage_ticks))
    net_rets = _apply_costs(gross_rets, roundtrip_bps)

    m = min(len(dirs_true), len(dirs_pred))
    acc_dir = float((dirs_true[:m] == dirs_pred[:m]).mean()) if m > 0 else np.nan

    net_down = net_rets[net_rets < 0.0]
    down_std = float(np.std(net_down)) if net_down.size > 0 else np.nan
    sortino = float(np.mean(net_rets) / down_std) if down_std not in [0.0, np.nan] else np.nan

    pf = _profit_factor(net_rets)
    equity = np.cumprod(1.0 + net_rets)
    maxdd_pct = _max_drawdown(equity)
    ret_acum_pct = float((equity[-1] - 1.0) * 100.0) if len(equity) > 0 else np.nan

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE_%": float(mape),
        "R2": float(r2),
        "Accuracy_direccional": float(acc_dir) if not np.isnan(acc_dir) else np.nan,
        "Sortino": float(sortino) if not np.isnan(sortino) else np.nan,
        "Profit_Factor": float(pf) if pf == pf else np.nan,
        "MaxDD_%": float(maxdd_pct),
        "Ret_Acumulado_%": float(ret_acum_pct),
        "Trades": int(len(net_rets)),
    }


# -------------------------------------------------------------------
# 3) Runner de BENCHMARK (lo llama app/main.py en modo benchmark)
# -------------------------------------------------------------------
def ejecutar_benchmark(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Corre benchmark por modelos × horizontes (walk-forward causal).
    Por ahora corre 'prophet'; el resto se deja para próxima iteración.
    """
    simbolo: str = cfg.get("simbolo", "EURUSD")
    timeframe_str: str = cfg.get("timeframe", "M5")
    cantidad: int = int(cfg.get("cantidad_datos", 2000))

    modelos: List[str] = list(cfg.get("modelos", ["prophet"]))
    horizontes: List[int] = list(cfg.get("horizontes", [1, 3, 5]))
    costos_bps: float = float(cfg.get("costos_bps", 0.0))
    slippage_ticks: float = float(cfg.get("slippage_ticks", 0.0))

    frecuencia_pred_def = _freq_from_timeframe_str(timeframe_str)

    _init_mt5(cfg.get("mt5", {}))
    try:
        if timeframe_str not in _TIMEFRAMES:
            raise ValueError(f"Timeframe '{timeframe_str}' no soportado. Usa uno de {list(_TIMEFRAMES.keys())}.")
        timeframe = _TIMEFRAMES[timeframe_str]

        df = _obtener_df_desde_mt5(simbolo, timeframe, cantidad)

        # Features (robusto: usa la función si existe; si no, identidad)
        aplicar_feats = _get_features_fn()
        df_indicadores = aplicar_feats(df)

        resultados: List[Dict[str, Any]] = []

        for modelo in modelos:
            modelo_lower = str(modelo).lower().strip()

            if modelo_lower == "prophet":
                from modelos.prophet_model import entrenar_modelo_prophet, predecir_precio
                entrenar_fn = entrenar_modelo_prophet
                predecir_fn = predecir_precio
            else:
                warnings.warn(f"Modelo '{modelo}' aún no implementado en benchmark; se omite.")
                continue

            for h in horizontes:
                try:
                    m = _walk_forward_metrics(
                        df_ind=df_indicadores,
                        entrenar_fn=entrenar_fn,
                        predecir_fn=predecir_fn,
                        horizonte=int(h),
                        frecuencia_pred=frecuencia_pred_def,
                        costos_bps=costos_bps,
                        slippage_ticks=slippage_ticks
                    )
                    fila = {
                        "Fecha": pd.Timestamp.now(),
                        "Activo": simbolo,
                        "Timeframe": timeframe_str,
                        "Modelo": modelo_lower,
                        "Horizonte": int(h),
                        "Costo_bps": float(costos_bps),
                        "Slippage_ticks(as_bps)": float(slippage_ticks),
                        **m
                    }
                    resultados.append(fila)
                except Exception as e:
                    warnings.warn(f"Fallo modelo={modelo_lower}, h={h}: {e}")
                    resultados.append({
                        "Fecha": pd.Timestamp.now(),
                        "Activo": simbolo,
                        "Timeframe": timeframe_str,
                        "Modelo": modelo_lower,
                        "Horizonte": int(h),
                        "Costo_bps": float(costos_bps),
                        "Slippage_ticks(as_bps)": float(slippage_ticks),
                        "Estado": f"error: {e}"
                    })

        if not resultados:
            raise NotImplementedError("No hay modelos implementados para benchmark (activa Prophet por ahora).")

        df_res = pd.DataFrame(resultados)
        cols_first = ["Fecha","Activo","Timeframe","Modelo","Horizonte","Costo_bps","Slippage_ticks(as_bps)"]
        other_cols = [c for c in df_res.columns if c not in cols_first]
        df_res = df_res[cols_first + other_cols]
        return df_res

    finally:
        mt5.shutdown()
