# modelos/evaluacion_modelos.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    r2   = r2_score(y_true, y_pred)

    # Sortino (retornos de predicción vs real)
    returns = (y_pred - y_true) / y_true
    downside = returns[returns < 0]
    downside_std = float(np.std(downside)) if downside.size > 0 else np.nan
    risk_free = 0.0
    sortino = float((np.mean(returns) - risk_free) / downside_std) if downside_std not in [0.0, np.nan] else np.nan

    # Accuracy direccional
    dir_real = np.sign(np.diff(y_true))
    dir_pred = np.sign(np.diff(y_pred))
    aciertos = int(np.sum(dir_real == dir_pred))
    total    = int(len(dir_real))
    accuracy_dir = float(aciertos / total) if total > 0 else np.nan

    # Horizonte (desde las predicciones live del bot)
    horizonte = predicciones_live['timestamp_prediccion'].max() - predicciones_live['timestamp_prediccion'].min()
    horizonte_dias = int(horizonte.days)
    horizonte_horas_totales = float(horizonte.total_seconds() / 3600)

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
        'Horizonte_dias': horizonte_dias,
        'Horizonte_horas_totales': horizonte_horas_totales
    }
