# mlp_adapter.py
# Adapter MLP compatible con evaluacion_modelos.compute_metrics_prophet
# Requisitos: scikit-learn, pandas, numpy

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# =========================
# Helpers internos
# =========================
def _ensure_time_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    if 'time' in df.columns:
        idx = pd.to_datetime(df['time'])
    elif isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    else:
        raise ValueError("Se requiere columna 'time' o índice datetime en el DataFrame.")
    if not idx.is_monotonic_increasing:
        df2 = df.copy()
        df2.index = idx
        df2 = df2.sort_index()
        return df2.index
    return idx


def _validate_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    if 'Close' not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'Close'.")
    idx = _ensure_time_index(df)
    base = df.copy()
    base.index = idx
    base = base.sort_index()
    base = base[['Close']].astype(float).dropna()
    if len(base) < 40:
        raise ValueError("Muy pocos datos para ajustar un MLP (mín ~40 observaciones).")
    return base


def _make_supervised(y: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    X, Y = [], []
    for i in range(lookback, len(y)):
        X.append(y[i - lookback:i])
        Y.append(y[i])
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


def _z_from_alpha(alpha: float) -> float:
    # z aproximado bilateral: 90%->1.64, 95%->1.96, 80%->1.28
    if alpha <= 0.05 + 1e-9:
        return 1.96
    if alpha <= 0.10 + 1e-9:
        return 1.64
    if alpha <= 0.20 + 1e-9:
        return 1.28
    return 1.64


def _build_future_index(last_time: pd.Timestamp, pasos: int, frecuencia: Optional[str], fallback: str) -> pd.DatetimeIndex:
    freq = frecuencia or fallback or 'D'
    return pd.date_range(start=last_time, periods=pasos + 1, freq=freq)[1:]


# =========================
# API pública del adapter
# =========================
def entrenar_modelo_mlp(
    df: pd.DataFrame,
    modo: str = 'nivel',                 # 'nivel' | 'retornos'
    lookback: int = 20,
    scaler_type: str = 'minmax',         # 'minmax' | 'standard'
    hidden_layer_sizes: Tuple[int, ...] = (64, 32),
    activation: str = 'relu',            # 'relu' | 'tanh' | 'logistic' | 'identity'
    solver: str = 'adam',
    alpha_reg: float = 1e-4,             # L2
    max_iter: int = 500,
    early_stopping: bool = True,
    validation_fraction: float = 0.1,
    n_iter_no_change: int = 10,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Entrena un MLP univariante para precio ('nivel') o retornos ('retornos').
    Devuelve dict con el modelo y metadatos para predicción recursiva multi-paso.
    """
    if modo not in ('nivel', 'retornos'):
        raise ValueError("Parametro 'modo' debe ser 'nivel' o 'retornos'.")

    base = _validate_and_prepare(df)
    idx = base.index

    if modo == 'retornos':
        serie = base['Close'].pct_change().dropna().to_numpy(dtype=np.float32)
        ultimo_close = float(base['Close'].iloc[-1])
        if len(serie) <= lookback + 1:
            raise ValueError("No hay suficientes observaciones de retornos para el lookback indicado.")
    else:
        serie = base['Close'].to_numpy(dtype=np.float32)
        ultimo_close = None
        if len(serie) <= lookback + 1:
            raise ValueError("No hay suficientes observaciones de nivel de precio para el lookback indicado.")

    scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    ys = scaler.fit_transform(serie.reshape(-1, 1)).ravel().astype(np.float32)

    X, Y = _make_supervised(ys, lookback)

    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha_reg,
        max_iter=max_iter,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        n_iter_no_change=n_iter_no_change,
        random_state=random_state,
        shuffle=True,
        learning_rate='adaptive'
    )
    mlp.fit(X, Y)

    # Residuo (en unidad original) para bandas aprox.
    yhat_scaled = mlp.predict(X).ravel()
    yhat = scaler.inverse_transform(yhat_scaled.reshape(-1, 1)).ravel()
    y_true = scaler.inverse_transform(Y.reshape(-1, 1)).ravel()
    resid_std = float(np.std(y_true - yhat))

    tail = ys[-lookback:].copy()
    last_time = idx.max()
    freq_inferida = pd.infer_freq(idx)

    return {
        'mlp': mlp,
        'scaler': scaler,
        'tail': tail,                 # última ventana en espacio escalado
        'lookback': lookback,
        'last_time': last_time,
        'freq': freq_inferida or 'D',
        'modo': modo,
        'ultimo_close': ultimo_close, # solo para reconstruir en 'retornos'
        'resid_std': resid_std,
        'scaler_type': scaler_type,
    }


def predecir_precio_mlp(
    modelo: Dict[str, Any],
    pasos: int = 3,
    frecuencia: Optional[str] = None,
    alpha: float = 0.10,  # 90% (z ~ 1.64)
) -> pd.DataFrame:
    """
    Predice 'pasos' periodos hacia adelante y devuelve:
    ['timestamp_prediccion','precio_estimado','min_esperado','max_esperado'].
    - 'retornos': interpreta predicciones como retornos y reconstruye precio por acumulación.
    - 'nivel'   : interpreta predicciones como precio.
    Bandas: aproximación ± z * std(residuo de entrenamiento) en unidad original.
    """
    if pasos < 1:
        raise ValueError("El numero de 'pasos' debe ser >= 1.")

    mlp = modelo['mlp']
    scaler = modelo['scaler']
    seq = modelo['tail'].copy()
    lookback = modelo['lookback']
    last_time = modelo['last_time']
    fallback_freq = modelo.get('freq', 'D')
    modo = modelo.get('modo', 'nivel')
    ultimo_close = modelo.get('ultimo_close', None)
    resid_std = float(modelo.get('resid_std', 0.0))
    z = _z_from_alpha(alpha)

    # Predicción recursiva (multi-step) en espacio escalado
    preds_scaled = []
    for _ in range(pasos):
        x = seq[-lookback:].reshape(1, -1)
        yhat = float(mlp.predict(x)[0])
        preds_scaled.append(yhat)
        seq = np.append(seq, yhat)

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).ravel()
    future_index = _build_future_index(last_time, pasos, frecuencia, fallback_freq)

    if modo == 'retornos':
        if ultimo_close is None:
            raise ValueError("No se encontró 'ultimo_close' para reconstruir precio en modo 'retornos'.")
        price_path = ultimo_close * np.cumprod(1.0 + preds)

        low_returns = preds - z * resid_std
        up_returns  = preds + z * resid_std
        price_low = ultimo_close * np.cumprod(1.0 + low_returns)
        price_up  = ultimo_close * np.cumprod(1.0 + up_returns)

        out = pd.DataFrame({
            'timestamp_prediccion': future_index,
            'precio_estimado': price_path.astype(float),
            'min_esperado': price_low.astype(float),
            'max_esperado': price_up.astype(float),
        })

    else:  # modo == 'nivel'
        price_est = preds
        price_low = preds - z * resid_std
        price_up  = preds + z * resid_std

        out = pd.DataFrame({
            'timestamp_prediccion': future_index,
            'precio_estimado': price_est.astype(float),
            'min_esperado': price_low.astype(float),
            'max_esperado': price_up.astype(float),
        })

    return out
