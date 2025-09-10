# lstm_adapter.py
# Adapter LSTM compatible con evaluacion_modelos.compute_metrics_prophet
# Autor: (tu equipo)
# Requisitos: tensorflow>=2, scikit-learn, pandas, numpy

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple

# Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Escalado
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# -------------------------------
# Helpers internos
# -------------------------------
def _ensure_time_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    if 'time' in df.columns:
        idx = pd.to_datetime(df['time'])
    elif isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    else:
        raise ValueError("Se requiere columna 'time' o índice datetime en el DataFrame.")
    if idx.is_monotonic_increasing is False:
        df_sorted = df.copy()
        df_sorted.index = idx
        df_sorted = df_sorted.sort_index()
        return df_sorted.index
    return idx


def _validate_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    if 'Close' not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'Close'.")
    idx = _ensure_time_index(df)
    base = df.copy()
    base.index = idx
    base = base.sort_index()
    base = base[['Close']].astype(float)
    base = base.dropna()
    if len(base) < 60:
        raise ValueError("Muy pocos datos para ajustar un LSTM (min ~60 observaciones).")
    return base


def _make_supervised(y: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crea ventanas supervisadas para series univariadas.
    Retorna:
      X: [n_samples, lookback, 1]
      Y: [n_samples,]
    """
    X, Y = [], []
    for i in range(lookback, len(y)):
        X.append(y[i - lookback:i])
        Y.append(y[i])
    X = np.array(X, dtype=np.float32)[:, :, np.newaxis]
    Y = np.array(Y, dtype=np.float32)
    return X, Y


def _z_from_alpha(alpha: float) -> float:
    """
    Aproximación de z-score bilateral para (1 - alpha) de confianza.
    Casos comunes: 90% -> 1.64, 95% -> 1.96, 80% -> 1.28
    """
    if alpha <= 0.05 + 1e-9:
        return 1.96
    if alpha <= 0.10 + 1e-9:
        return 1.64
    if alpha <= 0.20 + 1e-9:
        return 1.28
    return 1.64  # por defecto 90%


# -------------------------------
# API pública del adapter
# -------------------------------
def entrenar_modelo_lstm(
    df: pd.DataFrame,
    modo: str = 'nivel',           # 'nivel' | 'retornos'
    lookback: int = 40,
    epochs: int = 40,
    batch_size: int = 32,
    validation_split: float = 0.1,
    patience: int = 6,
    units: int = 64,
    dense_units: int = 32,
    dropout: float = 0.0,
    scaler_type: str = 'minmax',   # 'minmax' | 'standard'
) -> Dict[str, Any]:
    """
    Entrena un LSTM univariante sobre precio ('nivel') o retornos ('retornos').
    Devuelve un dict con el modelo y metadatos para predicción.
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

    # Escalado
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    ys = scaler.fit_transform(serie.reshape(-1, 1)).ravel().astype(np.float32)

    # Ventanas supervisadas
    X, Y = _make_supervised(ys, lookback)

    # Modelo
    model = Sequential()
    model.add(LSTM(units, input_shape=(lookback, 1)))
    if dropout > 0:
        model.add(Dropout(dropout))
    if dense_units and dense_units > 0:
        model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Early stopping
    es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0)

    # Entrenamiento
    model.fit(
        X, Y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[es],
        verbose=0
    )

    # Estimación de residuo (en unidad original) para bandas aprox.
    yhat_scaled = model.predict(X, verbose=0).ravel()
    yhat = scaler.inverse_transform(yhat_scaled.reshape(-1, 1)).ravel()
    y_true = scaler.inverse_transform(Y.reshape(-1, 1)).ravel()
    resid_std = float(np.std(y_true - yhat))

    # Cola (última ventana) para pronóstico recursivo
    tail = ys[-lookback:].copy()

    # Frecuencia e índice
    freq_inferida = pd.infer_freq(idx)
    last_time = idx.max()

    return {
        'model': model,
        'scaler': scaler,
        'tail': tail,                  # ventana en espacio escalado
        'lookback': lookback,
        'last_time': last_time,
        'freq': freq_inferida or 'D',
        'modo': modo,
        'ultimo_close': ultimo_close,  # usado si modo='retornos'
        'resid_std': resid_std,        # para bandas aproximadas
        'scaler_type': scaler_type,
    }


def predecir_precio_lstm(
    modelo: Dict[str, Any],
    pasos: int = 3,
    frecuencia: Optional[str] = None,
    alpha: float = 0.10,  # 90% CI aprox
) -> pd.DataFrame:
    """
    Predice 'pasos' periodos hacia adelante y devuelve el DataFrame estándar:
      ['timestamp_prediccion','precio_estimado','min_esperado','max_esperado'].
    - Si 'modo' fue 'retornos': interpreta predicciones como retornos y reconstruye precio por acumulación.
    - Si 'modo' fue 'nivel'  : interpreta predicciones como precio.
    Bandas: aproximación ± z * std(residuo de entrenamiento) en unidad original.
    """
    if pasos < 1:
        raise ValueError("El numero de 'pasos' debe ser >= 1.")

    model = modelo['model']
    scaler = modelo['scaler']
    seq = modelo['tail'].copy()        # en espacio escalado
    lookback = modelo['lookback']
    last_time = modelo['last_time']
    fallback_freq = modelo.get('freq', 'D')
    modo = modelo.get('modo', 'nivel')
    ultimo_close = modelo.get('ultimo_close', None)
    resid_std = float(modelo.get('resid_std', 0.0))
    z = _z_from_alpha(alpha)

    # Predicción recursiva en espacio escalado
    preds_scaled = []
    for _ in range(pasos):
        x = seq[-lookback:].reshape(1, lookback, 1)
        yhat = float(model.predict(x, verbose=0)[0, 0])
        preds_scaled.append(yhat)
        seq = np.append(seq, yhat)

    # A espacio original
    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).ravel()

    # Índice futuro
    freq = frecuencia or fallback_freq or 'D'
    future_index = pd.date_range(start=last_time, periods=pasos + 1, freq=freq)[1:]

    if modo == 'retornos':
        if ultimo_close is None:
            raise ValueError("No se encontró 'ultimo_close' para reconstruir precio en modo 'retornos'.")

        # preds: retornos simple por paso (aprox)
        price_path = ultimo_close * np.cumprod(1.0 + preds)

        # Bandas: aplicamos ±z*resid_std sobre retornos y reconstruimos (aprox)
        # Nota: resid_std está en unidad original del objetivo entrenado.
        # En modo 'retornos', resid_std se interpreta como desviación aprox. de retornos en unidad original,
        # si entrenaste directamente retornos (recomendado). Documentar esta aproximación en la tesis.
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
