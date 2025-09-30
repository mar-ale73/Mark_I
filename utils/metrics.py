# utils/metrics.py
from __future__ import annotations

from typing import Dict, Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# Métricas predictivas
# =========================
def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    """
    Devuelve MAE, RMSE, MAPE_% y R2 para dos series del mismo tamaño.
    Evita divisiones por 0 en el MAPE.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = min(len(y_true), len(y_pred))
    y_true = y_true[:m]
    y_pred = y_pred[:m]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    denom = np.where(y_true == 0.0, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE_%": float(mape),
        "R2": float(r2),
    }


def directional_accuracy(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """
    Accuracy direccional basada en el signo del primer diferencial.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    d_true = np.sign(np.diff(y_true))
    d_pred = np.sign(np.diff(y_pred))
    m = min(len(d_true), len(d_pred))
    if m == 0:
        return np.nan
    return float((d_true[:m] == d_pred[:m]).mean())


# =========================
# Métricas “decisionales” y utilidades
# =========================
def roundtrip_bps(costs_bps: float = 0.0, slippage_ticks_as_bps: float = 0.0) -> float:
    """
    Convierte costos de ida+vuelta (comisiones + slippage) a bps totales por trade “roundtrip”.
    Por compatibilidad con tu benchmark actual, tratamos slippage_ticks como bps.
    """
    return 2.0 * (float(costs_bps) + float(slippage_ticks_as_bps))


def apply_costs(gross_returns: Iterable[float], roundtrip_bps_val: float) -> np.ndarray:
    """
    Aplica costos (en bps roundtrip) a una serie de retornos brutos por trade.
    Devuelve retornos netos.
    """
    gross = np.asarray(gross_returns, dtype=float)
    cost_frac = float(roundtrip_bps_val) / 1e4
    return gross - cost_frac


def sortino_ratio(returns: Iterable[float], risk_free: float = 0.0) -> float:
    """
    Sortino con downside deviation. Ignora NaN/infs.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return np.nan
    downside = r[r < risk_free] - risk_free
    down_std = float(np.std(downside)) if downside.size > 0 else np.nan
    if not np.isfinite(down_std) or down_std == 0.0:
        return np.nan
    return float((np.mean(r) - risk_free) / down_std)


def profit_factor(returns: Iterable[float]) -> float:
    """
    Profit factor = sum(ganancias) / |sum(pérdidas)|
    """
    r = np.asarray(returns, dtype=float)
    gains = r[r > 0.0].sum()
    losses = r[r < 0.0].sum()
    if losses == 0.0:
        return float("inf") if gains > 0.0 else np.nan
    return float(gains / abs(losses))


def equity_curve(returns: Iterable[float]) -> np.ndarray:
    """
    Curva de capital simple (base 1.0): prod(1 + r_t).
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return np.array([], dtype=float)
    return np.cumprod(1.0 + r)


def max_drawdown_pct(equity: Iterable[float]) -> float:
    """
    Máximo drawdown en % (negativo), consistente con tu implementación actual.
    """
    eq = np.asarray(equity, dtype=float)
    if eq.size == 0:
        return np.nan
    peak = -np.inf
    max_dd = 0.0
    for x in eq:
        peak = max(peak, x)
        dd = (x / peak) - 1.0
        max_dd = min(max_dd, dd)
    return float(max_dd * 100.0)


def decision_metrics_from_returns(returns_net: Iterable[float]) -> Dict[str, float]:
    """
    Empaqueta Sortino, ProfitFactor, MaxDD_% y retorno acumulado en %.
    """
    r = np.asarray(returns_net, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {
            "Sortino": np.nan,
            "Profit_Factor": np.nan,
            "MaxDD_%": np.nan,
            "Ret_Acumulado_%": np.nan,
            "Trades": 0,
        }

    srt = sortino_ratio(r)
    pf = profit_factor(r)
    eq = equity_curve(r)
    maxdd = max_drawdown_pct(eq)
    ret_acum = float((eq[-1] - 1.0) * 100.0) if eq.size > 0 else np.nan

    return {
        "Sortino": float(srt) if np.isfinite(srt) else np.nan,
        "Profit_Factor": float(pf) if np.isfinite(pf) else np.nan,
        "MaxDD_%": float(maxdd),
        "Ret_Acumulado_%": float(ret_acum),
        "Trades": int(r.size),
    }
