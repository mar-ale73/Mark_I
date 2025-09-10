# procesamiento/eda_pipeline.py
# Ejecuta el EDA (PDF + Excel) y deriva recomendaciones de modelado/operación
# a partir de la evidencia (STL, volatilidad, ATR, correlación).
#
# Requisitos: pandas, numpy, matplotlib, statsmodels, scipy, xlsxwriter/openpyxl
# Usa: from procesamiento.eda_crispdm import ejecutar_eda

from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from statsmodels.tsa.seasonal import STL

try:
    from procesamiento.eda_crispdm import ejecutar_eda
except Exception:
    # fallback si el import relativo no aplica en tu estructura
    from eda_crispdm import ejecutar_eda


# =======================
# Helpers mínimos (independientes del EDA)
# =======================
def _safe_mkdir(p: str):
    if not os.path.exists(p):
        os.makedirs(p)

def _ensure_dt_index(df: pd.DataFrame, col_candidates=("time","timestamp","date","datetime","Date","Datetime")) -> pd.DataFrame:
    """Asegura índice datetime (sin tz) desde columna o índice existente."""
    d = df.copy()
    dtcol = next((c for c in col_candidates if c in d.columns), None)
    if dtcol is None:
        if isinstance(d.index, pd.DatetimeIndex):
            idx = d.index
        else:
            raise ValueError(f"No se encontró columna de tiempo en {list(d.columns)}")
    else:
        idx = pd.to_datetime(d[dtcol], errors="coerce", utc=True)
    d = d.copy()
    d.index = pd.to_datetime(idx).tz_localize(None)
    d = d.sort_index()
    return d

def _find_close(df: pd.DataFrame) -> str:
    for c in ["Close", "close", "Adj Close", "price", "Price"]:
        if c in df.columns:
            return c
    raise ValueError("No se encontró columna de precio/cierre.")

def _resample_ohlc(df: pd.DataFrame, freq: str, price_col: str) -> pd.DataFrame:
    """Resamplea a frecuencia deseada; preserva OHLC si existen, si no usa último cierre."""
    freq = str(freq)
    cols_lower = [c.lower() for c in df.columns]
    has_ohlc = all(x in cols_lower for x in ["open","high","low",price_col.lower()])

    if has_ohlc:
        def _first(name):
            return [c for c in df.columns if c.lower()==name][0]
        agg = {
            _first("open"): "first",
            _first("high"): "max",
            _first("low"):  "min",
            _first(price_col.lower()): "last",
        }
        vol_candidates = [c for c in df.columns if c.lower() in ("volume","tick_volume","vol")]
        if vol_candidates:
            agg[vol_candidates[0]] = "sum"
        out = df.resample(freq).agg(agg)
    else:
        out = pd.DataFrame(index=df.index)
        out[price_col] = df[price_col].resample(freq).last()
        vol_candidates = [c for c in df.columns if c.lower() in ("volume","tick_volume","vol")]
        if vol_candidates:
            out[vol_candidates[0]] = df[vol_candidates[0]].resample(freq).sum()

    return out.dropna(how="any")

def _compute_returns_blocks(df: pd.DataFrame, price_col: str) -> Tuple[pd.Series, pd.Series]:
    r = df[price_col].pct_change()
    lr = np.log(df[price_col]).diff()
    return r, lr

def _atr_if_available(df: pd.DataFrame) -> Optional[pd.Series]:
    cols = [c.lower() for c in df.columns]
    has = all(x in cols for x in ["high","low"])
    close_name = next((c for c in df.columns if c.lower() in ("close","price","adj close")), None)
    if has and close_name is not None:
        high = df[[c for c in df.columns if c.lower()=="high"][0]].astype(float)
        low = df[[c for c in df.columns if c.lower()=="low"][0]].astype(float)
        close = df[close_name].astype(float)
        prev_close = close.shift(1)
        tr = np.maximum((high - low).to_numpy(),
                        np.maximum((high - prev_close).abs().to_numpy(),
                                   (low - prev_close).abs().to_numpy()))
        atr = pd.Series(tr, index=df.index).rolling(14).mean()
        return atr
    return None

def _seasonal_period_by_freq(freq: str) -> int:
    f = str(freq).upper()
    if f in ("D","1D"): return 7        # semana
    if f in ("H","1H"): return 24       # 24 horas
    if f.endswith("T"):                 # minutos → ciclo diario aprox
        try:
            minutes = int(f[:-1])
            return max(7, int((24*60)/minutes))
        except:
            return 7
    return 7

def _stl_strength(y: pd.Series, period: int) -> Dict[str, float]:
    """Mide 'fuerza' estacional y de tendencia (Wang et al., 2006): 1 - Var(resid)/Var(resid+component)."""
    y = np.log(y.dropna())
    if len(y) < period*3:
        return {"seasonal_strength": np.nan, "trend_strength": np.nan}
    stl = STL(y, period=period, robust=True).fit()
    resid = stl.resid
    seasonal = stl.seasonal
    trend = stl.trend
    var = np.var
    s_strength = 1.0 - var(resid) / var(resid + seasonal)
    t_strength = 1.0 - var(resid) / var(resid + trend)
    return {"seasonal_strength": float(s_strength), "trend_strength": float(t_strength)}

def _pips_from_price(delta_price: float, symbol_alias: str) -> float:
    # Para EURUSD y pares con 4 decimales, 1 pip ≈ 0.0001
    # Ajusta aquí si incorporas JPY u otros.
    if symbol_alias.upper().startswith("EURUSD"):
        return delta_price * 10_000.0
    return delta_price * 10_000.0  # default


# =======================
# Núcleo: derivar recomendaciones
# =======================
def derive_recommendations(
    df_eurusd: Optional[pd.DataFrame],
    df_spy: Optional[pd.DataFrame],
    frecuencia: str,
    outdir: str = "outputs/eda",
    alias_eur: str = "EURUSD",
    alias_spy: str = "SPY",
) -> Dict[str, Any]:
    """
    Devuelve recomendaciones de modelado/operación a partir de evidencia estadística:
      - Prophet/SARIMA: estacionalidad (daily/weekly/yearly, m)
      - Modo de modelado: 'retornos' intradía, 'nivel' diario (heurística)
      - Umbrales: pips (EURUSD) y % (SPY) basados en ATR y σ de log-returns
      - Sizing: riesgo_pct ajustado por volatilidad relativa
    """
    _safe_mkdir(outdir)
    rec: Dict[str, Any] = {"frecuencia": frecuencia, "activos": {}, "portafolio": {}}

    def _process_asset(df: pd.DataFrame, alias: str, is_fx: bool):
        d = _ensure_dt_index(df)
        price_col = _find_close(d)
        d = _resample_ohlc(d, freq=frecuencia, price_col=price_col)

        # Derivadas
        ret, lret = _compute_returns_blocks(d, price_col)
        sigma20 = float(lret.rolling(20).std().iloc[-1]) if lret.dropna().size > 20 else float(np.nan)
        sigma60 = float(lret.rolling(60).std().iloc[-1]) if lret.dropna().size > 60 else float(np.nan)
        atr = _atr_if_available(d)
        atr_last = float(atr.iloc[-1]) if atr is not None and atr.notna().any() else float('nan')

        # STL strength
        m = _seasonal_period_by_freq(frecuencia)
        stl_s = _stl_strength(d[price_col], m)

        # Heurísticas de seasonality / modo
        intradia = any(x in frecuencia.lower() for x in ['t', 'min', 'h'])
        modo = 'retornos' if intradia else 'nivel'

        prophet_seasonal = {
            "daily": bool(intradia),
            "weekly": True,
            "yearly": False if intradia else True
        }
        sarima_m = m

        # Umbral recomendado
        if is_fx:
            # Basado en ATR: convertir a pips
            if np.isfinite(atr_last):
                atr_pips = _pips_from_price(atr_last, alias)
                umbral_mid_pips = max(3.0, min(20.0, 0.5 * atr_pips))   # 50% del ATR, acotado
                umbral_low_pips = max(2.0, min(15.0, 0.3 * atr_pips))   # 30% del ATR
                umbral_high_pips = max(4.0, min(30.0, 0.8 * atr_pips))  # 80% del ATR
            else:
                # fallback por σ de log-returns: aproximación de pips
                if np.isfinite(sigma20) and d[price_col].notna().any():
                    px = float(d[price_col].iloc[-1])
                    est_move = px * sigma20  # ~ desviación del precio
                    est_pips = _pips_from_price(est_move, alias)
                    umbral_mid_pips = max(3.0, min(20.0, 0.5 * est_pips))
                    umbral_low_pips = max(2.0, min(15.0, 0.3 * est_pips))
                    umbral_high_pips = max(4.0, min(30.0, 0.8 * est_pips))
                else:
                    umbral_low_pips = 3.0; umbral_mid_pips = 5.0; umbral_high_pips = 8.0
            umbral_reco = {
                "tipo": "pips",
                "low": round(umbral_low_pips, 2),
                "mid": round(umbral_mid_pips, 2),
                "high": round(umbral_high_pips, 2),
            }
        else:
            # Equity/ETF: umbral en %
            if np.isfinite(sigma20):
                # Regla: low=0.7σ, mid=1.0σ, high=1.3σ (en %)
                umbral_low = max(0.05, min(2.0, 100.0 * 0.7 * sigma20))
                umbral_mid = max(0.07, min(3.0, 100.0 * 1.0 * sigma20))
                umbral_high = max(0.10, min(4.0, 100.0 * 1.3 * sigma20))
            else:
                umbral_low = 0.10; umbral_mid = 0.15; umbral_high = 0.25
            umbral_reco = {
                "tipo": "percent",
                "low": round(umbral_low, 3),
                "mid": round(umbral_mid, 3),
                "high": round(umbral_high, 3),
            }

        # Sizing recomendado (ajuste por volatilidad)
        # Base 2%; escala por sigma60 vs mediana histórica sigma60 (proxy robusto)
        sigma_hist = lret.rolling(60).std().dropna()
        if not sigma_hist.empty:
            sigma_mediana = float(sigma_hist.median())
            if np.isfinite(sigma60) and sigma_mediana > 0:
                factor = min(1.0, max(0.4, sigma_mediana / sigma60))  # si volatilidad actual > histórica → baja tamaño
                riesgo_pct = round(0.02 * factor, 4)
            else:
                riesgo_pct = 0.02
        else:
            riesgo_pct = 0.02

        rec['activos'][alias] = {
            "frecuencia": frecuencia,
            "modo_modelado": modo,  # 'retornos' intradía, 'nivel' diario (heurística)
            "prophet": {
                "daily_seasonality": prophet_seasonal["daily"],
                "weekly_seasonality": prophet_seasonal["weekly"],
                "yearly_seasonality": prophet_seasonal["yearly"],
                "interval_width": 0.90,
                "seasonality_mode": "additive"
            },
            "sarima": {
                "m": sarima_m,
                "order_sugerido": (3,0,3) if modo == 'retornos' else (1,1,1),
                "seasonal_order_sugerido": (1,1,1,sarima_m) if (not intradia or sarima_m in (7,24)) else (0,0,0,0)
            },
            "mlp": {
                "lookback": 20 if not intradia else 20,
                "hidden_layer_sizes": (64,32),
                "modo": modo
            },
            "lstm": {
                "lookback": 40 if intradia else 40,
                "epochs": 40 if intradia else 30,
                "modo": modo
            },
            "umbrales": umbral_reco,
            "risk": {
                "riesgo_pct_recomendado": riesgo_pct,
                "sigma20": sigma20,
                "sigma60": sigma60,
                "ATR14": atr_last if np.isfinite(atr_last) else None
            },
            "stl_strength": stl_s
        }

    # Procesa EURUSD
    if df_eurusd is not None:
        _process_asset(df_eurusd, alias_eur, is_fx=True)
    # Procesa SPY/US500
    if df_spy is not None:
        _process_asset(df_spy, alias_spy, is_fx=False)

    # Sugerencias de portafolio (placeholder sencillo)
    if "EURUSD" in rec["activos"] and alias_spy in rec["activos"]:
        rec["portafolio"] = {
            "nota": "Las correlaciones intradía suelen ser inestables; trate los activos con sizing independiente.",
            "diversificacion": "Asignación equitativa inicial; ajustar según desempeño y drawdown."
        }
    return rec


# =======================
# Exportadores de recomendaciones
# =======================
def save_recommendations_json(reco: Dict[str, Any], outdir: str = "outputs/eda", filename: str = "recomendaciones.json") -> str:
    _safe_mkdir(outdir)
    path = os.path.join(outdir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reco, f, ensure_ascii=False, indent=2)
    print(f"💾 Recomendaciones (JSON): {path}")
    return path

def save_recommendations_excel(reco: Dict[str, Any], outdir: str = "outputs/eda", filename: str = "recomendaciones.xlsx") -> str:
    _safe_mkdir(outdir)
    path = os.path.join(outdir, filename)
    # aplanar
    rows = []
    for sym, v in reco.get("activos", {}).items():
        um = v.get("umbrales", {})
        risk = v.get("risk", {})
        rows.append({
            "Activo": sym,
            "Frecuencia": v.get("frecuencia"),
            "Modo_modelado": v.get("modo_modelado"),
            "Prophet_daily": v.get("prophet",{}).get("daily_seasonality"),
            "Prophet_weekly": v.get("prophet",{}).get("weekly_seasonality"),
            "Prophet_yearly": v.get("prophet",{}).get("yearly_seasonality"),
            "SARIMA_m": v.get("sarima",{}).get("m"),
            "SARIMA_order_sug": str(v.get("sarima",{}).get("order_sugerido")),
            "SARIMA_seasonal_sug": str(v.get("sarima",{}).get("seasonal_order_sugerido")),
            "MLP_lookback": v.get("mlp",{}).get("lookback"),
            "LSTM_lookback": v.get("lstm",{}).get("lookback"),
            "Umbral_tipo": um.get("tipo"),
            "Umbral_low": um.get("low"),
            "Umbral_mid": um.get("mid"),
            "Umbral_high": um.get("high"),
            "Riesgo_pct": risk.get("riesgo_pct_recomendado"),
            "sigma20": risk.get("sigma20"),
            "sigma60": risk.get("sigma60"),
            "ATR14": risk.get("ATR14"),
            "STL_seasonal_strength": v.get("stl_strength",{}).get("seasonal_strength"),
            "STL_trend_strength": v.get("stl_strength",{}).get("trend_strength"),
        })
    df_out = pd.DataFrame(rows)
    try:
        with pd.ExcelWriter(path, engine="xlsxwriter", datetime_format="yyyy-mm-dd hh:mm") as w:
            df_out.to_excel(w, sheet_name="Recomendaciones", index=False)
    except Exception:
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            df_out.to_excel(w, sheet_name="Recomendaciones", index=False)
    print(f"📊 Recomendaciones (Excel): {path}")
    return path


# =======================
# Pipeline principal
# =======================
def run_eda_and_recommend(
    df_eurusd: Optional[pd.DataFrame],
    df_spy: Optional[pd.DataFrame],
    cfg: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    1) Ejecuta EDA (genera PDF + Excel de diagnóstico).
    2) Deriva recomendaciones (JSON + Excel) para modelado/operación.
    Devuelve el diccionario de recomendaciones.
    """
    eda_cfg = (cfg or {}).get("eda", {})
    frecuencia = str(eda_cfg.get("frecuencia_resampleo", "D"))
    outdir = eda_cfg.get("outdir", "outputs/eda")
    alias_eur = eda_cfg.get("alias_eur", "EURUSD")
    alias_spy = eda_cfg.get("alias_spy", "SPY")

    # 1) Ejecutar EDA con tu script (genera PDF + Excel de EDA)
    ejecutar_eda(df_eurusd=df_eurusd, df_spy=df_spy, cfg=cfg)

    # 2) Derivar recomendaciones (modelo, umbrales, sizing)
    reco = derive_recommendations(
        df_eurusd=df_eurusd,
        df_spy=df_spy,
        frecuencia=frecuencia,
        outdir=outdir,
        alias_eur=alias_eur,
        alias_spy=alias_spy
    )

    # Guardar recomendaciones
    save_recommendations_json(reco, outdir=outdir, filename="recomendaciones.json")
    save_recommendations_excel(reco, outdir=outdir, filename="recomendaciones.xlsx")

    # Imprimir resumen corto en consola
    for sym, v in reco.get("activos", {}).items():
        um = v["umbrales"]
        print(f"→ {sym}: modo={v['modo_modelado']}, Prophet(d={v['prophet']['daily_seasonality']},w={v['prophet']['weekly_seasonality']},y={v['prophet']['yearly_seasonality']}), "
              f"SARIMA m={v['sarima']['m']}, umbral {um['tipo']}=({um['low']}, {um['mid']}, {um['high']}), riesgo_pct≈{v['risk']['riesgo_pct_recomendado']}")
    return reco


# =======================
# Ejemplo de uso (quítalo o adáptalo a tu runner)
# =======================
if __name__ == "__main__":
    # Ejemplo mínimo: carga tus DataFrames aquí (reemplaza por tu carga real)
    # df_eurusd = pd.read_csv("data/eurusd.csv")   # Debe incluir columna de tiempo y Close/close
    # df_spy    = pd.read_csv("data/spy.csv")
    df_eurusd = None
    df_spy = None

    cfg = {
        "eda": {
            "frecuencia_resampleo": "15T",     # 'D' para diario (SPY), '15T' para M15 (EUR/USD)
            "outdir": "outputs/eda",
            "ventana_media_movil": 30,
            "acf_lags": 40,
            "rolling_vol_windows": [20,60,120],
            "rolling_corr_window": 60,
            "export_pdf": True,
            "pdf_filename": "EDA_informe.pdf",
            "alias_eur": "EURUSD",
            "alias_spy": "SPY"
        }
    }

    # Corre pipeline (generará PDF/Excel del EDA y JSON/Excel de recomendaciones)
    run_eda_and_recommend(df_eurusd, df_spy, cfg)
