import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
import inspect
import warnings

# Silenciar algunos warnings cosméticos
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# === Reusar tus features si están disponibles ===
try:
    from procesamiento.features import aplicar_todos_los_indicadores
except Exception:
    aplicar_todos_los_indicadores = None


# ------------ Utilidades ------------
def _make_tz_naive_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte index y columnas datetime con tz a naive (sin zona horaria).
    Solo actúa sobre tipos datetime con tz; no intenta parsear columnas object.
    """
    d = df.copy()

    # Index
    if isinstance(d.index, pd.DatetimeIndex) and getattr(d.index, "tz", None) is not None:
        d.index = d.index.tz_localize(None)

    # Columnas datetime con tz
    for c in d.columns:
        if pd.api.types.is_datetime64tz_dtype(d[c]):
            d[c] = d[c].dt.tz_localize(None)

    return d


def _safe_mkdir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def _ensure_dt_index(df, col_candidates=("timestamp", "date", "datetime", "Date", "Datetime")):
    df = df.copy()
    dtcol = None
    for c in col_candidates:
        if c in df.columns:
            dtcol = c
            break
    if dtcol is None:
        raise ValueError(f"No se encontró columna de tiempo en {list(df.columns)}")
    df[dtcol] = pd.to_datetime(df[dtcol], errors="coerce", utc=True)
    df = df.dropna(subset=[dtcol]).sort_values(dtcol).set_index(dtcol)
    return df


def _find_close(df):
    for c in ["close", "Close", "Adj Close", "price", "Price"]:
        if c in df.columns:
            return c
    raise ValueError("No se encontró columna de precio/cierre.")


def _resample(df, freq, price_col):
    """
    Resamplea preservando OHLC (+ Volume si existe) o, si solo hay close, toma el último valor.
    Volume se agrega con suma.
    """
    freq = str(freq).lower()  # evita FutureWarning por 'H'
    cols_lower = df.columns.str.lower()
    has_ohlc = all(x in cols_lower for x in ["open", "high", "low", price_col.lower()])

    if has_ohlc:
        agg = {
            [c for c in df.columns if c.lower() == "open"][0]: "first",
            [c for c in df.columns if c.lower() == "high"][0]: "max",
            [c for c in df.columns if c.lower() == "low"][0]: "min",
            [c for c in df.columns if c.lower() == price_col.lower()][0]: "last",
        }
        # incluir volumen si existe
        vol_candidates = [c for c in df.columns if c.lower() in ("volume", "tick_volume", "vol")]
        if vol_candidates:
            agg[vol_candidates[0]] = "sum"
        return df.resample(freq).agg(agg).dropna(how="any")

    # Caso solo close
    out = pd.DataFrame(index=df.index)
    out[price_col] = df[price_col].resample(freq).last()

    vol_candidates = [c for c in df.columns if c.lower() in ("volume", "tick_volume", "vol")]
    if vol_candidates:
        out[vol_candidates[0]] = df[vol_candidates[0]].resample(freq).sum()

    return out.dropna()


def _prep_base(df, freq):
    df = _ensure_dt_index(df)
    price_col = _find_close(df)
    df = _resample(df, freq, price_col)
    df["ret"] = df[price_col].pct_change()
    df["logret"] = np.log(df[price_col]).diff()
    return df, price_col


def _stationarity(series):
    s = series.dropna()
    out = {"ADF_stat": np.nan, "ADF_p": np.nan, "KPSS_stat": np.nan, "KPSS_p": np.nan}
    if len(s) > 10:
        adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
        out.update({"ADF_stat": float(adf_stat), "ADF_p": float(adf_p)})
        try:
            kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
            out.update({"KPSS_stat": float(kpss_stat), "KPSS_p": float(kpss_p)})
        except Exception:
            # KPSS puede emitir warnings o fallar en casos límite; ignoramos para robustez
            pass
    return out


# ------------ Gráficos ------------
def _plot_series(df, price_col, symbol, outdir):
    plt.figure(figsize=(11, 5))
    plt.plot(df.index, df[price_col], label=symbol)
    for w in [20, 60, 120, 200]:
        c = f"sma_{w}"
        if c in df.columns:
            plt.plot(df.index, df[c], label=c)
    plt.title(f"Precio y SMAs - {symbol}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{symbol}_precio_sma.png"))
    plt.close()

    plt.figure(figsize=(11, 4))
    plt.plot(df.index, df["logret"])
    plt.title(f"Log-returns - {symbol}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{symbol}_logret.png"))
    plt.close()

    if any(c.startswith("vol_") for c in df.columns):
        plt.figure(figsize=(11, 4))
        for c in [c for c in df.columns if c.startswith("vol_")]:
            plt.plot(df.index, df[c], label=c)
        plt.legend()
        plt.title(f"Volatilidad (rolling) - {symbol}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{symbol}_vol.png"))
        plt.close()

    # ACF/PACF de logret
    if df["logret"].dropna().size > 50:
        fig = plt.figure(figsize=(12, 4))
        plot_acf(df["logret"].dropna(), lags=40, ax=plt.gca())
        plt.title(f"ACF logret - {symbol}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{symbol}_acf_logret.png"))
        plt.close()

        fig = plt.figure(figsize=(12, 4))
        plot_pacf(df["logret"].dropna(), lags=40, ax=plt.gca(), method="ywm")
        plt.title(f"PACF logret - {symbol}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{symbol}_pacf_logret.png"))
        plt.close()


def _plot_stl(df, price_col, symbol, outdir, seasonal):
    y = np.log(df[price_col].dropna())
    if len(y) < seasonal * 3:
        return
    stl = STL(y, period=seasonal, robust=True).fit()
    fig = stl.plot()
    fig.set_size_inches(10, 7)
    fig.suptitle(f"STL (log precio) - {symbol}")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{symbol}_stl.png"))
    plt.close(fig)


# ------------ Tablas / Excel ------------
def _summary_tables(df, price_col, symbol):
    basic = pd.DataFrame(
        {
            "symbol": [symbol],
            "n": [df[price_col].count()],
            "start": [df.index.min()],
            "end": [df.index.max()],
            "close_mean": [df[price_col].mean()],
            "close_std": [df[price_col].std()],
            "ret_mean": [df["ret"].mean()],
            "ret_std": [df["ret"].std()],
            "logret_mean": [df["logret"].mean()],
            "logret_std": [df["logret"].std()],
            "ann_vol_252": [df["logret"].std() * np.sqrt(252)],
        }
    )

    dd = df[price_col] / df[price_col].cummax() - 1
    dd_tbl = pd.DataFrame({"timestamp": df.index, "drawdown": dd}).set_index("timestamp")

    stat_price = _stationarity(df[price_col])
    stat_ret = _stationarity(df["logret"])
    stat = pd.DataFrame(
        [
            {"series": "price", **stat_price},
            {"series": "logret", **stat_ret},
        ]
    )

    return basic, dd_tbl, pd.DataFrame({"max_drawdown": [dd.min()]}), stat


def _export_excel(outpath, resumen_por_activo, corr_df=None, roll_corr=None):
    # Motor: intenta xlsxwriter; si no está, cae a openpyxl
    try:
        import xlsxwriter  # noqa: F401
        writer_kwargs = {"engine": "xlsxwriter", "datetime_format": "yyyy-mm-dd hh:mm"}
    except ImportError:
        writer_kwargs = {"engine": "openpyxl"}

    with pd.ExcelWriter(outpath, **writer_kwargs) as w:
        for k, v in resumen_por_activo.items():
            # Copias y limpieza de tz
            basic = _make_tz_naive_df(v["basic"])
            drawdown = _make_tz_naive_df(v["drawdown"])
            dd_summary = _make_tz_naive_df(v["dd_summary"])
            stationarity = _make_tz_naive_df(v["stationarity"])

            # Asegura que columnas 'start'/'end' sean naive si existen
            for col in ("start", "end"):
                if col in basic.columns:
                    basic[col] = pd.to_datetime(basic[col], errors="coerce").dt.tz_localize(None)

            basic.to_excel(w, sheet_name=f"{k}_basic", index=False)
            drawdown.to_excel(w, sheet_name=f"{k}_drawdown")
            dd_summary.to_excel(w, sheet_name=f"{k}_dd_summary", index=False)
            stationarity.to_excel(w, sheet_name=f"{k}_stationarity", index=False)

        if corr_df is not None:
            _make_tz_naive_df(corr_df).to_excel(w, sheet_name="Correlation_matrix")
        if roll_corr is not None:
            _make_tz_naive_df(roll_corr).to_excel(w, sheet_name="Rolling_corr_60")


# ------------ Público ------------
def ejecutar_eda(df_eurusd=None, df_spy=None, cfg: dict = None):
    """
    Ejecuta EDA siguiendo CRISP-DM (Understanding & Preparation) reutilizando tu proyecto.
    - df_eurusd / df_spy: DataFrames crudos con timestamp & close/ohlc
    - cfg: dict completo de tu config (para opciones EDA)
    """
    eda_cfg = (cfg or {}).get("eda", {})
    freq = str(eda_cfg.get("frecuencia_resampleo", "D")).lower()
    outdir = eda_cfg.get("outdir", "outputs/eda")
    stl_period = eda_cfg.get("stl_periodo", 7)
    _safe_mkdir(outdir)

    resumen = {}
    datos = {}

    for symbol, df in [("EURUSD", df_eurusd), ("SPY", df_spy)]:
        if df is None:
            continue

        df, price_col = _prep_base(df, freq)

        # Reusar tus features si están disponibles (tolera distintas firmas)
        if callable(aplicar_todos_los_indicadores):
            try:
                params = inspect.signature(aplicar_todos_los_indicadores).parameters
                if "price_col" in params:
                    df = aplicar_todos_los_indicadores(df, price_col=price_col)
                else:
                    df = aplicar_todos_los_indicadores(df)  # firma sin price_col
            except TypeError:
                # Por si la firma difiere (args/kwargs), cae a llamada simple
                df = aplicar_todos_los_indicadores(df)
            except Exception as e:
                print(f"⚠️ aplicar_todos_los_indicadores falló en EDA: {e}. Uso fallback básico.")
                for w in [5, 20, 60, 120]:
                    df[f"sma_{w}"] = df[price_col].rolling(w).mean()
                    df[f"vol_{w}"] = df["logret"].rolling(w).std() * np.sqrt(252)
        else:
            # Fallback si no existe la función
            for w in [5, 20, 60, 120]:
                df[f"sma_{w}"] = df[price_col].rolling(w).mean()
                df[f"vol_{w}"] = df["logret"].rolling(w).std() * np.sqrt(252)

        _plot_series(df, price_col, symbol, outdir)
        _plot_stl(df, price_col, symbol, outdir, stl_period)

        basic, dd_tbl, dd_sum, stat = _summary_tables(df, price_col, symbol)
        resumen[symbol] = {
            "basic": basic,
            "drawdown": dd_tbl,
            "dd_summary": dd_sum,
            "stationarity": stat,
        }
        datos[symbol] = df

    # Correlación entre activos (si hay ambos)
    corr_df = roll_corr = None
    if "EURUSD" in datos and "SPY" in datos:
        m = (
            datos["EURUSD"][["logret"]]
            .rename(columns={"logret": "logret_EURUSD"})
            .join(
                datos["SPY"][["logret"]].rename(columns={"logret": "logret_SPY"}),
                how="inner",
            )
            .dropna()
        )
                # Correlación entre activos (si hay ambos)
        corr_df = roll_corr = None
        if "EURUSD" in datos and "SPY" in datos:
            m = (
                datos["EURUSD"][["logret"]].rename(columns={"logret": "logret_EURUSD"})
                .join(datos["SPY"][["logret"]].rename(columns={"logret": "logret_SPY"}), how="inner")
                .dropna()
            )
            # Matriz de correlación estática
            corr_df = m.corr()

            # Correlación móvil emparejada (ventana 60)
            roll_corr = (
                m["logret_EURUSD"].rolling(60).corr(m["logret_SPY"])
                .dropna()
                .to_frame("rolling_corr_60")
            )

            # Gráfico
            plt.figure(figsize=(11, 4))
            plt.plot(roll_corr.index, roll_corr["rolling_corr_60"])
            plt.axhline(0, linestyle="--")
            plt.title("Correlación móvil (60) logret: EURUSD vs S&P500")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "EURUSD_SPY_rolling_corr.png"))
            plt.close()


        plt.figure(figsize=(11, 4))
        plt.plot(roll_corr.index, roll_corr["rolling_corr_60"])
        plt.axhline(0, linestyle="--")
        plt.title("Correlación móvil (60) logret: EURUSD vs SPY")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "EURUSD_SPY_rolling_corr.png"))
        plt.close()

    _export_excel(os.path.join(outdir, "EDA_resumen.xlsx"), resumen, corr_df, roll_corr)
    print(f"✅ EDA completado. Resultados en: {outdir}")
