# reportes/reportes_excel.py
import pandas as pd

import os
from pathlib import Path


def write_metrics_sheet(ruta_excel: str, metrics: dict, sheet_name: str = 'Métricas Modelo'):
    """Reemplaza la hoja con las métricas de la última corrida."""
    df = pd.DataFrame([metrics])
    with pd.ExcelWriter(ruta_excel, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

def append_history(ruta_excel: str, metrics: dict, hist_sheet: str = 'Historico Métricas'):
    """Agrega una fila al histórico (crea la hoja si no existe)."""
    row = pd.DataFrame([metrics])
    try:
        hist_exist = pd.read_excel(ruta_excel, sheet_name=hist_sheet)
        hist_concat = pd.concat([hist_exist, row], ignore_index=True)
    except Exception:
        hist_concat = row

    with pd.ExcelWriter(ruta_excel, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        hist_concat.to_excel(writer, sheet_name=hist_sheet, index=False)
# --- NUEVO: escribir hoja de benchmark en el libro principal ---


def write_benchmark_sheet(ruta_excel: str, df_benchmark: pd.DataFrame, sheet_name: str = "Comparación_Modelos") -> None:
    """
    Crea o actualiza (reemplaza) la hoja 'Comparación_Modelos' en el Excel principal.
    Si el archivo no existe, lo crea. Si la hoja existe, la elimina y vuelve a escribirla.
    """
    ruta = Path(ruta_excel)
    ruta.parent.mkdir(parents=True, exist_ok=True)

    # Si ya existe el archivo, abrimos en modo append y reemplazamos la hoja
    if ruta.exists():
        with pd.ExcelWriter(ruta, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            df_benchmark.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(ruta, engine="openpyxl", mode="w") as writer:
            df_benchmark.to_excel(writer, sheet_name=sheet_name, index=False)
