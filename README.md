# Mark_I — Análisis (EDA) y Modelado de Activos (EURUSD / SPY)

Este proyecto implementa un flujo **CRISP-DM** para:
1) **EDA (Exploratory Data Analysis)** de **EURUSD** y **SPY**  
2) **Entrenamiento y predicción** con **Prophet**, generación de **señal operativa**, **asignación de capital** y **reporte**.

> Se ejecuta desde terminal, sin necesidad de modificar el código. La **configuración** se controla mediante `utils/config.yaml`.

---

## 📁 Estructura del proyecto (carpetas clave)

```
Mark_I/
├─ app/
│  ├─ __init__.py
│  └─ main.py                 # punto de entrada
├─ procesamiento/
│  ├─ eda_crispdm.py          # EDA CRISP-DM (EURUSD y SPY)
│  └─ features.py
├─ modelos/
│  ├─ prophet_model.py
│  └─ evaluacion_modelos.py
├─ agentes/
│  ├─ agente_analisis.py
│  ├─ agente_portafolio.py
│  └─ agente_ejecucion.py
├─ conexion/
│  └─ easy_Trading.py
├─ reportes/
│  └─ reportes_excel.py
├─ utils/
│  └─ config.yaml             # configuración del usuario (ver ejemplo abajo)
├─ outputs/                   # resultados (se crea al ejecutar)
└─ requirements.txt
```

> Asegúrate de que `app/` tenga `__init__.py` (aunque sea vacío) para poder ejecutar con `python -m app.main`.

---

## 🛠️ Requisitos e instalación

1) **Python 3.10+** y (opcional) **entorno virtual**  
   **PowerShell (Windows):**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   **CMD (Windows):**
   ```cmd
   python -m venv .venv
   .\.venv\Scripts\activate.bat
   ```
   **macOS/Linux:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2) **Dependencias**
   - Usa el `requirements.txt` del repo y agrega (si no están) estas librerías para el EDA:
     ```
     scipy>=1.10
     statsmodels>=0.14
     xlsxwriter>=3.0   # opcional si NO usas el fallback a openpyxl
     ```
   - Instala:
     ```bash
     python -m pip install --upgrade pip setuptools wheel
     python -m pip install -r requirements.txt
     ```

3) **MetaTrader 5** instalado y credenciales válidas del broker si usarás extracción en vivo.

---

## ⚙️ Configuración del usuario (`utils/config.yaml`)

Ajusta estos campos. Puedes partir del **ejemplo** más abajo.

### 🔎 Descripción de variables principales

- **simbolo**: activo base (ej. `EURUSD`).
- **timeframe**: marco temporal (`M1`, `M5`, `M15`, `H1`, `D1`).
- **cantidad_datos**: número de velas a extraer de MT5.
- **modelo**: hoy `prophet` (otros modelos pueden integrarse).
- **pasos_prediccion / frecuencia_prediccion**: horizonte de proyección (p. ej. `48` * `H` = 48 horas).
- **umbral_senal**: define cuándo la predicción se considera compra/venta vs. mantener.
- **riesgo_por_trade**: % del balance a arriesgar por operación (ej. `0.02` = 2%).
- **volumen_minimo**: piso de lotaje permitido (ej. `0.01`).
- **stop_loss_pips / take_profit_pips**: distancias en pips para SL/TP.
- **pip_size** (opcional): fuerza el tamaño de pip si el broker lo reporta raro.
- **eda.habilitar**: si `true`, el flujo normal genera EDA además del modelado.
- **eda.frecuencia_resampleo**: agregación del EDA (`D`, `H`, `15T`, …).
- **simbolo_spy**: símbolo SPY en tu broker. Si no existe, usa `spy_csv`.
- **spy_csv**: ruta a CSV de SPY si no hay símbolo en MT5.  
  - Mínimo: `timestamp`, `Close`. Ideal: `timestamp, Open, High, Low, Close, Volume`.
- **mt5**: credenciales y ruta del terminal MT5. Considera variables de entorno por seguridad.

> **Seguridad**: mueve `mt5.login`, `mt5.password`, etc. a variables de entorno (`MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`, `MT5_PATH`) y no subas tus credenciales al repo.

---

## ▶️ Cómo ejecutar

> **Siempre ejecuta desde la raíz del proyecto** (la carpeta que contiene `app/` y `utils/`).

### 1) Solo EDA (EURUSD + SPY)

**Frecuencia diaria:**
```bash
python -m app.main --modo eda --freq D
```

**Frecuencia horaria:**
```bash
python -m app.main --modo eda --freq H
```

**Frecuencia 15 minutos:**
```bash
python -m app.main --modo eda --freq 15T
```

- Si `simbolo_spy` no existe en tu broker, define un **CSV** en `spy_csv` (ver Configuración).
- Salidas del EDA:
  - **Excel**: `outputs/eda/EDA_resumen.xlsx`  
    - Hojas: `EURUSD_basic`, `EURUSD_drawdown`, `EURUSD_dd_summary`, `EURUSD_stationarity`, `SPY_*`, `Correlation_matrix`, `Rolling_corr_60`.
  - **Gráficos**: `outputs/eda/*.png`  
    - Precio + SMAs, log-returns, volatilidad rolling, ACF/PACF, STL, correlación móvil EURUSD–SPY.

### 2) Flujo normal (modelado + señal + reporte + ejecución)

```bash
# modo normal explícito
python -m app.main --modo normal

# o simplemente (por defecto es normal)
python -m app.main
```

- Entrena **Prophet**, genera **predicciones**, **señal**, **asignación** y, si aplica, envía orden a MT5.
- Reporte base en: `outputs/reporte_inversion.xlsx`.
- Métricas del modelo (MAE, RMSE, MAPE, R², Sortino, Accuracy direccional, horizonte) se escriben en el reporte (hoja de métricas e histórico).

---

## 📤 Resultados generados

- `outputs/eda/`  
  - `EDA_resumen.xlsx` (tablas EDA)  
  - `*.png` (gráficos EDA)
- `outputs/reporte_inversion.xlsx`  
  - Señal, predicciones, asignación, operación simulada.
  - Métricas del modelo (hoja de métricas + histórico).

---

## 🧪 Inspección rápida de tablas (opcional)

Ver tablas del Excel EDA sin abrir Excel:

```bash
python -c "import pandas as pd; print(pd.read_excel('outputs/eda/EDA_resumen.xlsx','EURUSD_basic')); print(); print(pd.read_excel('outputs/eda/EDA_resumen.xlsx','SPY_basic'))"
python -c "import pandas as pd; print(pd.read_excel('outputs/eda/EDA_resumen.xlsx','EURUSD_stationarity')); print(); print(pd.read_excel('outputs/eda/EDA_resumen.xlsx','SPY_stationarity'))"
python -c "import pandas as pd; print(pd.read_excel('outputs/eda/EDA_resumen.xlsx','Correlation_matrix'))"
```

---

## 🔧 Solución de problemas (FAQ)

- **`ModuleNotFoundError: statsmodels`**  
  ```bash
  python -m pip install statsmodels scipy
  ```

- **`ModuleNotFoundError: xlsxwriter`**  
  - Instala:
    ```bash
    python -m pip install xlsxwriter
    ```
  - O usa el **fallback a openpyxl** (ya soportado si aplicaste el cambio en `eda_crispdm.py`).

- **Error de conexión MT5**  
  - Verifica credenciales/servidor/ruta en `utils/config.yaml` (o variables de entorno).
  - Asegúrate de tener **MetaTrader 5** instalado y sesión disponible.

- **El símbolo SPY no existe en tu broker**  
  - Usa `spy_csv` en `config.yaml`.  
    - Columnas mínimas: `timestamp`, `Close` (UTC o normalizables por pandas).

- **`pip` instala en otro Python**  
  ```bash
  python -c "import sys; print(sys.executable)"
  python -m pip -V
  ```
  Deben apuntar a `.venv`. Si no, activa:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```

---

## 🧭 Metodología (CRISP-DM)

- **Business Understanding**: `utils/config.yaml` (parámetros de negocio, riesgo y operación).
- **Data Understanding**: `procesamiento/eda_crispdm.py` (EDA EURUSD y SPY).
- **Data Preparation**: `procesamiento/features.py` (indicadores y transformaciones).
- **Modeling**: `modelos/prophet_model.py`.
- **Evaluation**: `modelos/evaluacion_modelos.py` + reportes/EDA.
- **Deployment**: `agentes/*` + integración MT5 en `app/main.py`.

---

## 📚 Referencias

- **CRISP-DM 1.0** (SPSS/IBM) – metodología de minería de datos.
- Hyndman & Athanasopoulos. *Forecasting: Principles and Practice*.
- Box, Jenkins & Reinsel. *Time Series Analysis: Forecasting and Control*.
- **MetaTrader5 (Python)** – API para extracción OHLCV.
