from prophet import Prophet
import pandas as pd


def entrenar_modelo_prophet(df: pd.DataFrame) -> Prophet:
    """
    Entrena un modelo Prophet con los datos históricos de cierre.
    El DataFrame debe contener las columnas 'time' y 'Close'.
    """
    data = df[['Close']].copy()
    data['ds'] = df.index if df.index.name == 'time' else df['time']
    data['y'] = data['Close']
    data = data[['ds', 'y']]

    modelo = Prophet(daily_seasonality=False, weekly_seasonality=True)
    modelo.fit(data)
    return modelo


def predecir_precio(modelo: Prophet, pasos: int = 3, frecuencia: str = '15min') -> pd.DataFrame:
    """
    Genera una predicción futura con el modelo entrenado.
    - pasos: número de periodos a predecir
    - frecuencia: intervalo entre predicciones (ej: '15min' para M15)
    """
    futuro = modelo.make_future_dataframe(periods=pasos, freq=frecuencia)
    forecast = modelo.predict(futuro)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(pasos)
