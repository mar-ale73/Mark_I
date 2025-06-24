import pandas as pd

def generar_senal_operativa(predicciones: pd.DataFrame, umbral: float = 0.0003) -> str:
    """
    Genera una señal operativa a partir de las predicciones de un modelo.

    Parámetros:
    - predicciones: DataFrame con columnas ['ds', 'yhat'].
    - umbral: variación mínima esperada para considerar una señal de compra o venta.

    Retorna:
    - 'comprar', 'vender' o 'mantener'
    """
    if len(predicciones) < 2:
        return 'mantener'

    precio_actual = predicciones.iloc[-2]['precio_estimado']
    precio_futuro = predicciones.iloc[-1]['precio_estimado']
    variacion = precio_futuro - precio_actual

    if variacion > umbral:
        return 'comprar'
    elif variacion < -umbral:
        return 'vender'
    else:
        return 'mantener'
