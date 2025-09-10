import pandas as pd

def generar_senal_operativa(predicciones: pd.DataFrame, umbral: float = 0.0003) -> str:
    """
    Genera una señal operativa a partir de la primera y última predicción del modelo.
    """
    if len(predicciones) < 2:
        return 'mantener'

    precio_inicial = predicciones.iloc[0]['precio_estimado']
    precio_final = predicciones.iloc[-1]['precio_estimado']
    variacion = precio_final - precio_inicial

    print(f"📊 Comparando precio inicial {precio_inicial:.5f} y final {precio_final:.5f}")
    print(f"📈 Variación estimada: {variacion:.5f} ({variacion * 10000:.2f} pips)")

    if variacion > umbral:
        return 'comprar'
    elif variacion < -umbral:
        return 'vender'
    else:
        return 'mantener'