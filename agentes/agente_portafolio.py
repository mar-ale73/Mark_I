def asignar_capital(balance: float, senal: str, riesgo_pct: float = 0.02) -> float:
    """
    Calcula el capital a asignar según la señal operativa.

    Parámetros:
    - balance: balance actual de la cuenta
    - senal: 'comprar', 'vender' o 'mantener'
    - riesgo_pct: porcentaje del balance a arriesgar por operación

    Retorna:
    - monto en USD a asignar, o 0 si no se opera
    """
    if senal == 'mantener':
        return 0.0

    capital_asignado = balance * riesgo_pct
    return round(capital_asignado, 2)
