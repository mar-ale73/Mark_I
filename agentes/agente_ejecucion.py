import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import Workbook
from datetime import datetime


def ejecutar_operacion(simbolo: str, senal: str, capital: float, precio_actual: float) -> dict:
    if senal == 'mantener' or capital <= 0:
        return {
            'operacion': 'ninguna',     
            'capital_usado': 0,
            'precio': precio_actual,
            'resultado': 'sin acción'
        }

    tipo = 'compra' if senal == 'comprar' else 'venta'
    return {
        'operacion': tipo,
        'capital_usado': capital,
        'precio': precio_actual,
        'resultado': 'simulada'
    }


def generar_reporte_excel(predicciones: pd.DataFrame, senal: str, capital: float, operacion: dict, umbral: float, ruta: str = 'outputs/reporte_inversion.xlsx'):
    with pd.ExcelWriter(ruta, engine='openpyxl') as writer:
        # Hoja 1: Predicciones
        predicciones.to_excel(writer, sheet_name='Predicciones', index=False)

        # Hoja 2: Señal
        pd.DataFrame([{
            'Señal': senal,
            'Capital asignado': capital,
            'Umbral de decisión': umbral,
            'Fecha': datetime.now()
        }]).to_excel(writer, sheet_name='Señal', index=False)

        # Hoja 3: Operación
        pd.DataFrame([operacion]).to_excel(writer, sheet_name='Operacion', index=False)

        # Hoja 4: Métricas
        variacion = predicciones['precio_estimado'].iloc[-1] - predicciones['precio_estimado'].iloc[-2]
        retorno_pct = variacion / predicciones['precio_estimado'].iloc[-2]
        df_metricas = pd.DataFrame([{
            'Variación estimada': round(variacion, 6),
            'Retorno estimado (%)': round(retorno_pct * 100, 4),
            'Señal': senal,
            'Capital asignado': capital,
            'Umbral utilizado': umbral
        }])
        df_metricas.to_excel(writer, sheet_name='Métricas', index=False)

    # Gráfico de predicción
    plt.figure(figsize=(8, 4))
    plt.plot(predicciones['timestamp_prediccion'], predicciones['precio_estimado'], label='precio_estimado', color='blue')
    plt.fill_between(predicciones['timestamp_prediccion'], predicciones['min_esperado'], predicciones['max_esperado'], color='gray', alpha=0.3, label='rango de confianza')
    plt.title('Predicción del precio')
    plt.xlabel('Tiempo')
    plt.ylabel('Precio estimado')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/grafico_prediccion.png')
    plt.close()

    print(f"📄 Reporte generado: {ruta}")
    print(f"📊 Gráfico guardado: outputs/grafico_prediccion.png")
