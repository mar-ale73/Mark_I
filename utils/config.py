import MetaTrader5 as mt5

if not mt5.initialize(login=68238343, password='Colombia123*', server='RoboForex-PRO', path=r'C:\Program Files\RoboForex - MetaTrader 5\terminal64.exe'):
    print("❌ Error al conectar:", mt5.last_error())
else:
    print("✅ Conexión exitosa")
    mt5.shutdown()