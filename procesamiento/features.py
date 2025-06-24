import pandas as pd
import numpy as np


def calcular_rsi(df, periodo=14):
    delta = df['Close'].diff()
    ganancia = np.where(delta > 0, delta, 0)
    perdida = np.where(delta < 0, -delta, 0)
    
    ganancia_prom = pd.Series(ganancia).rolling(window=periodo).mean()
    perdida_prom = pd.Series(perdida).rolling(window=periodo).mean()

    rs = ganancia_prom / perdida_prom
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi
    return df


def calcular_macd(df, rapida=12, lenta=26, signal=9):
    ema_rapida = df['Close'].ewm(span=rapida, adjust=False).mean()
    ema_lenta = df['Close'].ewm(span=lenta, adjust=False).mean()
    df['MACD'] = ema_rapida - ema_lenta
    df['Signal_Line'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    return df


def calcular_retornos_log(df):
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    return df


def calcular_atr(df, periodo=14):
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = np.abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = np.abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=periodo).mean()
    df.drop(['H-L', 'H-PC', 'L-PC', 'TR'], axis=1, inplace=True)
    return df


def calcular_bollinger(df, periodo=20, num_std=2):
    media = df['Close'].rolling(window=periodo).mean()
    std = df['Close'].rolling(window=periodo).std()
    df['BB_Media'] = media
    df['BB_Upper'] = media + num_std * std
    df['BB_Lower'] = media - num_std * std
    return df


def calcular_momentum(df, periodo=10):
    df['Momentum'] = df['Close'] - df['Close'].shift(periodo)
    return df


def calcular_sma(df, periodo=20):
    df[f'SMA_{periodo}'] = df['Close'].rolling(window=periodo).mean()
    return df


def calcular_ema(df, periodo=20):
    df[f'EMA_{periodo}'] = df['Close'].ewm(span=periodo, adjust=False).mean()
    return df


def agregar_volumen(df):
    df['Volumen_normalizado'] = df['tick_volume'] / df['tick_volume'].rolling(window=20).mean()
    return df


def aplicar_todos_los_indicadores(df):
    df = calcular_rsi(df)
    df = calcular_macd(df)
    df = calcular_retornos_log(df)
    df = calcular_atr(df)
    df = calcular_bollinger(df)
    df = calcular_momentum(df)
    df = calcular_sma(df)
    df = calcular_ema(df)
    df = agregar_volumen(df)
    return df
