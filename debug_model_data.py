#!/usr/bin/env python3
"""
🔍 DEBUG MODEL DATA - Diagnóstico completo de datos de entrada

Verifica paso a paso dónde se origina el problema de NaN
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pickle

def check_data_integrity():
    """Verificar integridad de datos paso a paso"""
    print("🔍 DIAGNÓSTICO DE DATOS - SUPER BALANCED TCN")
    print("="*60)
    
    # Cargar scaler
    try:
        with open('models/super_feature_scalers.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("✅ Scaler cargado correctamente")
    except Exception as e:
        print(f"❌ Error cargando scaler: {e}")
        return
    
    # Simular datos como en el entrenamiento
    from binance.client import Client as BinanceClient
    client = BinanceClient()
    
    print("\n📊 1. VERIFICANDO DATOS RAW...")
    klines = client.get_historical_klines("BTCUSDT", "5m", limit=100)
    
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"✅ Datos OHLCV shape: {df.shape}")
    print(f"✅ Rango de precios: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print(f"✅ NaN en datos raw: {df.isnull().sum().sum()}")
    
    # Verificar features técnicas
    print("\n🔧 2. VERIFICANDO FEATURES TÉCNICAS...")
    
    # SMAs
    for period in [5, 7, 10, 14, 20]:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
    
    # EMAs
    for period in [5, 9, 12, 21, 26]:
        df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi_14'] = 100 - (100 / (1 + gain / loss))
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    df['macd_normalized'] = df['macd'] / df['close']
    df['macd_signal_normalized'] = df['macd_signal'] / df['close']
    df['macd_histogram_normalized'] = df['macd_histogram'] / df['close']
    
    # Volatilidad
    df['volatility_5'] = df['close'].pct_change().rolling(5).std()
    
    # Change features
    df['close_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    
    df = df.dropna()
    
    feature_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'sma_5', 'sma_7', 'sma_10', 'sma_14', 'sma_20',
        'ema_5', 'ema_9', 'ema_12', 'ema_21', 'ema_26',
        'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
        'macd_normalized', 'macd_signal_normalized', 'macd_histogram_normalized',
        'volatility_5', 'close_change', 'volume_change'
    ]
    
    print(f"✅ Features shape: {df[feature_cols].shape}")
    print(f"❌ NaN en features: {df[feature_cols].isnull().sum().sum()}")
    print(f"❌ Inf en features: {np.isinf(df[feature_cols]).sum().sum()}")
    
    # Verificar valores extremos
    print("\n⚠️ 3. VERIFICANDO VALORES EXTREMOS...")
    for col in feature_cols:
        data = df[col]
        min_val = data.min()
        max_val = data.max()
        if abs(min_val) > 1e6 or abs(max_val) > 1e6:
            print(f"⚠️ {col}: min={min_val:.6f}, max={max_val:.6f}")
        
        # Verificar NaN/Inf específicos
        if data.isnull().sum() > 0:
            print(f"❌ {col}: {data.isnull().sum()} NaN values")
        if np.isinf(data).sum() > 0:
            print(f"❌ {col}: {np.isinf(data).sum()} Inf values")
    
    # Verificar normalización
    print("\n📊 4. VERIFICANDO NORMALIZACIÓN...")
    
    df_clean = df[feature_cols].dropna()
    df_clean = df_clean.replace([np.inf, -np.inf], 0)
    
    try:
        scaler_test = MinMaxScaler()
        scaled_data = scaler_test.fit_transform(df_clean)
        
        print(f"✅ Normalización exitosa: {scaled_data.shape}")
        print(f"✅ Rango normalizado: {scaled_data.min():.6f} - {scaled_data.max():.6f}")
        print(f"❌ NaN en datos normalizados: {np.isnan(scaled_data).sum()}")
        print(f"❌ Inf en datos normalizados: {np.isinf(scaled_data).sum()}")
        
    except Exception as e:
        print(f"❌ Error en normalización: {e}")
    
    # Verificar labels
    print("\n🎯 5. VERIFICANDO LABELS...")
    
    df['future_price'] = df['close'].shift(-5)
    df['price_change'] = (df['future_price'] - df['close']) / df['close']
    
    labels = []
    for change in df['price_change']:
        if pd.isna(change):
            labels.append(1)  # HOLD
        elif change < -0.005:  # -0.5%
            labels.append(0)  # SELL
        elif change > 0.005:   # +0.5%
            labels.append(2)  # BUY
        else:
            labels.append(1)  # HOLD
    
    df['label'] = labels
    unique_labels = np.unique([l for l in labels if not pd.isna(l)])
    
    print(f"✅ Labels únicas: {unique_labels}")
    print(f"✅ Distribución de labels: {pd.Series(labels).value_counts()}")
    
    # Test del modelo simplificado
    print("\n🧠 6. TEST DE MODELO SIMPLIFICADO...")
    
    # Crear modelo básico para test
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(25,)),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Crear datos de test pequeños
    X_test = scaled_data[:50]
    y_test = np.array(labels[:50])
    
    try:
        # Test forward pass
        predictions = model.predict(X_test, verbose=0)
        print(f"✅ Predicciones shape: {predictions.shape}")
        print(f"✅ Predicciones rango: {predictions.min():.6f} - {predictions.max():.6f}")
        print(f"❌ NaN en predicciones: {np.isnan(predictions).sum()}")
        
        # Test loss calculation
        loss_val = model.evaluate(X_test, y_test, verbose=0)
        print(f"✅ Loss value: {loss_val}")
        
    except Exception as e:
        print(f"❌ Error en modelo test: {e}")
    
    print("\n" + "="*60)
    print("🎯 DIAGNÓSTICO COMPLETADO")


if __name__ == "__main__":
    check_data_integrity() 