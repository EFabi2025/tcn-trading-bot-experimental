#!/usr/bin/env python3
"""
🔍 ETH DEBUG SIMPLE - Diagnóstico paso a paso
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from binance.client import Client as BinanceClient
from sklearn.preprocessing import StandardScaler
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def simple_eth_debug():
    """Diagnóstico simple paso a paso"""
    
    print("🔍 ETH DEBUG SIMPLE")
    print("="*50)
    
    # === PASO 1: DATOS ===
    print("\n📊 PASO 1: OBTENIENDO DATOS")
    client = BinanceClient()
    
    klines = client.get_historical_klines("ETHUSDT", "5m", limit=200)
    
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    df['close'] = pd.to_numeric(df['close'])
    print(f"✅ {len(df)} datos obtenidos")
    
    # === PASO 2: FEATURES SIMPLES ===
    print("\n📊 PASO 2: FEATURES SIMPLES")
    
    # Solo 3 features básicas
    df['returns'] = df['close'].pct_change()
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    
    df = df.dropna()
    features = ['returns', 'sma_5', 'sma_20']
    
    print(f"✅ Features: {features}")
    print(f"✅ Datos limpios: {len(df)}")
    
    # === PASO 3: LABELS SIMPLES ===
    print("\n📊 PASO 3: LABELS SIMPLES")
    
    threshold = 0.002  # 0.2%
    
    labels = []
    for i in range(len(df) - 3):
        current = df['close'].iloc[i]
        future = df['close'].iloc[i + 3]
        change = (future - current) / current
        
        if change < -threshold:
            labels.append(0)  # SELL
        elif change > threshold:
            labels.append(2)  # BUY
        else:
            labels.append(1)  # HOLD
    
    # Truncar DataFrame para que coincida con labels
    df = df.iloc[:-3].copy()
    df['label'] = labels
    
    label_counts = Counter(labels)
    print(f"✅ Labels: SELL={label_counts[0]}, HOLD={label_counts[1]}, BUY={label_counts[2]}")
    
    # === PASO 4: DATOS DE ENTRENAMIENTO ===
    print("\n📊 PASO 4: PREPARANDO DATOS")
    
    # Normalizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    y = np.array(labels)
    
    print(f"✅ X shape: {X_scaled.shape}")
    print(f"✅ y shape: {y.shape}")
    print(f"✅ y unique: {np.unique(y)}")
    
    # Verificar distribución
    print(f"✅ y distribution: {Counter(y)}")
    
    # === PASO 5: MODELO SUPER SIMPLE ===
    print("\n📊 PASO 5: MODELO ULTRA SIMPLE")
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(3,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Modelo: {model.count_params()} parámetros")
    
    # === PASO 6: ENTRENAR ===
    print("\n📊 PASO 6: ENTRENAMIENTO RÁPIDO")
    
    # Split simple
    split = int(len(X_scaled) * 0.8)
    X_train, X_val = X_scaled[:split], X_scaled[split:]
    y_train, y_val = y[:split], y[split:]
    
    print(f"✅ Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Entrenar pocas épocas
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=16,
        verbose=1
    )
    
    # === PASO 7: PREDICCIONES ===
    print("\n📊 PASO 7: VERIFICANDO PREDICCIONES")
    
    # Predicciones en validación
    pred_probs = model.predict(X_val, verbose=0)
    pred_classes = np.argmax(pred_probs, axis=1)
    
    print(f"✅ Predicciones shape: {pred_probs.shape}")
    print(f"✅ Clases predichas: {np.unique(pred_classes)}")
    
    # Distribución
    pred_counts = Counter(pred_classes)
    total = len(pred_classes)
    
    print("\n📊 DISTRIBUCIÓN FINAL:")
    class_names = ['SELL', 'HOLD', 'BUY']
    for i, name in enumerate(class_names):
        count = pred_counts[i]
        pct = count / total * 100
        print(f"   {name}: {count} ({pct:.1f}%)")
    
    # Bias score
    percentages = [pred_counts[i]/total for i in range(3)]
    bias_score = (max(percentages) - min(percentages)) * 10
    print(f"\n🎯 BIAS SCORE: {bias_score:.1f}/10")
    
    # === PASO 8: DIAGNÓSTICO PROFUNDO ===
    print("\n📊 PASO 8: DIAGNÓSTICO DETALLADO")
    
    # Verificar algunas predicciones individuales
    print("🔍 Primeras 10 predicciones:")
    for i in range(min(10, len(pred_probs))):
        probs = pred_probs[i]
        pred_class = pred_classes[i]
        true_class = y_val[i]
        print(f"   {i}: Pred={pred_class} Real={true_class} | Probs=[{probs[0]:.3f}, {probs[1]:.3f}, {probs[2]:.3f}]")
    
    # Verificar pesos del modelo
    weights = model.get_weights()
    print(f"\n🔍 Pesos capa final shape: {weights[-2].shape}")
    print(f"🔍 Bias capa final: {weights[-1]}")
    
    if bias_score < 5.0:
        print("\n🏆 ¡MODELO SIMPLE EXITOSO!")
        return True
    else:
        print("\n⚠️ Modelo simple también tiene bias")
        return False


if __name__ == "__main__":
    simple_eth_debug() 