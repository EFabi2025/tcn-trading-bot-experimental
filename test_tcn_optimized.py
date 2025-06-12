#!/usr/bin/env python3
"""
Test del TCN Optimizado para Alta Frecuencia
Versión simplificada para verificar mejoras implementadas
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Configuración para Apple Silicon
tf.random.set_seed(42)
np.random.seed(42)

def create_optimized_features(data):
    """
    Crea features multi-timeframe optimizados para alta frecuencia
    """
    features = pd.DataFrame(index=data.index)
    
    # Returns multi-timeframe
    features['returns_1'] = data['close'].pct_change(1)
    features['returns_3'] = data['close'].pct_change(3)
    features['returns_12'] = data['close'].pct_change(12)  # 1 hora
    
    # Volatilidad realizada
    features['volatility_12'] = features['returns_1'].rolling(12).std()
    features['volatility_36'] = features['returns_1'].rolling(36).std()
    
    # Momentum
    features['momentum_12'] = data['close'] / data['close'].shift(12) - 1
    features['momentum_36'] = data['close'] / data['close'].shift(36) - 1
    
    # RSI adaptativo
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs))
    
    # Bandas de Bollinger
    sma = data['close'].rolling(20).mean()
    std = data['close'].rolling(20).std()
    features['bb_position'] = (data['close'] - (sma - 2*std)) / (4*std)
    
    # Volumen
    features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
    
    # Price velocity
    features['price_velocity'] = (data['close'] - data['close'].shift(5)) / 5
    
    return features.fillna(method='ffill').fillna(0)

def create_high_freq_sequences(features, sequence_length=30):
    """
    Crea secuencias optimizadas para alta frecuencia con técnicas anti-overfitting
    """
    print(f"Creando secuencias temporales optimizadas (longitud: {sequence_length})...")
    
    # Normalización robusta
    scalers = {}
    normalized_features = features.copy()
    for col in features.columns:
        scaler = RobustScaler()
        normalized_features[col] = scaler.fit_transform(features[col].values.reshape(-1, 1)).flatten()
        scalers[col] = scaler
    
    sequences = []
    targets = []
    
    # Step size mayor para reducir autocorrelación (50% overlap máximo)
    step_size = max(1, sequence_length // 2)
    
    for i in range(sequence_length, len(normalized_features) - 1, step_size):
        # Secuencia
        seq = normalized_features.iloc[i-sequence_length:i].values
        
        # Target con umbrales adaptativos basados en volatilidad
        future_return = features.iloc[i+1]['returns_1'] if 'returns_1' in features.columns else 0
        volatility = features.iloc[i]['volatility_12'] if 'volatility_12' in features.columns else 0.01
        
        # Umbrales adaptativos (más conservadores)
        buy_threshold = 0.5 * volatility
        sell_threshold = -0.5 * volatility
        
        if future_return > buy_threshold:
            target = 2  # BUY
        elif future_return < sell_threshold:
            target = 0  # SELL
        else:
            target = 1  # HOLD
        
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets), scalers

def build_simple_tcn(input_shape, num_classes=3):
    """
    TCN simplificado pero optimizado para alta frecuencia
    """
    print("Construyendo TCN simplificado...")
    
    inputs = layers.Input(shape=input_shape)
    
    # Normalization
    x = layers.LayerNormalization()(inputs)
    
    # TCN con menos capas pero más eficientes
    x = layers.Conv1D(32, 3, dilation_rate=1, padding='causal', activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv1D(64, 3, dilation_rate=2, padding='causal', activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv1D(64, 3, dilation_rate=4, padding='causal', activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv1D(32, 3, dilation_rate=8, padding='causal', activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Global pooling + last timestep
    last_timestep = x[:, -1, :]
    global_pool = layers.GlobalAveragePooling1D()(x)
    combined = layers.Concatenate()([last_timestep, global_pool])
    
    # Dense layers
    x = layers.Dense(64, activation='relu', 
                    kernel_regularizer=tf.keras.regularizers.l2(1e-4))(combined)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def evaluate_model(sequences, targets):
    """
    Evalúa el modelo con validación temporal
    """
    print(f"\n=== EVALUACIÓN DEL MODELO ===")
    print(f"Datos: {sequences.shape[0]} secuencias, {sequences.shape[1]} timesteps, {sequences.shape[2]} features")
    
    # Distribución original
    unique, counts = np.unique(targets, return_counts=True)
    class_names = ['SELL', 'HOLD', 'BUY']
    print(f"\nDistribución original:")
    for i, count in zip(unique, counts):
        print(f"  {class_names[i]}: {count} ({count/len(targets)*100:.1f}%)")
    
    # Split temporal simple
    split_point = int(0.8 * len(sequences))
    X_train, X_test = sequences[:split_point], sequences[split_point:]
    y_train, y_test = targets[:split_point], targets[split_point:]
    
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    # Construir y entrenar modelo
    model = build_simple_tcn(X_train.shape[1:])
    
    print(f"\nEntrenando modelo...")
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=16,
        validation_split=0.2,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
        ]
    )
    
    # Evaluar
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    confidence_scores = np.max(predictions, axis=1)
    
    return predicted_classes, confidence_scores, y_test

def calculate_advanced_metrics(predictions, true_labels, confidences):
    """
    Calcula métricas avanzadas de trading
    """
    print(f"\n=== MÉTRICAS AVANZADAS ===")
    
    class_names = ['SELL', 'HOLD', 'BUY']
    unique, counts = np.unique(predictions, return_counts=True)
    
    print(f"\nDistribución de predicciones:")
    for i, class_name in enumerate(class_names):
        if i in unique:
            idx = list(unique).index(i)
            percentage = counts[idx] / len(predictions) * 100
            print(f"  {class_name}: {counts[idx]} ({percentage:.1f}%)")
        else:
            print(f"  {class_name}: 0 (0.0%)")
    
    # Bias Score
    sell_pct = (counts[list(unique).index(0)] if 0 in unique else 0) / len(predictions)
    hold_pct = (counts[list(unique).index(1)] if 1 in unique else 0) / len(predictions)  
    buy_pct = (counts[list(unique).index(2)] if 2 in unique else 0) / len(predictions)
    
    target_pct = 1/3
    bias_score = 10 * (1 - (abs(sell_pct - target_pct) + abs(hold_pct - target_pct) + abs(buy_pct - target_pct)) / 2)
    
    print(f"\nBias Score: {bias_score:.1f}/10 (5.0 = balance perfecto)")
    print(f"Confianza promedio: {np.mean(confidences):.3f}")
    
    # Accuracy por clase
    from sklearn.metrics import classification_report
    report = classification_report(true_labels, predictions, target_names=class_names, output_dict=True)
    
    print(f"\nAccuracy por clase:")
    trading_ready = True
    for i, class_name in enumerate(class_names):
        if str(i) in report:
            acc = report[str(i)]['recall']
            print(f"  {class_name}: {acc:.3f}")
            if acc < 0.4:
                trading_ready = False
        else:
            print(f"  {class_name}: 0.000")
            trading_ready = False
    
    # Evaluación final
    if bias_score > 7.0 or np.mean(confidences) < 0.5:
        trading_ready = False
    
    print(f"\n{'🎯 MODELO TRADING-READY!' if trading_ready else '⚠️  REQUIERE OPTIMIZACIÓN'}")
    
    return {
        'bias_score': bias_score,
        'confidence': np.mean(confidences),
        'trading_ready': trading_ready
    }

def main():
    """
    Test principal del TCN optimizado
    """
    print("=== TEST TCN OPTIMIZADO PARA ALTA FRECUENCIA ===")
    print("Implementando mejores prácticas para secuencias temporales\n")
    
    # Generar datos realistas
    n_samples = 2000
    np.random.seed(42)
    
    base_price = 2000
    returns = np.random.normal(0, 0.005, n_samples)  # 0.5% volatilidad
    returns = np.cumsum(returns)
    
    data = pd.DataFrame({
        'close': base_price * np.exp(returns),
        'volume': np.random.lognormal(10, 0.3, n_samples)
    })
    
    print(f"Datos generados: {len(data)} muestras")
    print(f"Precio inicial: ${data['close'].iloc[0]:.2f}")
    print(f"Precio final: ${data['close'].iloc[-1]:.2f}")
    print(f"Volatilidad: {data['close'].pct_change().std():.4f}")
    
    # Crear features optimizados
    features = create_optimized_features(data)
    print(f"\nFeatures creados: {list(features.columns)}")
    
    # Crear secuencias con técnicas anti-overfitting
    sequences, targets, scalers = create_high_freq_sequences(features, sequence_length=24)  # 2 horas
    
    # Evaluar modelo
    predictions, confidences, true_labels = evaluate_model(sequences, targets)
    
    # Métricas avanzadas
    metrics = calculate_advanced_metrics(predictions, true_labels, confidences)
    
    print(f"\n=== TÉCNICAS IMPLEMENTADAS ===")
    print(f"✅ Ventanas temporales multi-timeframe")
    print(f"✅ Normalización robusta (RobustScaler)")
    print(f"✅ Step size optimizado (50% overlap máximo)")
    print(f"✅ Umbrales adaptativos basados en volatilidad")
    print(f"✅ Arquitectura TCN con dilatación progresiva")
    print(f"✅ Global pooling + last timestep")
    print(f"✅ Regularización L2 y Dropout")
    print(f"✅ Validación temporal (no random split)")
    
    return metrics

if __name__ == "__main__":
    main() 