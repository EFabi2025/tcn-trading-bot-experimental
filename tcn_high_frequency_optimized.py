#!/usr/bin/env python3
"""
TCN Optimizado para Datos de Alta Frecuencia - Trading Binance
Implementa técnicas avanzadas para manejo adecuado de secuencias temporales
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuración para Apple Silicon
tf.random.set_seed(42)
tf.config.experimental.enable_op_determinism()

class HighFrequencyTCN:
    """
    TCN especializado para trading de alta frecuencia con técnicas avanzadas
    """
    
    def __init__(self, config=None):
        self.config = config or self._get_default_config()
        self.model = None
        self.scalers = {}
        self.feature_importance = {}
        
    def _get_default_config(self):
        """Configuración optimizada para alta frecuencia"""
        return {
            # Arquitectura TCN optimizada
            'tcn_layers': 6,  # Reducido para evitar overfitting en alta frecuencia
            'filters': [32, 64, 64, 128, 128, 64],  # Progresión más suave
            'kernel_size': 3,  # Kernel pequeño para capturar patrones locales
            'dilations': [1, 2, 4, 8, 16, 32],  # Dilatación exponencial
            'dropout_rate': 0.4,  # Dropout más alto para alta frecuencia
            
            # Ventanas temporales específicas para alta frecuencia
            'short_window': 12,   # 1 hora (12 * 5min)
            'medium_window': 36,  # 3 horas (36 * 5min) 
            'long_window': 144,   # 12 horas (144 * 5min)
            
            # Parámetros de entrenamiento
            'batch_size': 32,  # Batch más pequeño para mejor generalización
            'learning_rate': 1e-4,  # Learning rate conservador
            'epochs': 100,
            'patience': 20,
            
            # Gestión de riesgo
            'max_position_size': 0.02,  # 2% máximo por operación
            'risk_threshold': 0.6,  # Umbral de confianza mínimo
        }
    
    def create_multi_timeframe_features(self, data):
        """
        Crea features multi-timeframe optimizados para alta frecuencia
        """
        features = pd.DataFrame(index=data.index)
        
        # Features básicos OHLCV
        features['returns_1'] = data['close'].pct_change(1)
        features['returns_3'] = data['close'].pct_change(3)
        features['returns_12'] = data['close'].pct_change(12)  # 1 hora
        
        # Volatilidad realizada en múltiples ventanas
        features['volatility_short'] = features['returns_1'].rolling(self.config['short_window']).std()
        features['volatility_medium'] = features['returns_1'].rolling(self.config['medium_window']).std()
        
        # Momentum multi-timeframe
        features['momentum_12'] = data['close'] / data['close'].shift(12) - 1  # 1h momentum
        features['momentum_36'] = data['close'] / data['close'].shift(36) - 1  # 3h momentum
        
        # RSI adaptativo para alta frecuencia
        features['rsi_14'] = self._calculate_rsi(data['close'], 14)
        features['rsi_21'] = self._calculate_rsi(data['close'], 21)
        
        # MACD optimizado para 5min
        features['macd'], features['macd_signal'] = self._calculate_macd(data['close'], 12, 26, 9)
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Bandas de Bollinger dinámicas
        bb_period = 20
        bb_std = 2
        sma = data['close'].rolling(bb_period).mean()
        std = data['close'].rolling(bb_period).std()
        features['bb_upper'] = sma + (std * bb_std)
        features['bb_lower'] = sma - (std * bb_std)
        features['bb_position'] = (data['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # Volumen relativo y perfil
        features['volume_sma'] = data['volume'].rolling(20).mean()
        features['volume_ratio'] = data['volume'] / features['volume_sma']
        features['volume_momentum'] = data['volume'].pct_change(3)
        
        # Price velocity (velocidad de cambio de precio)
        features['price_velocity'] = (data['close'] - data['close'].shift(5)) / 5
        features['price_acceleration'] = features['price_velocity'].diff()
        
        # Support/Resistance levels
        features['support'] = data['low'].rolling(24).min()  # Soporte 2h
        features['resistance'] = data['high'].rolling(24).max()  # Resistencia 2h
        features['sr_position'] = (data['close'] - features['support']) / (features['resistance'] - features['support'])
        
        return features.fillna(method='ffill').fillna(0)
    
    def _calculate_rsi(self, prices, period=14):
        """RSI optimizado para alta frecuencia"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """MACD adaptado para trading de 5 minutos"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line
    
    def create_temporal_sequences(self, features, sequence_length=60):
        """
        Crea secuencias temporales con técnicas anti-overfitting para alta frecuencia
        """
        print(f"Creando secuencias temporales de longitud {sequence_length}...")
        
        # Normalización robusta por feature
        normalized_features = features.copy()
        for col in features.columns:
            scaler = RobustScaler()  # Más robusto que StandardScaler para datos financieros
            normalized_features[col] = scaler.fit_transform(features[col].values.reshape(-1, 1)).flatten()
            self.scalers[col] = scaler
        
        # Crear secuencias con overlapping reducido para evitar autocorrelación
        sequences = []
        targets = []
        
        # Step size mayor para reducir correlación entre muestras
        step_size = max(1, sequence_length // 4)  # 25% de overlap máximo
        
        for i in range(sequence_length, len(normalized_features) - 1, step_size):
            # Secuencia de features
            seq = normalized_features.iloc[i-sequence_length:i].values
            
            # Target: siguiente movimiento significativo
            current_price = features.iloc[i]['returns_1'] if 'returns_1' in features.columns else 0
            future_price = features.iloc[i+1]['returns_1'] if 'returns_1' in features.columns else 0
            
            # Clasificación con umbrales dinámicos basados en volatilidad
            volatility = features.iloc[i]['volatility_short'] if 'volatility_short' in features.columns else 0.01
            
            # Umbrales adaptativos
            buy_threshold = 0.3 * volatility
            sell_threshold = -0.3 * volatility
            
            if future_price > buy_threshold:
                target = 2  # BUY
            elif future_price < sell_threshold:
                target = 0  # SELL
            else:
                target = 1  # HOLD
            
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def build_high_frequency_tcn(self, input_shape, num_classes=3):
        """
        Construye TCN especializado para datos de alta frecuencia
        """
        print("Construyendo TCN para alta frecuencia...")
        
        # Input layer
        inputs = layers.Input(shape=input_shape, name='temporal_input')
        
        # Normalization layer para estabilidad
        x = layers.LayerNormalization()(inputs)
        
        # TCN Blocks con residual connections
        x = self._tcn_block(x, self.config['filters'][0], self.config['kernel_size'], 
                           self.config['dilations'][0], dropout=self.config['dropout_rate'])
        
        # Residual connections para gradientes estables
        for i in range(1, self.config['tcn_layers']):
            residual = x
            x = self._tcn_block(x, self.config['filters'][i], self.config['kernel_size'],
                               self.config['dilations'][i], dropout=self.config['dropout_rate'])
            
            # Residual connection si las dimensiones coinciden
            if x.shape[-1] == residual.shape[-1]:
                x = layers.Add()([x, residual])
        
        # Global features extraction
        # Usar tanto la última posición como un pooling global
        last_timestep = x[:, -1, :]
        global_max_pool = layers.GlobalMaxPooling1D()(x)
        global_avg_pool = layers.GlobalAveragePooling1D()(x)
        
        # Combinar diferentes representaciones
        combined = layers.Concatenate()([last_timestep, global_max_pool, global_avg_pool])
        
        # Dense layers con regularización
        x = layers.Dense(128, activation='mish', 
                        kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                        name='dense_1')(combined)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(64, activation='mish',
                        kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                        name='dense_2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer con softmax
        outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
        
        # Crear modelo
        model = models.Model(inputs=inputs, outputs=outputs, name='HighFrequency_TCN')
        
        # Compilar con optimizer optimizado para series temporales
        optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=self.config['learning_rate'],
            clipnorm=1.0  # Gradient clipping
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _tcn_block(self, x, filters, kernel_size, dilation_rate, dropout=0.2):
        """
        Bloque TCN optimizado con técnicas anti-overfitting
        """
        # Convolución causal con dilatación
        conv1 = layers.Conv1D(filters, kernel_size, dilation_rate=dilation_rate,
                             padding='causal', activation='mish')(x)
        conv1 = layers.LayerNormalization()(conv1)
        conv1 = layers.SpatialDropout1D(dropout)(conv1)
        
        # Segunda convolución
        conv2 = layers.Conv1D(filters, kernel_size, dilation_rate=dilation_rate,
                             padding='causal', activation='mish')(conv1)
        conv2 = layers.LayerNormalization()(conv2)
        conv2 = layers.SpatialDropout1D(dropout)(conv2)
        
        # Projection si es necesario
        if x.shape[-1] != filters:
            x = layers.Conv1D(filters, 1, padding='same')(x)
        
        # Residual connection
        output = layers.Add()([x, conv2])
        return output
    
    def walk_forward_validation(self, sequences, targets, n_splits=5):
        """
        Validación walk-forward específica para series temporales de alta frecuencia
        """
        print(f"Iniciando validación walk-forward con {n_splits} splits...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        results = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(sequences)):
            print(f"\nFold {fold + 1}/{n_splits}")
            
            # Dividir datos
            X_train, X_val = sequences[train_idx], sequences[val_idx]
            y_train, y_val = targets[train_idx], targets[val_idx]
            
            # Construir modelo para este fold
            model = self.build_high_frequency_tcn(X_train.shape[1:])
            
            # Callbacks específicos para alta frecuencia
            callbacks_list = [
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.config['patience'],
                    restore_best_weights=True,
                    min_delta=1e-4
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.7,
                    patience=10,
                    min_lr=1e-7
                ),
                callbacks.ModelCheckpoint(
                    f'model_fold_{fold}.h5',
                    save_best_only=True,
                    monitor='val_accuracy'
                )
            ]
            
            # Entrenar
            history = model.fit(
                X_train, y_train,
                batch_size=self.config['batch_size'],
                epochs=self.config['epochs'],
                validation_data=(X_val, y_val),
                callbacks=callbacks_list,
                verbose=1
            )
            
            # Evaluar
            predictions = model.predict(X_val)
            predicted_classes = np.argmax(predictions, axis=1)
            
            # Métricas del fold
            fold_results = {
                'fold': fold + 1,
                'val_accuracy': max(history.history['val_accuracy']),
                'val_loss': min(history.history['val_loss']),
                'predictions': predicted_classes,
                'true_labels': y_val,
                'confidence_scores': np.max(predictions, axis=1)
            }
            
            results.append(fold_results)
            
            # Análisis de distribución
            unique, counts = np.unique(predicted_classes, return_counts=True)
            distribution = dict(zip(unique, counts / len(predicted_classes) * 100))
            print(f"Distribución de predicciones: {distribution}")
            
        return results
    
    def calculate_trading_metrics(self, results):
        """
        Calcula métricas específicas para trading de alta frecuencia
        """
        print("\n=== MÉTRICAS DE TRADING ===")
        
        all_predictions = np.concatenate([r['predictions'] for r in results])
        all_true_labels = np.concatenate([r['true_labels'] for r in results])
        all_confidences = np.concatenate([r['confidence_scores'] for r in results])
        
        # Distribución de clases
        class_names = ['SELL', 'HOLD', 'BUY']
        unique, counts = np.unique(all_predictions, return_counts=True)
        total_predictions = len(all_predictions)
        
        print("\n--- Distribución de Señales ---")
        for i, class_name in enumerate(class_names):
            if i in unique:
                idx = list(unique).index(i)
                percentage = counts[idx] / total_predictions * 100
                print(f"{class_name}: {counts[idx]} ({percentage:.1f}%)")
            else:
                print(f"{class_name}: 0 (0.0%)")
        
        # Bias Score (0-10, donde 5 es perfecto balance)
        sell_pct = (counts[list(unique).index(0)] if 0 in unique else 0) / total_predictions
        hold_pct = (counts[list(unique).index(1)] if 1 in unique else 0) / total_predictions  
        buy_pct = (counts[list(unique).index(2)] if 2 in unique else 0) / total_predictions
        
        # Calcular desviación del balance perfecto (33.33% cada clase)
        target_pct = 1/3
        bias_score = 10 * (1 - (abs(sell_pct - target_pct) + abs(hold_pct - target_pct) + abs(buy_pct - target_pct)) / 2)
        
        print(f"\n--- Métricas de Trading ---")
        print(f"Bias Score: {bias_score:.1f}/10 (5.0 = balance perfecto)")
        print(f"Confianza promedio: {np.mean(all_confidences):.3f}")
        print(f"Confianza mínima: {np.min(all_confidences):.3f}")
        print(f"Confianza máxima: {np.max(all_confidences):.3f}")
        
        # Accuracy por clase
        from sklearn.metrics import classification_report
        report = classification_report(all_true_labels, all_predictions, 
                                     target_names=class_names, output_dict=True)
        
        print(f"\n--- Accuracy por Clase ---")
        for i, class_name in enumerate(class_names):
            if str(i) in report:
                acc = report[str(i)]['recall']  # Recall es accuracy por clase
                print(f"{class_name}: {acc:.3f}")
        
        # Evaluación de requisitos de trading
        min_accuracy = 0.4
        max_bias_score = 5.0
        min_confidence = 0.6
        
        trading_ready = True
        issues = []
        
        if bias_score > max_bias_score:
            trading_ready = False
            issues.append(f"Bias score muy alto: {bias_score:.1f} > {max_bias_score}")
        
        if np.mean(all_confidences) < min_confidence:
            trading_ready = False
            issues.append(f"Confianza baja: {np.mean(all_confidences):.3f} < {min_confidence}")
        
        for i, class_name in enumerate(class_names):
            if str(i) in report and report[str(i)]['recall'] < min_accuracy:
                trading_ready = False
                issues.append(f"Accuracy {class_name} baja: {report[str(i)]['recall']:.3f} < {min_accuracy}")
        
        print(f"\n--- Evaluación para Trading Real ---")
        print(f"Trading Ready: {'✅ SÍ' if trading_ready else '❌ NO'}")
        if issues:
            print("Problemas detectados:")
            for issue in issues:
                print(f"  • {issue}")
        
        return {
            'bias_score': bias_score,
            'trading_ready': trading_ready,
            'confidence_mean': np.mean(all_confidences),
            'class_accuracies': {class_names[i]: report[str(i)]['recall'] if str(i) in report else 0.0 
                               for i in range(len(class_names))},
            'issues': issues
        }

def main():
    """
    Función principal para entrenar y evaluar el TCN de alta frecuencia
    """
    print("=== TCN OPTIMIZADO PARA ALTA FRECUENCIA ===")
    print("Especializando en datos de trading de 5 minutos\n")
    
    # Simular datos de alta frecuencia (normalmente vendrían de Binance API)
    print("Generando datos de prueba de alta frecuencia...")
    n_samples = 5000  # ~17 días de datos de 5min
    
    # Simular datos OHLCV realistas
    np.random.seed(42)
    base_price = 2000
    returns = np.random.normal(0, 0.003, n_samples)  # 0.3% volatilidad por período
    returns = np.cumsum(returns)
    
    data = pd.DataFrame({
        'open': base_price * np.exp(returns + np.random.normal(0, 0.001, n_samples)),
        'high': base_price * np.exp(returns + np.abs(np.random.normal(0, 0.002, n_samples))),
        'low': base_price * np.exp(returns - np.abs(np.random.normal(0, 0.002, n_samples))),
        'close': base_price * np.exp(returns),
        'volume': np.random.lognormal(10, 0.5, n_samples)
    })
    
    # Asegurar coherencia OHLC
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    # Crear instancia del modelo
    config = {
        'tcn_layers': 6,
        'filters': [32, 64, 64, 128, 128, 64],
        'kernel_size': 3,
        'dilations': [1, 2, 4, 8, 16, 32],
        'dropout_rate': 0.4,
        'short_window': 12,
        'medium_window': 36,
        'long_window': 144,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'epochs': 50,  # Reducido para demo
        'patience': 15,
    }
    
    tcn = HighFrequencyTCN(config)
    
    # Crear features multi-timeframe
    print("Generando features multi-timeframe...")
    features = tcn.create_multi_timeframe_features(data)
    print(f"Features generados: {list(features.columns)}")
    print(f"Shape de features: {features.shape}")
    
    # Crear secuencias temporales optimizadas
    sequence_length = 60  # 5 horas de datos
    sequences, targets = tcn.create_temporal_sequences(features, sequence_length)
    
    print(f"\nSecuencias creadas:")
    print(f"Shape de secuencias: {sequences.shape}")
    print(f"Shape de targets: {targets.shape}")
    
    # Distribución de clases inicial
    unique, counts = np.unique(targets, return_counts=True)
    print(f"\nDistribución de clases:")
    class_names = ['SELL', 'HOLD', 'BUY']
    for i, count in zip(unique, counts):
        print(f"{class_names[i]}: {count} ({count/len(targets)*100:.1f}%)")
    
    # Validación walk-forward
    print("\nIniciando validación walk-forward...")
    results = tcn.walk_forward_validation(sequences, targets, n_splits=3)
    
    # Calcular métricas de trading
    metrics = tcn.calculate_trading_metrics(results)
    
    print(f"\n=== RESUMEN FINAL ===")
    print(f"Modelo optimizado para alta frecuencia completado")
    print(f"Técnicas implementadas:")
    print(f"  ✅ Ventanas temporales multi-timeframe")
    print(f"  ✅ Normalización robusta por feature")
    print(f"  ✅ Umbrales adaptativos basados en volatilidad")
    print(f"  ✅ Arquitectura TCN con residual connections")
    print(f"  ✅ Validación walk-forward temporal")
    print(f"  ✅ Regularización específica para alta frecuencia")
    
    if metrics['trading_ready']:
        print(f"\n🎯 MODELO LISTO PARA TRADING!")
    else:
        print(f"\n⚠️  MODELO REQUIERE OPTIMIZACIÓN")

if __name__ == "__main__":
    main() 