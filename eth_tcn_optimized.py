#!/usr/bin/env python3
"""
🔷 ETH TCN OPTIMIZED - Modelo ETH con threshold ajustado
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from binance.client import Client as BinanceClient
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import pickle
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EthereumTCNOptimized:
    """Modelo TCN optimizado para Ethereum con threshold dinámico"""
    
    def __init__(self):
        self.pair_symbol = "ETHUSDT"
        self.pair_name = "Ethereum"
        self.lookback_window = 60
        self.expected_features = 30
        self.class_names = ['SELL', 'HOLD', 'BUY']
        
        print("🔷 ETHEREUM TCN OPTIMIZED")
        print("="*50)
        print(f"📊 Par: {self.pair_symbol}")
        print(f"📈 Features: {self.expected_features}")
        print(f"⏰ Ventana: {self.lookback_window} períodos")
    
    def connect_and_get_data(self):
        """Conectar y obtener datos"""
        try:
            print("\n📋 CONECTANDO Y OBTENIENDO DATOS")
            print("-" * 40)
            
            self.client = BinanceClient()
            
            # Test conectividad
            ticker = self.client.get_symbol_ticker(symbol=self.pair_symbol)
            price = float(ticker['price'])
            print(f"✅ Conectado - Precio ETH: ${price:,.2f}")
            
            # Obtener datos
            klines = self.client.get_historical_klines(
                symbol=self.pair_symbol,
                interval='5m',
                limit=800
            )
            
            # Crear DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()
            print(f"✅ {len(df)} períodos obtenidos")
            
            return df
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def calculate_optimal_threshold(self, df):
        """Calcular threshold óptimo para ETH"""
        try:
            print("\n📋 CALCULANDO THRESHOLD ÓPTIMO")
            print("-" * 40)
            
            # Calcular volatilidades históricas
            price_changes = []
            for i in range(len(df) - 6):
                current = df['close'].iloc[i]
                future = df['close'].iloc[i + 6]
                change = abs((future - current) / current)
                price_changes.append(change)
            
            price_changes = np.array(price_changes)
            
            # Estadísticas
            mean_change = np.mean(price_changes)
            std_change = np.std(price_changes)
            percentile_25 = np.percentile(price_changes, 25)
            percentile_75 = np.percentile(price_changes, 75)
            
            print(f"📊 Análisis de volatilidad ETH:")
            print(f"   - Mean: {mean_change*100:.3f}%")
            print(f"   - Std: {std_change*100:.3f}%")
            print(f"   - P25: {percentile_25*100:.3f}%")
            print(f"   - P75: {percentile_75*100:.3f}%")
            
            # Probar diferentes thresholds
            thresholds = [0.002, 0.003, 0.004, 0.005, 0.006]  # 0.2% a 0.6%
            best_threshold = 0.004
            best_balance = 10.0
            
            print("\n🎯 Probando thresholds:")
            for threshold in thresholds:
                labels = []
                for i in range(len(df)):
                    if i >= len(df) - 6:
                        labels.append(1)
                        continue
                    
                    current_price = df['close'].iloc[i]
                    future_price = df['close'].iloc[i + 6]
                    price_change = (future_price - current_price) / current_price
                    
                    if price_change < -threshold:
                        labels.append(0)  # SELL
                    elif price_change > threshold:
                        labels.append(2)  # BUY
                    else:
                        labels.append(1)  # HOLD
                
                # Calcular balance
                counts = Counter(labels)
                total = len(labels)
                percentages = [counts[i]/total for i in range(3)]
                bias_score = (max(percentages) - min(percentages)) * 10
                
                print(f"   {threshold*100:.1f}%: SELL={counts[0]} HOLD={counts[1]} BUY={counts[2]} | Bias={bias_score:.1f}")
                
                if bias_score < best_balance:
                    best_balance = bias_score
                    best_threshold = threshold
            
            print(f"\n🏆 THRESHOLD ÓPTIMO: {best_threshold*100:.1f}% (Bias: {best_balance:.1f})")
            self.volatility_threshold = best_threshold
            
            return best_threshold
            
        except Exception as e:
            print(f"❌ Error calculando threshold: {e}")
            return 0.004
    
    def create_features_and_labels(self, df):
        """Crear features y labels optimizados"""
        try:
            print("\n📋 CREANDO FEATURES Y LABELS")
            print("-" * 40)
            
            df = df.copy()
            features = []
            
            # === FEATURES BÁSICAS (5) ===
            features.extend(['open', 'high', 'low', 'close', 'volume'])
            
            # === MOVING AVERAGES (5) ===
            for period in [5, 10, 20, 50, 100]:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                features.append(f'sma_{period}')
            
            # === EXPONENTIAL MOVING AVERAGES (5) ===
            for period in [5, 12, 26, 50, 100]:
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                features.append(f'ema_{period}')
            
            # === RSI (3) ===
            for period in [9, 14, 21]:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
                features.append(f'rsi_{period}')
            
            # === MACD (3) ===
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            features.extend(['macd', 'macd_signal', 'macd_histogram'])
            
            # === BOLLINGER BANDS (3) ===
            bb_period = 20
            bb_middle = df['close'].rolling(bb_period).mean()
            bb_std = df['close'].rolling(bb_period).std()
            df['bb_upper'] = bb_middle + (bb_std * 2)
            df['bb_lower'] = bb_middle - (bb_std * 2)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            features.extend(['bb_upper', 'bb_lower', 'bb_position'])
            
            # === VOLATILIDAD (3) ===
            for period in [5, 10, 20]:
                df[f'volatility_{period}'] = df['close'].pct_change().rolling(window=period).std()
                features.append(f'volatility_{period}')
            
            # === VOLUME FEATURES (3) ===
            for period in [10, 20, 50]:
                df[f'volume_ma_{period}'] = df['volume'].rolling(window=period).mean()
                features.append(f'volume_ma_{period}')
            
            # Exactamente 30 features
            features = features[:30]
            df = df.dropna()
            
            # Crear labels con threshold optimizado
            print(f"🎯 Creando labels con threshold: {self.volatility_threshold*100:.1f}%")
            
            labels = []
            for i in range(len(df)):
                if i >= len(df) - 6:
                    labels.append(1)  # HOLD
                    continue
                
                current_price = df['close'].iloc[i]
                future_price = df['close'].iloc[i + 6]
                price_change = (future_price - current_price) / current_price
                
                if price_change < -self.volatility_threshold:
                    labels.append(0)  # SELL
                elif price_change > self.volatility_threshold:
                    labels.append(2)  # BUY
                else:
                    labels.append(1)  # HOLD
            
            df['label'] = labels
            
            # Verificar distribución final
            label_counts = Counter(labels)
            total = len(labels)
            
            print("📊 Distribución final de labels:")
            for i, name in enumerate(self.class_names):
                count = label_counts[i]
                pct = count / total * 100
                print(f"   - {name}: {count} ({pct:.1f}%)")
            
            percentages = [label_counts[i]/total for i in range(3)]
            bias_score = (max(percentages) - min(percentages)) * 10
            print(f"📏 Bias Score final: {bias_score:.1f}/10")
            
            self.feature_list = features
            return df[['timestamp'] + features + ['label']]
            
        except Exception as e:
            print(f"❌ Error creando features: {e}")
            return None
    
    def train_optimized_model(self):
        """Entrenar modelo optimizado completo"""
        print("🔷 ENTRENAMIENTO ETH OPTIMIZADO")
        print("="*60)
        
        # 1. Obtener datos
        df = self.connect_and_get_data()
        if df is None:
            return False
        
        # 2. Calcular threshold óptimo
        self.calculate_optimal_threshold(df)
        
        # 3. Crear features y labels
        df = self.create_features_and_labels(df)
        if df is None:
            return False
        
        # 4. Preparar datos de entrenamiento
        feature_columns = self.feature_list
        
        print("\n📋 PREPARANDO DATOS DE ENTRENAMIENTO")
        print("-" * 40)
        
        # Normalizar
        scaler = MinMaxScaler()
        df_scaled = df.copy()
        df_scaled[feature_columns] = scaler.fit_transform(df[feature_columns])
        
        # Crear directorio
        os.makedirs('models/eth_optimized', exist_ok=True)
        
        # Guardar scaler
        with open('models/eth_optimized/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        # Crear secuencias
        X = []
        y = []
        
        for i in range(self.lookback_window, len(df_scaled)):
            feature_seq = df_scaled[feature_columns].iloc[i-self.lookback_window:i].values
            X.append(feature_seq)
            y.append(int(df_scaled['label'].iloc[i]))
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✅ Datos preparados: X{X.shape}, y{y.shape}")
        
        # 5. Crear modelo
        print("\n📋 CREANDO MODELO TCN OPTIMIZADO")
        print("-" * 40)
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.lookback_window, self.expected_features)),
            
            # TCN layers con más capacidad
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Conv1D(filters=128, kernel_size=3, dilation_rate=2, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=4, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.GlobalAveragePooling1D(),
            
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0005),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Modelo creado: {model.count_params():,} parámetros")
        
        # 6. Entrenar
        print("\n📋 ENTRENANDO MODELO")
        print("-" * 40)
        
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # 7. Evaluar
        print("\n📋 EVALUANDO MODELO OPTIMIZADO")
        print("-" * 40)
        
        predictions = model.predict(X_val, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        
        pred_counts = Counter(pred_classes)
        total_preds = len(pred_classes)
        
        print("📊 Predicciones finales:")
        for i, name in enumerate(self.class_names):
            count = pred_counts[i]
            pct = count / total_preds * 100
            print(f"   - {name}: {count} ({pct:.1f}%)")
        
        percentages = [pred_counts[i]/total_preds for i in range(3)]
        final_bias = (max(percentages) - min(percentages)) * 10
        
        print(f"\n🎯 BIAS SCORE FINAL: {final_bias:.1f}/10")
        
        # 8. Guardar
        model_path = 'models/eth_optimized/eth_tcn_optimized.h5'
        model.save(model_path)
        
        metadata = {
            'pair': self.pair_symbol,
            'model_type': 'ETH Optimized TCN',
            'volatility_threshold': self.volatility_threshold,
            'bias_score': float(final_bias),
            'final_distribution': {
                'SELL': float(pred_counts[0]/total_preds),
                'HOLD': float(pred_counts[1]/total_preds),
                'BUY': float(pred_counts[2]/total_preds)
            },
            'training_date': datetime.now().isoformat()
        }
        
        with open('models/eth_optimized/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Modelo guardado: {model_path}")
        
        if final_bias < 4.0:
            print("🏆 ¡MODELO ETH OPTIMIZADO EXITOSO!")
        else:
            print("⚠️ Modelo funcional pero puede mejorar")
        
        return True


if __name__ == "__main__":
    trainer = EthereumTCNOptimized()
    success = trainer.train_optimized_model()
    
    if success:
        print("\n✅ ¡ETH OPTIMIZADO COMPLETADO!")
    else:
        print("\n❌ Error en optimización ETH") 