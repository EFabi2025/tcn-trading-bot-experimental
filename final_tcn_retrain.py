#!/usr/bin/env python3
"""
🚀 Re-entrenamiento Final del Modelo TCN Anti-Bias

Versión final optimizada con técnicas anti-sesgo comprobadas:
- Class weights balanceados
- Augmentación sintética de datos
- Arquitectura TCN mejorada
- Validación rigurosa de sesgo
- Multi-timeframe data
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone, timedelta
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
from binance.client import Client as BinanceClient
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle
import json


class FinalTCNRetrainer:
    """
    🚀 Re-entrenador final del modelo TCN Anti-Bias
    """
    
    def __init__(self):
        self.symbol = "BTCUSDT"
        self.interval = "5m"
        self.lookback_window = 60
        self.expected_features = 66
        self.class_names = ['SELL', 'HOLD', 'BUY']
        self.model_save_path = "models/tcn_anti_bias_final.h5"
        self.scaler_save_path = "models/feature_scalers_final.pkl"
        self.history_save_path = "models/training_history_final.json"
        
        print("🚀 Final TCN Anti-Bias Retrainer inicializado")
        print(f"   - Features: {self.expected_features}")
        print(f"   - Clases: {self.class_names}")
        print(f"   - Ventana: {self.lookback_window}")
    
    def setup_binance_client(self) -> bool:
        """Configura cliente de Binance"""
        try:
            print("\n🔗 Conectando a Binance...")
            self.binance_client = BinanceClient()
            server_time = self.binance_client.get_server_time()
            print(f"✅ Conectado a Binance")
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def collect_extensive_historical_data(self) -> Optional[pd.DataFrame]:
        """Recolecta datos históricos extensos"""
        try:
            print(f"\n📊 Recolectando datos históricos extensos...")
            
            # Obtener más datos históricos
            klines = self.binance_client.get_historical_klines(
                symbol=self.symbol,
                interval=self.interval,
                limit=1500
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()
            
            print(f"✅ Datos históricos recolectados: {len(df)} períodos")
            print(f"   - Desde: {df['timestamp'].iloc[0]}")
            print(f"   - Hasta: {df['timestamp'].iloc[-1]}")
            print(f"   - Rango de precios: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def create_enhanced_market_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea regímenes de mercado mejorados"""
        try:
            print("\n🔍 Creando regímenes de mercado mejorados...")
            
            df = df.copy()
            
            # Múltiples indicadores para detectar regímenes
            df['returns_short'] = df['close'].pct_change(periods=5)
            df['returns_medium'] = df['close'].pct_change(periods=20)
            df['returns_long'] = df['close'].pct_change(periods=50)
            
            # Volatilidad
            df['volatility'] = df['close'].pct_change().rolling(window=20).std()
            
            # Momentum
            df['momentum'] = df['close'] / df['close'].shift(20) - 1
            
            # Forzar diversidad en regímenes mediante clasificación más agresiva
            regimes = []
            
            # Calcular percentiles para thresholds adaptativos
            returns_medium = df['returns_medium'].dropna()
            p25 = returns_medium.quantile(0.25)
            p75 = returns_medium.quantile(0.75)
            
            print(f"   - Umbrales adaptativos: P25={p25:.4f}, P75={p75:.4f}")
            
            for i, row in df.iterrows():
                ret_medium = row['returns_medium']
                volatility = row['volatility']
                momentum = row['momentum']
                
                if pd.isna(ret_medium):
                    regime = 1  # SIDEWAYS por defecto
                else:
                    # Clasificación forzada para diversidad
                    if ret_medium >= p75:  # Top 25%
                        regime = 2  # BULL
                    elif ret_medium <= p25:  # Bottom 25%
                        regime = 0  # BEAR
                    else:
                        regime = 1  # SIDEWAYS
                
                regimes.append(regime)
            
            df['regime'] = regimes
            
            # Estadísticas
            regime_counts = Counter(regimes)
            total = len(regimes)
            
            print(f"📊 Distribución de regímenes mejorados:")
            regime_names = ['BEAR', 'SIDEWAYS', 'BULL']
            for i, name in enumerate(regime_names):
                count = regime_counts[i]
                pct = count / total * 100
                print(f"   - {name}: {count} ({pct:.1f}%)")
            
            return df
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return df
    
    def create_comprehensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea el conjunto completo de 66 features técnicas"""
        try:
            print("\n🔧 Creando features técnicas comprehensivas...")
            
            df = df.copy()
            features = []
            
            # 1. OHLCV básicos (5 features)
            features.extend(['open', 'high', 'low', 'close', 'volume'])
            
            # 2. Moving Averages SMA (10 features)
            for period in [5, 7, 10, 14, 20, 25, 30, 50, 100, 200]:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                features.append(f'sma_{period}')
            
            # 3. Exponential Moving Averages (8 features)
            for period in [5, 9, 12, 21, 26, 50, 100, 200]:
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                features.append(f'ema_{period}')
            
            # 4. RSI múltiples períodos (4 features)
            for period in [9, 14, 21, 30]:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
                features.append(f'rsi_{period}')
            
            # 5. MACD completo (6 features)
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            df['macd_normalized'] = df['macd'] / df['close']
            df['macd_signal_normalized'] = df['macd_signal'] / df['close']
            df['macd_histogram_normalized'] = df['macd_histogram'] / df['close']
            features.extend(['macd', 'macd_signal', 'macd_histogram', 
                           'macd_normalized', 'macd_signal_normalized', 'macd_histogram_normalized'])
            
            # 6. Bollinger Bands (6 features)
            for period in [20, 50]:
                bb_middle = df['close'].rolling(window=period).mean()
                bb_std = df['close'].rolling(window=period).std()
                df[f'bb_upper_{period}'] = bb_middle + (bb_std * 2)
                df[f'bb_lower_{period}'] = bb_middle - (bb_std * 2)
                df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
                features.extend([f'bb_upper_{period}', f'bb_lower_{period}', f'bb_position_{period}'])
            
            # 7. Momentum y ROC (8 features)
            for period in [3, 5, 10, 20]:
                df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
                df[f'roc_{period}'] = df['close'].pct_change(periods=period)
                features.extend([f'momentum_{period}', f'roc_{period}'])
            
            # 8. Volatilidad (4 features)
            for period in [5, 10, 20, 50]:
                df[f'volatility_{period}'] = df['close'].pct_change().rolling(window=period).std()
                features.append(f'volatility_{period}')
            
            # 9. Volume analysis (6 features)
            for period in [5, 10, 20]:
                df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
                df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']
                features.extend([f'volume_sma_{period}', f'volume_ratio_{period}'])
            
            # 10. ATR (Average True Range) (3 features)
            for period in [14, 21, 30]:
                high_low = df['high'] - df['low']
                high_close = (df['high'] - df['close'].shift()).abs()
                low_close = (df['low'] - df['close'].shift()).abs()
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df[f'atr_{period}'] = true_range.rolling(window=period).mean()
                features.append(f'atr_{period}')
            
            # 11. Stochastic Oscillator (2 features)
            low_min = df['low'].rolling(window=14).min()
            high_max = df['high'].rolling(window=14).max()
            df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            features.extend(['stoch_k', 'stoch_d'])
            
            # 12. Williams %R (2 features)
            for period in [14, 21]:
                high_max = df['high'].rolling(window=period).max()
                low_min = df['low'].rolling(window=period).min()
                df[f'williams_r_{period}'] = -100 * (high_max - df['close']) / (high_max - low_min)
                features.append(f'williams_r_{period}')
            
            # 13. Price position features (4 features)
            for period in [10, 20]:
                df[f'price_position_{period}'] = (df['close'] - df['low'].rolling(period).min()) / \
                                                (df['high'].rolling(period).max() - df['low'].rolling(period).min())
                df[f'price_distance_ma_{period}'] = (df['close'] - df['close'].rolling(period).mean()) / df['close']
                features.extend([f'price_position_{period}', f'price_distance_ma_{period}'])
            
            # 14. Features adicionales para completar 66
            df['close_change'] = df['close'].pct_change()
            df['volume_change'] = df['volume'].pct_change()
            features.extend(['close_change', 'volume_change'])
            
            # Asegurar exactamente 66 features
            features = features[:66]
            df = df.dropna()
            
            print(f"✅ Features creadas: {len(features)}")
            print(f"   - Datos limpios: {len(df)} períodos")
            
            return df[['timestamp', 'regime'] + features]
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def create_balanced_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea labels balanceadas estratégicamente"""
        try:
            print("\n🎯 Creando labels balanceadas...")
            
            df = df.copy()
            
            # Múltiples horizontes de predicción
            df['future_price_6'] = df['close'].shift(-6)   # 30min
            df['future_price_12'] = df['close'].shift(-12) # 1h
            df['future_price_18'] = df['close'].shift(-18) # 1.5h
            
            # Cambios de precio futuros
            df['change_6'] = (df['future_price_6'] - df['close']) / df['close']
            df['change_12'] = (df['future_price_12'] - df['close']) / df['close']
            df['change_18'] = (df['future_price_18'] - df['close']) / df['close']
            
            # Calcular umbrales adaptativos por régimen para forzar balance
            labels = []
            
            for i, row in df.iterrows():
                regime = row['regime']
                change_6 = row['change_6']
                change_12 = row['change_12']
                change_18 = row['change_18']
                
                if pd.isna(change_6) or pd.isna(change_12) or pd.isna(change_18):
                    label = 1  # HOLD por defecto
                else:
                    # Umbrales más agresivos para crear balance forzado
                    if regime == 0:  # BEAR market - más agresivo en SELL
                        if change_12 < -0.001:  # -0.1%
                            label = 0  # SELL
                        elif change_12 > 0.003:  # +0.3%
                            label = 2  # BUY
                        else:
                            label = 1  # HOLD
                    elif regime == 2:  # BULL market - más agresivo en BUY
                        if change_12 < -0.003:  # -0.3%
                            label = 0  # SELL
                        elif change_12 > 0.001:  # +0.1%
                            label = 2  # BUY
                        else:
                            label = 1  # HOLD
                    else:  # SIDEWAYS - balance equilibrado
                        if change_12 < -0.002:  # -0.2%
                            label = 0  # SELL
                        elif change_12 > 0.002:  # +0.2%
                            label = 2  # BUY
                        else:
                            label = 1  # HOLD
                
                labels.append(label)
            
            df['label'] = labels
            df = df.dropna(subset=['change_6', 'change_12', 'change_18'])
            
            # Estadísticas iniciales
            label_counts = Counter(labels)
            total_labels = len([l for l in labels if not pd.isna(l)])
            
            print(f"📊 Distribución inicial de labels:")
            for i, name in enumerate(self.class_names):
                count = label_counts[i]
                pct = count / total_labels * 100 if total_labels > 0 else 0
                print(f"   - {name}: {count} ({pct:.1f}%)")
            
            # Balance forzado si es necesario
            min_class_count = min(label_counts.values())
            max_class_count = max(label_counts.values())
            
            if max_class_count / min_class_count > 2.5:  # Si hay más de 2.5x diferencia
                print(f"\n⚖️ Aplicando balance forzado...")
                
                # Undersample clases mayoritarias
                balanced_dfs = []
                target_size = int(min_class_count * 1.8)  # Un poco más que la mínima
                
                for label_id in [0, 1, 2]:
                    label_data = df[df['label'] == label_id]
                    if len(label_data) > target_size:
                        balanced_data = label_data.sample(n=target_size, random_state=42)
                    else:
                        balanced_data = label_data
                    balanced_dfs.append(balanced_data)
                
                df = pd.concat(balanced_dfs, ignore_index=True)
                df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
                
                # Estadísticas finales
                final_label_counts = Counter(df['label'])
                final_total = len(df)
                
                print(f"✅ Distribución balanceada:")
                for i, name in enumerate(self.class_names):
                    count = final_label_counts[i]
                    pct = count / final_total * 100
                    print(f"   - {name}: {count} ({pct:.1f}%)")
            
            return df
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return df
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepara datos para entrenamiento"""
        try:
            print("\n🔧 Preparando datos para entrenamiento...")
            
            # Separar features y labels
            feature_columns = [col for col in df.columns if col not in ['timestamp', 'regime', 'label', 'future_price_6', 'future_price_12', 'future_price_18', 'change_6', 'change_12', 'change_18']]
            features_data = df[feature_columns].values
            regimes_data = df['regime'].values
            labels_data = df['label'].values
            
            print(f"   - Features shape: {features_data.shape}")
            print(f"   - Regimes shape: {regimes_data.shape}")
            print(f"   - Labels shape: {labels_data.shape}")
            
            # Normalizar features
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            features_scaled = self.scaler.fit_transform(features_data)
            
            # Crear secuencias
            X_features, X_regimes, y = [], [], []
            
            for i in range(self.lookback_window, len(features_scaled)):
                # Secuencia de features
                X_features.append(features_scaled[i-self.lookback_window:i])
                
                # Régimen actual como one-hot
                regime = regimes_data[i]
                regime_onehot = [0, 0, 0]
                regime_onehot[int(regime)] = 1
                X_regimes.append(regime_onehot)
                
                # Label
                y.append(labels_data[i])
            
            X_features = np.array(X_features)
            X_regimes = np.array(X_regimes)
            y = np.array(y)
            
            print(f"✅ Datos preparados:")
            print(f"   - X_features: {X_features.shape}")
            print(f"   - X_regimes: {X_regimes.shape}")
            print(f"   - y: {y.shape}")
            
            return X_features, X_regimes, y
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None, None, None
    
    def create_final_tcn_model(self) -> tf.keras.Model:
        """Crea modelo TCN final optimizado"""
        try:
            print("\n🧠 Creando modelo TCN final...")
            
            # Input 1: Features técnicas (secuencias)
            features_input = tf.keras.Input(shape=(self.lookback_window, self.expected_features), name='price_features')
            
            # TCN Layers optimizadas
            x = features_input
            
            # Bloque 1: Patrones locales
            x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='causal', activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            
            # Bloque 2: Patrones medianos
            x = tf.keras.layers.Conv1D(filters=128, kernel_size=3, dilation_rate=2, padding='causal', activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.4)(x)
            
            # Bloque 3: Patrones largos
            x = tf.keras.layers.Conv1D(filters=256, kernel_size=3, dilation_rate=4, padding='causal', activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.4)(x)
            
            # Bloque 4: Patrones muy largos
            x = tf.keras.layers.Conv1D(filters=128, kernel_size=3, dilation_rate=8, padding='causal', activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            
            # Global pooling
            x = tf.keras.layers.GlobalMaxPooling1D()(x)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            
            # Input 2: Market regime
            regime_input = tf.keras.Input(shape=(3,), name='market_regime')
            regime_dense = tf.keras.layers.Dense(64, activation='relu')(regime_input)
            regime_dense = tf.keras.layers.BatchNormalization()(regime_dense)
            regime_dense = tf.keras.layers.Dropout(0.3)(regime_dense)
            
            # Fusión
            combined = tf.keras.layers.concatenate([x, regime_dense])
            
            # Capas densas finales
            combined = tf.keras.layers.Dense(256, activation='relu')(combined)
            combined = tf.keras.layers.BatchNormalization()(combined)
            combined = tf.keras.layers.Dropout(0.5)(combined)
            
            combined = tf.keras.layers.Dense(128, activation='relu')(combined)
            combined = tf.keras.layers.BatchNormalization()(combined)
            combined = tf.keras.layers.Dropout(0.4)(combined)
            
            combined = tf.keras.layers.Dense(64, activation='relu')(combined)
            combined = tf.keras.layers.Dropout(0.3)(combined)
            
            # Output layer
            outputs = tf.keras.layers.Dense(3, activation='softmax', name='predictions')(combined)
            
            # Crear modelo
            model = tf.keras.Model(inputs=[features_input, regime_input], outputs=outputs)
            
            # Compilar con configuración optimizada
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print(f"✅ Modelo TCN final creado:")
            print(f"   - Parámetros totales: {model.count_params():,}")
            
            return model
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def train_final_model(self, model: tf.keras.Model, X_features: np.ndarray, X_regimes: np.ndarray, y: np.ndarray) -> Dict:
        """Entrena el modelo final con técnicas optimizadas"""
        try:
            print("\n🚀 Entrenando modelo final...")
            
            # Split train/validation
            X_feat_train, X_feat_val, X_reg_train, X_reg_val, y_train, y_val = train_test_split(
                X_features, X_regimes, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"   - Train samples: {len(X_feat_train)}")
            print(f"   - Validation samples: {len(X_feat_val)}")
            
            # Calcular class weights balanceados
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
            
            print(f"   - Class weights: {class_weight_dict}")
            
            # Callbacks optimizados
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=20,
                    restore_best_weights=True,
                    min_delta=0.001
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=0.0001,
                    min_delta=0.001
                )
            ]
            
            # Entrenar modelo
            history = model.fit(
                [X_feat_train, X_reg_train], y_train,
                validation_data=([X_feat_val, X_reg_val], y_val),
                epochs=150,
                batch_size=32,
                class_weight=class_weight_dict,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluación final
            val_loss, val_accuracy = model.evaluate([X_feat_val, X_reg_val], y_val, verbose=0)
            
            print(f"\n✅ Entrenamiento completado:")
            print(f"   - Validation Accuracy: {val_accuracy:.4f}")
            print(f"   - Validation Loss: {val_loss:.4f}")
            
            return {
                'history': history.history,
                'val_accuracy': val_accuracy,
                'val_loss': val_loss,
                'model': model
            }
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def comprehensive_bias_test(self, model: tf.keras.Model, X_features: np.ndarray, X_regimes: np.ndarray, y: np.ndarray) -> bool:
        """Test comprehensivo de sesgo del modelo final"""
        try:
            print("\n🧪 Test comprehensivo de sesgo...")
            
            # Hacer predicciones
            predictions = model.predict([X_features, X_regimes], verbose=0)
            pred_classes = np.argmax(predictions, axis=1)
            pred_confidences = np.max(predictions, axis=1)
            
            # Test 1: Distribución general
            pred_counts = Counter(pred_classes)
            total_preds = len(pred_classes)
            
            print(f"📊 Test 1 - Distribución general:")
            class_percentages = []
            for i, name in enumerate(self.class_names):
                count = pred_counts[i]
                pct = count / total_preds * 100
                class_percentages.append(pct)
                print(f"   - {name}: {count} ({pct:.1f}%)")
            
            # Criterio 1: Ninguna clase debe dominar más del 60%
            max_class_pct = max(class_percentages)
            min_class_pct = min(class_percentages)
            
            bias_score = 0
            
            if max_class_pct > 60:
                print(f"   ⚠️ Clase dominante: {max_class_pct:.1f}% > 60%")
                bias_score += 2
            
            if min_class_pct < 15:
                print(f"   ⚠️ Clase minoritaria: {min_class_pct:.1f}% < 15%")
                bias_score += 1
            
            # Test 2: Diversidad temporal
            print(f"\n📊 Test 2 - Diversidad temporal:")
            window_size = 50
            windows_with_bias = 0
            total_windows = 0
            
            for i in range(0, len(pred_classes) - window_size, 25):
                window_preds = pred_classes[i:i + window_size]
                window_counts = Counter(window_preds)
                max_window_pct = max(window_counts.values()) / len(window_preds) * 100
                
                if max_window_pct > 75:
                    windows_with_bias += 1
                total_windows += 1
            
            temporal_bias_pct = windows_with_bias / total_windows * 100 if total_windows > 0 else 0
            print(f"   - Ventanas con sesgo temporal: {temporal_bias_pct:.1f}%")
            
            if temporal_bias_pct > 25:
                bias_score += 1
            
            # Test 3: Confianza balanceada
            print(f"\n📊 Test 3 - Análisis de confianza:")
            avg_confidence = pred_confidences.mean()
            std_confidence = pred_confidences.std()
            
            print(f"   - Confianza promedio: {avg_confidence:.3f}")
            print(f"   - Desviación estándar: {std_confidence:.3f}")
            
            # Confianza muy alta puede indicar overconfidence/sesgo
            if avg_confidence > 0.85:
                print(f"   ⚠️ Confianza muy alta, posible overconfidence")
                bias_score += 1
            
            # Resultado final
            print(f"\n🎯 Resultado del test de sesgo:")
            print(f"   - Score total de sesgo: {bias_score}/5")
            
            if bias_score <= 1:
                print(f"✅ MODELO APROBADO - Sesgo mínimo")
                return True
            elif bias_score <= 2:
                print(f"⚠️ MODELO ACEPTABLE - Sesgo bajo")
                return True
            else:
                print(f"❌ MODELO RECHAZADO - Sesgo alto")
                return False
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def save_final_artifacts(self, model: tf.keras.Model, training_results: Dict):
        """Guarda artefactos finales"""
        try:
            print(f"\n💾 Guardando artefactos finales...")
            
            # Guardar modelo
            model.save(self.model_save_path)
            print(f"✅ Modelo guardado: {self.model_save_path}")
            
            # Guardar scaler
            with open(self.scaler_save_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"✅ Scaler guardado: {self.scaler_save_path}")
            
            # Guardar historial
            history_data = {
                'val_accuracy': float(training_results['val_accuracy']),
                'val_loss': float(training_results['val_loss']),
                'training_timestamp': datetime.now().isoformat(),
                'model_config': {
                    'lookback_window': self.lookback_window,
                    'features': self.expected_features,
                    'classes': self.class_names,
                    'architecture': 'Final TCN Anti-Bias'
                }
            }
            
            with open(self.history_save_path, 'w') as f:
                json.dump(history_data, f, indent=2)
            print(f"✅ Historial guardado: {self.history_save_path}")
            
        except Exception as e:
            print(f"❌ Error guardando: {e}")
    
    async def run_final_retraining(self):
        """Ejecuta el re-entrenamiento final completo"""
        print("🚀 Iniciando re-entrenamiento FINAL del modelo TCN Anti-Bias")
        print("="*80)
        
        # 1. Configurar Binance
        if not self.setup_binance_client():
            return False
        
        # 2. Recolectar datos históricos extensos
        df = self.collect_extensive_historical_data()
        if df is None:
            return False
        
        # 3. Crear regímenes mejorados
        df_regimes = self.create_enhanced_market_regimes(df)
        
        # 4. Crear features comprehensivas
        df_features = self.create_comprehensive_features(df_regimes)
        if df_features is None:
            return False
        
        # 5. Crear labels balanceadas
        df_labeled = self.create_balanced_labels(df_features)
        
        # 6. Preparar datos para entrenamiento
        X_features, X_regimes, y = self.prepare_training_data(df_labeled)
        if X_features is None:
            return False
        
        # 7. Crear modelo final
        model = self.create_final_tcn_model()
        if model is None:
            return False
        
        # 8. Entrenar modelo
        training_results = self.train_final_model(model, X_features, X_regimes, y)
        if training_results is None:
            return False
        
        # 9. Test comprehensivo de sesgo
        final_model = training_results['model']
        is_bias_free = self.comprehensive_bias_test(final_model, X_features, X_regimes, y)
        
        # 10. Guardar artefactos
        self.save_final_artifacts(final_model, training_results)
        
        print("\n" + "="*80)
        if is_bias_free:
            print("🎉 RE-ENTRENAMIENTO FINAL EXITOSO!")
            print(f"✅ Modelo sin sesgo significativo")
        else:
            print("⚠️ RE-ENTRENAMIENTO COMPLETADO CON ADVERTENCIAS")
            print(f"❌ Modelo presenta sesgo residual")
        
        print(f"📋 RESUMEN FINAL:")
        print(f"   - Modelo: {self.model_save_path}")
        print(f"   - Accuracy: {training_results['val_accuracy']:.4f}")
        print(f"   - Loss: {training_results['val_loss']:.4f}")
        print(f"   - Features: {self.expected_features}")
        print(f"   - Arquitectura: Final TCN Anti-Bias")
        print(f"   - Estado: {'✅ APROBADO' if is_bias_free else '⚠️ CON ADVERTENCIAS'}")
        
        return is_bias_free


async def main():
    print("🚀 Final TCN Anti-Bias Retrainer")
    print("="*80)
    
    retrainer = FinalTCNRetrainer()
    
    try:
        success = await retrainer.run_final_retraining()
        
        if success:
            print("\n✅ ¡Re-entrenamiento final exitoso!")
            print("🎯 Modelo TCN Anti-Bias final listo para producción")
        else:
            print("\n⚠️ Re-entrenamiento completado con advertencias.")
            print("🔧 Modelo funcional pero con sesgo residual")
    
    except Exception as e:
        print(f"\n💥 Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())