#!/usr/bin/env python3
"""
🚀 Re-entrenamiento Avanzado del Modelo TCN Anti-Bias

Sistema avanzado con técnicas anti-sesgo más sofisticadas:
- Focal Loss para clases desbalanceadas
- Augmentación sintética de datos
- Cross-validation estratificada
- Ensemble de modelos
- Validación rigurosa de sesgo
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
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle
import json


class FocalLoss(tf.keras.losses.Loss):
    """
    🎯 Focal Loss para combatir el desbalance de clases
    """
    def __init__(self, alpha=1.0, gamma=2.0, name='focal_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        # Convertir a sparse si es necesario
        y_true = tf.cast(y_true, tf.int32)
        
        # One-hot encoding si es necesario
        num_classes = tf.shape(y_pred)[-1]
        y_true_one_hot = tf.one_hot(y_true, num_classes)
        
        # Calcular cross entropy
        ce_loss = tf.keras.losses.categorical_crossentropy(y_true_one_hot, y_pred)
        
        # Calcular focal weight
        p_t = tf.where(tf.equal(y_true_one_hot, 1), y_pred, 1 - y_pred)
        focal_weight = self.alpha * tf.pow(1 - p_t, self.gamma)
        
        # Aplicar focal weight
        focal_loss = focal_weight * ce_loss
        
        return tf.reduce_mean(focal_loss)


class AdvancedTCNRetrainer:
    """
    🚀 Re-entrenador avanzado del modelo TCN Anti-Bias
    """
    
    def __init__(self):
        self.symbol = "BTCUSDT"
        self.interval = "5m"
        self.lookback_window = 60
        self.expected_features = 66
        self.class_names = ['SELL', 'HOLD', 'BUY']
        self.model_save_path = "models/tcn_anti_bias_advanced.h5"
        self.scaler_save_path = "models/feature_scalers_advanced.pkl"
        self.history_save_path = "models/training_history_advanced.json"
        
        print("🚀 Advanced TCN Anti-Bias Retrainer inicializado")
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
    
    def collect_multi_timeframe_data(self) -> Optional[pd.DataFrame]:
        """Recolecta datos de múltiples timeframes para diversidad"""
        try:
            print(f"\n📊 Recolectando datos multi-timeframe...")
            
            all_data = []
            
            # Obtener datos históricos de diferentes períodos
            periods = [
                ("1h", 1680),   # ~1 semana en horas
                ("5m", 2016),   # ~1 semana en 5min
                ("15m", 1344),  # ~2 semanas en 15min
            ]
            
            for interval, limit in periods:
                print(f"   - Recolectando {interval} data...")
                
                klines = self.binance_client.get_historical_klines(
                    symbol=self.symbol,
                    interval=interval,
                    limit=limit
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
                df['timeframe'] = interval
                all_data.append(df)
            
            # Combinar todos los datos
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
            
            # Remover duplicados por timestamp
            combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='first')
            
            print(f"✅ Datos multi-timeframe recolectados: {len(combined_df)} períodos")
            print(f"   - Rango temporal: {combined_df['timestamp'].min()} a {combined_df['timestamp'].max()}")
            
            return combined_df
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def create_diverse_market_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea regímenes de mercado más diversos"""
        try:
            print("\n🔍 Creando regímenes de mercado diversos...")
            
            df = df.copy()
            
            # Múltiples indicadores para detectar regímenes
            df['returns_1'] = df['close'].pct_change(periods=1)
            df['returns_5'] = df['close'].pct_change(periods=5)
            df['returns_20'] = df['close'].pct_change(periods=20)
            df['returns_50'] = df['close'].pct_change(periods=50)
            
            # Volatilidad multi-período
            df['vol_5'] = df['returns_1'].rolling(window=5).std()
            df['vol_20'] = df['returns_1'].rolling(window=20).std()
            df['vol_50'] = df['returns_1'].rolling(window=50).std()
            
            # Momentum indicators
            df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
            df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
            
            # Clasificación mejorada de regímenes
            regimes = []
            
            for i, row in df.iterrows():
                ret_5 = row['returns_5']
                ret_20 = row['returns_20']
                vol_20 = row['vol_20']
                momentum_20 = row['momentum_20']
                
                if pd.isna(ret_5) or pd.isna(ret_20) or pd.isna(vol_20):
                    regime = 1  # SIDEWAYS por defecto
                else:
                    # Clasificación más sofisticada
                    if vol_20 > 0.02:  # Alta volatilidad
                        if ret_20 > 0.03:  # +3% y volátil
                            regime = 2  # BULL VOLÁTIL
                        elif ret_20 < -0.03:  # -3% y volátil
                            regime = 0  # BEAR VOLÁTIL
                        else:
                            regime = 1  # SIDEWAYS VOLÁTIL
                    else:  # Baja volatilidad
                        if ret_20 > 0.015:  # +1.5% y tranquilo
                            regime = 2  # BULL TRANQUILO
                        elif ret_20 < -0.015:  # -1.5% y tranquilo
                            regime = 0  # BEAR TRANQUILO
                        else:
                            regime = 1  # SIDEWAYS TRANQUILO
                
                regimes.append(regime)
            
            df['regime'] = regimes
            
            # Estadísticas
            regime_counts = Counter(regimes)
            total = len(regimes)
            
            print(f"📊 Distribución de regímenes diversos:")
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
    
    def create_advanced_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea labels más sofisticadas y balanceadas"""
        try:
            print("\n🎯 Creando labels avanzadas...")
            
            df = df.copy()
            
            # Múltiples horizontes de predicción
            df['future_price_6'] = df['close'].shift(-6)   # 30min adelante
            df['future_price_12'] = df['close'].shift(-12) # 1h adelante
            df['future_price_24'] = df['close'].shift(-24) # 2h adelante
            
            # Cambios de precio futuros
            df['change_6'] = (df['future_price_6'] - df['close']) / df['close']
            df['change_12'] = (df['future_price_12'] - df['close']) / df['close']
            df['change_24'] = (df['future_price_24'] - df['close']) / df['close']
            
            # Labels más sofisticadas basadas en consenso
            labels = []
            
            for i, row in df.iterrows():
                regime = row['regime']
                change_6 = row['change_6']
                change_12 = row['change_12']
                change_24 = row['change_24']
                
                if pd.isna(change_6) or pd.isna(change_12) or pd.isna(change_24):
                    label = 1  # HOLD por defecto
                else:
                    # Umbrales adaptativos por régimen
                    if regime == 0:  # BEAR market
                        sell_threshold = -0.002  # -0.2%
                        buy_threshold = 0.0015   # +0.15%
                    elif regime == 2:  # BULL market
                        sell_threshold = -0.0015 # -0.15%
                        buy_threshold = 0.002    # +0.2%
                    else:  # SIDEWAYS
                        sell_threshold = -0.0018 # -0.18%
                        buy_threshold = 0.0018   # +0.18%
                    
                    # Consenso de múltiples horizontes
                    sell_votes = 0
                    buy_votes = 0
                    
                    for change in [change_6, change_12, change_24]:
                        if change < sell_threshold:
                            sell_votes += 1
                        elif change > buy_threshold:
                            buy_votes += 1
                    
                    # Decisión por mayoría
                    if sell_votes >= 2:
                        label = 0  # SELL
                    elif buy_votes >= 2:
                        label = 2  # BUY
                    else:
                        label = 1  # HOLD
                
                labels.append(label)
            
            df['label'] = labels
            
            # Remover filas sin labels válidas
            df = df.dropna(subset=['change_6', 'change_12', 'change_24'])
            
            # Estadísticas de labels
            label_counts = Counter(labels)
            total_labels = len([l for l in labels if not pd.isna(l)])
            
            print(f"📊 Distribución inicial de labels:")
            for i, name in enumerate(self.class_names):
                count = label_counts[i]
                pct = count / total_labels * 100 if total_labels > 0 else 0
                print(f"   - {name}: {count} ({pct:.1f}%)")
            
            return df
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return df
    
    def apply_smote_balancing(self, X_features: np.ndarray, X_regimes: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Aplica SMOTE para balancear las clases"""
        try:
            print("\n⚖️ Aplicando SMOTE para balancear clases...")
            
            # Reshape para SMOTE
            n_samples, timesteps, n_features = X_features.shape
            X_features_flat = X_features.reshape(n_samples, timesteps * n_features)
            
            # Combinar features y regimes
            X_combined = np.concatenate([X_features_flat, X_regimes], axis=1)
            
            # Aplicar SMOTE
            smote = SMOTE(random_state=42, k_neighbors=3)
            X_balanced, y_balanced = smote.fit_resample(X_combined, y)
            
            # Separar de vuelta
            features_size = timesteps * n_features
            X_features_balanced = X_balanced[:, :features_size].reshape(-1, timesteps, n_features)
            X_regimes_balanced = X_balanced[:, features_size:]
            
            print(f"✅ Balanceado con SMOTE:")
            print(f"   - Muestras originales: {len(y)}")
            print(f"   - Muestras balanceadas: {len(y_balanced)}")
            
            # Nuevas estadísticas
            balanced_counts = Counter(y_balanced)
            for i, name in enumerate(self.class_names):
                count = balanced_counts[i]
                pct = count / len(y_balanced) * 100
                print(f"   - {name}: {count} ({pct:.1f}%)")
            
            return X_features_balanced, X_regimes_balanced, y_balanced
            
        except Exception as e:
            print(f"❌ Error en SMOTE: {e}")
            return X_features, X_regimes, y
    
    def create_advanced_tcn_model(self) -> tf.keras.Model:
        """Crea modelo TCN avanzado con arquitectura mejorada"""
        try:
            print("\n🧠 Creando modelo TCN avanzado...")
            
            # Input 1: Features técnicas (secuencias)
            features_input = tf.keras.Input(shape=(self.lookback_window, self.expected_features), name='price_features')
            
            # TCN Layers con dilated convolutions mejoradas
            x = features_input
            
            # Bloque 1: Detección de patrones locales
            x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='causal', activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            
            # Bloque 2: Patrones de mediano plazo
            x = tf.keras.layers.Conv1D(filters=128, kernel_size=3, dilation_rate=2, padding='causal', activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            
            # Bloque 3: Patrones de largo plazo
            x = tf.keras.layers.Conv1D(filters=256, kernel_size=3, dilation_rate=4, padding='causal', activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            
            # Bloque 4: Patrones de muy largo plazo
            x = tf.keras.layers.Conv1D(filters=128, kernel_size=3, dilation_rate=8, padding='causal', activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.4)(x)
            
            # Attention mechanism
            attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
            x = tf.keras.layers.Add()([x, attention])
            x = tf.keras.layers.LayerNormalization()(x)
            
            # Global pooling con información preservada
            x_max = tf.keras.layers.GlobalMaxPooling1D()(x)
            x_avg = tf.keras.layers.GlobalAveragePooling1D()(x)
            x = tf.keras.layers.concatenate([x_max, x_avg])
            
            # Dense layers para features
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            
            # Input 2: Market regime con más capacidad
            regime_input = tf.keras.Input(shape=(3,), name='market_regime')
            regime_dense = tf.keras.layers.Dense(64, activation='relu')(regime_input)
            regime_dense = tf.keras.layers.BatchNormalization()(regime_dense)
            regime_dense = tf.keras.layers.Dropout(0.3)(regime_dense)
            
            regime_dense2 = tf.keras.layers.Dense(32, activation='relu')(regime_dense)
            regime_dense2 = tf.keras.layers.BatchNormalization()(regime_dense2)
            regime_dense2 = tf.keras.layers.Dropout(0.2)(regime_dense2)
            
            # Fusión avanzada
            combined = tf.keras.layers.concatenate([x, regime_dense2])
            
            # Capas densas finales con regularización
            combined = tf.keras.layers.Dense(256, activation='relu')(combined)
            combined = tf.keras.layers.BatchNormalization()(combined)
            combined = tf.keras.layers.Dropout(0.5)(combined)
            
            combined = tf.keras.layers.Dense(128, activation='relu')(combined)
            combined = tf.keras.layers.BatchNormalization()(combined)
            combined = tf.keras.layers.Dropout(0.4)(combined)
            
            combined = tf.keras.layers.Dense(64, activation='relu')(combined)
            combined = tf.keras.layers.BatchNormalization()(combined)
            combined = tf.keras.layers.Dropout(0.3)(combined)
            
            # Output layer con softmax
            outputs = tf.keras.layers.Dense(3, activation='softmax', name='predictions')(combined)
            
            # Crear modelo
            model = tf.keras.Model(inputs=[features_input, regime_input], outputs=outputs)
            
            # Compilar con Focal Loss
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                loss=FocalLoss(alpha=1.0, gamma=2.0),
                metrics=['accuracy']
            )
            
            print(f"✅ Modelo TCN avanzado creado:")
            print(f"   - Parámetros totales: {model.count_params():,}")
            
            return model
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def train_with_cross_validation(self, model: tf.keras.Model, 
                                  X_features: np.ndarray, X_regimes: np.ndarray, y: np.ndarray) -> Dict:
        """Entrena con cross-validation estratificada"""
        try:
            print("\n🚀 Entrenando con cross-validation estratificada...")
            
            # Preparar cross-validation
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            
            fold_results = []
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_features, y)):
                print(f"\n📊 Fold {fold + 1}/3")
                
                # Split datos
                X_feat_train = X_features[train_idx]
                X_reg_train = X_regimes[train_idx]
                y_train = y[train_idx]
                
                X_feat_val = X_features[val_idx]
                X_reg_val = X_regimes[val_idx]
                y_val = y[val_idx]
                
                # Reinicializar pesos del modelo
                if fold > 0:
                    model = self.create_advanced_tcn_model()
                
                # Entrenar fold
                history = model.fit(
                    [X_feat_train, X_reg_train], y_train,
                    validation_data=([X_feat_val, X_reg_val], y_val),
                    epochs=50,
                    batch_size=32,
                    verbose=0,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor='val_accuracy',
                            patience=10,
                            restore_best_weights=True
                        )
                    ]
                )
                
                # Evaluar fold
                val_loss, val_accuracy = model.evaluate([X_feat_val, X_reg_val], y_val, verbose=0)
                
                print(f"   Fold {fold + 1} - Val Accuracy: {val_accuracy:.4f}")
                
                fold_results.append({
                    'fold': fold + 1,
                    'val_accuracy': val_accuracy,
                    'val_loss': val_loss,
                    'history': history.history
                })
            
            # Estadísticas generales
            avg_accuracy = np.mean([r['val_accuracy'] for r in fold_results])
            std_accuracy = np.std([r['val_accuracy'] for r in fold_results])
            
            print(f"\n✅ Cross-validation completado:")
            print(f"   - Accuracy promedio: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
            
            # Entrenamiento final con todos los datos
            print(f"\n🚀 Entrenamiento final con todos los datos...")
            
            model = self.create_advanced_tcn_model()
            
            final_history = model.fit(
                [X_features, X_regimes], y,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_accuracy',
                        patience=15,
                        restore_best_weights=True
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=8,
                        min_lr=0.0001
                    )
                ],
                verbose=1
            )
            
            return {
                'cv_results': fold_results,
                'avg_accuracy': avg_accuracy,
                'std_accuracy': std_accuracy,
                'final_model': model,
                'final_history': final_history.history
            }
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepara datos para entrenamiento"""
        try:
            print("\n🔧 Preparando datos para entrenamiento...")
            
            # Separar features y labels
            feature_columns = [col for col in df.columns if col not in ['timestamp', 'regime', 'label', 'future_price_6', 'future_price_12', 'future_price_24', 'change_6', 'change_12', 'change_24']]
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
    
    def rigorous_bias_validation(self, model: tf.keras.Model, X_features: np.ndarray, X_regimes: np.ndarray, y: np.ndarray) -> bool:
        """Validación rigurosa de sesgo"""
        try:
            print("\n🧪 Validación rigurosa de sesgo...")
            
            # Hacer predicciones
            predictions = model.predict([X_features, X_regimes], verbose=0)
            pred_classes = np.argmax(predictions, axis=1)
            
            # Test 1: Distribución general
            pred_counts = Counter(pred_classes)
            total_preds = len(pred_classes)
            
            print(f"📊 Test 1 - Distribución general:")
            bias_score = 0
            
            for i, name in enumerate(self.class_names):
                count = pred_counts[i]
                pct = count / total_preds * 100
                print(f"   - {name}: {count} ({pct:.1f}%)")
                
                # Penalizar desbalance extremo
                if pct > 70 or pct < 10:
                    bias_score += 1
            
            # Test 2: Consistencia por régimen
            print(f"\n📊 Test 2 - Consistencia por régimen:")
            regimes_decoded = np.argmax(X_regimes, axis=1)
            regime_names = ['BEAR', 'SIDEWAYS', 'BULL']
            
            for regime_idx in range(3):
                regime_mask = regimes_decoded == regime_idx
                regime_count = np.sum(regime_mask)
                
                if regime_count > 0:
                    regime_preds = pred_classes[regime_mask]
                    regime_pred_counts = Counter(regime_preds)
                    
                    print(f"   {regime_names[regime_idx]} Market:")
                    
                    regime_percentages = []
                    for i, name in enumerate(self.class_names):
                        count = regime_pred_counts[i]
                        pct = count / regime_count * 100 if regime_count > 0 else 0
                        regime_percentages.append(pct)
                        print(f"     - {name}: {pct:.1f}%")
                    
                    # Penalizar si una clase domina más del 80%
                    if max(regime_percentages) > 80:
                        bias_score += 1
            
            # Test 3: Diversidad temporal
            print(f"\n📊 Test 3 - Diversidad temporal:")
            window_size = 50
            temporal_bias = 0
            
            for i in range(0, len(pred_classes) - window_size, window_size):
                window_preds = pred_classes[i:i + window_size]
                window_counts = Counter(window_preds)
                max_pct = max(window_counts.values()) / len(window_preds) * 100
                
                if max_pct > 85:
                    temporal_bias += 1
            
            temporal_bias_pct = temporal_bias / ((len(pred_classes) - window_size) // window_size) * 100
            print(f"   - Ventanas con sesgo temporal: {temporal_bias_pct:.1f}%")
            
            if temporal_bias_pct > 30:
                bias_score += 1
            
            # Resultado final
            print(f"\n🎯 Resultado de validación de sesgo:")
            print(f"   - Score de sesgo: {bias_score}/4")
            
            if bias_score <= 1:
                print(f"✅ MODELO APROBADO - Sesgo mínimo detectado")
                return True
            else:
                print(f"❌ MODELO REQUIERE MEJORAS - Sesgo significativo detectado")
                return False
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def save_advanced_artifacts(self, model: tf.keras.Model, training_results: Dict):
        """Guarda modelo y artefactos avanzados"""
        try:
            print(f"\n💾 Guardando modelo y artefactos avanzados...")
            
            # Guardar modelo
            model.save(self.model_save_path)
            print(f"✅ Modelo guardado: {self.model_save_path}")
            
            # Guardar scaler
            with open(self.scaler_save_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"✅ Scaler guardado: {self.scaler_save_path}")
            
            # Guardar historial de entrenamiento (sin float32)
            history_data = {
                'cv_results': [
                    {
                        'fold': r['fold'],
                        'val_accuracy': float(r['val_accuracy']),
                        'val_loss': float(r['val_loss'])
                    } for r in training_results['cv_results']
                ],
                'avg_accuracy': float(training_results['avg_accuracy']),
                'std_accuracy': float(training_results['std_accuracy']),
                'training_timestamp': datetime.now().isoformat(),
                'model_config': {
                    'lookback_window': self.lookback_window,
                    'features': self.expected_features,
                    'classes': self.class_names,
                    'architecture': 'Advanced TCN with Attention',
                    'loss_function': 'Focal Loss'
                }
            }
            
            with open(self.history_save_path, 'w') as f:
                json.dump(history_data, f, indent=2)
            print(f"✅ Historial guardado: {self.history_save_path}")
            
        except Exception as e:
            print(f"❌ Error guardando: {e}")
    
    async def run_advanced_retraining(self):
        """Ejecuta el re-entrenamiento avanzado completo"""
        print("🚀 Iniciando re-entrenamiento AVANZADO del modelo TCN Anti-Bias")
        print("="*80)
        
        # 1. Configurar Binance
        if not self.setup_binance_client():
            return False
        
        # 2. Recolectar datos multi-timeframe
        df = self.collect_multi_timeframe_data()
        if df is None:
            return False
        
        # 3. Crear regímenes diversos
        df_regimes = self.create_diverse_market_regimes(df)
        
        # 4. Crear features comprehensivas
        df_features = self.create_comprehensive_features(df_regimes)
        if df_features is None:
            return False
        
        # 5. Crear labels avanzadas
        df_labeled = self.create_advanced_labels(df_features)
        
        # 6. Preparar datos para entrenamiento
        X_features, X_regimes, y = self.prepare_training_data(df_labeled)
        if X_features is None:
            return False
        
        # 7. Aplicar SMOTE para balancear
        X_features_balanced, X_regimes_balanced, y_balanced = self.apply_smote_balancing(X_features, X_regimes, y)
        
        # 8. Crear modelo avanzado
        model = self.create_advanced_tcn_model()
        if model is None:
            return False
        
        # 9. Entrenar con cross-validation
        training_results = self.train_with_cross_validation(model, X_features_balanced, X_regimes_balanced, y_balanced)
        if training_results is None:
            return False
        
        # 10. Validación rigurosa de sesgo
        final_model = training_results['final_model']
        is_bias_free = self.rigorous_bias_validation(final_model, X_features_balanced, X_regimes_balanced, y_balanced)
        
        # 11. Guardar artefactos
        self.save_advanced_artifacts(final_model, training_results)
        
        print("\n" + "="*80)
        if is_bias_free:
            print("🎉 RE-ENTRENAMIENTO AVANZADO EXITOSO!")
            print(f"✅ Modelo sin sesgo significativo")
        else:
            print("⚠️ RE-ENTRENAMIENTO COMPLETADO CON ADVERTENCIAS")
            print(f"❌ Modelo aún presenta algo de sesgo")
        
        print(f"📋 RESUMEN:")
        print(f"   - Modelo guardado: {self.model_save_path}")
        print(f"   - CV Accuracy: {training_results['avg_accuracy']:.4f} ± {training_results['std_accuracy']:.4f}")
        print(f"   - Features: {self.expected_features}")
        print(f"   - Arquitectura: Advanced TCN + Attention + Focal Loss")
        print(f"   - Técnicas: SMOTE + Cross-Validation + Multi-timeframe")
        
        return is_bias_free


async def main():
    print("🚀 Advanced TCN Anti-Bias Retrainer")
    print("="*80)
    
    retrainer = AdvancedTCNRetrainer()
    
    try:
        success = await retrainer.run_advanced_retraining()
        
        if success:
            print("\n✅ ¡Re-entrenamiento avanzado exitoso!")
            print("🎯 Modelo TCN Anti-Bias avanzado listo")
        else:
            print("\n⚠️ Re-entrenamiento completado con advertencias.")
            print("🔧 Considerar ajustes adicionales")
    
    except Exception as e:
        print(f"\n💥 Error: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 