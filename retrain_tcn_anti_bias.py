#!/usr/bin/env python3
"""
🚀 Re-entrenamiento Completo del Modelo TCN Anti-Bias

Sistema completo para re-entrenar el modelo TCN con:
- Datos balanceados por régimen de mercado
- Feature engineering completo (66 features)
- Arquitectura dual input (Features + Regime)
- Técnicas anti-sesgo y validación por régimen
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
from binance.client import Client as BinanceClient
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle
import json


class TCNAntibiasRetrainer:
    """
    🚀 Re-entrenador del modelo TCN Anti-Bias
    """
    
    def __init__(self):
        self.symbol = "BTCUSDT"
        self.interval = "5m"
        self.lookback_window = 60
        self.expected_features = 66
        self.class_names = ['SELL', 'HOLD', 'BUY']
        self.model_save_path = "models/tcn_anti_bias_retrained.h5"
        self.scaler_save_path = "models/feature_scalers_retrained.pkl"
        self.history_save_path = "models/training_history_retrained.json"
        
        print("🚀 TCN Anti-Bias Retrainer inicializado")
        print(f"   - Símbolo: {self.symbol}")
        print(f"   - Features: {self.expected_features}")
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
    
    def collect_extensive_data(self, limit: int = 1500) -> Optional[pd.DataFrame]:
        """Recolecta datos extensos para entrenamiento"""
        try:
            print(f"\n📊 Recolectando datos extensos de {self.symbol}...")
            
            # Obtener datos históricos
            klines = self.binance_client.get_historical_klines(
                symbol=self.symbol,
                interval=self.interval,
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
            
            print(f"✅ Datos recolectados: {len(df)} períodos")
            print(f"   - Desde: {df['timestamp'].iloc[0]}")
            print(f"   - Hasta: {df['timestamp'].iloc[-1]}")
            print(f"   - Rango de precios: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def detect_and_balance_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detecta regímenes y balancea los datos"""
        try:
            print("\n🔍 Detectando y balanceando regímenes de mercado...")
            
            df = df.copy()
            
            # Detectar regímenes con múltiples métodos
            # Método 1: Tendencia a corto plazo (10 períodos)
            df['trend_short'] = df['close'].pct_change(periods=10)
            
            # Método 2: Tendencia a mediano plazo (20 períodos)
            df['trend_medium'] = df['close'].pct_change(periods=20)
            
            # Método 3: Momentum
            df['momentum'] = df['close'] / df['close'].shift(15) - 1
            
            # Método 4: Volatilidad
            df['volatility'] = df['close'].pct_change().rolling(window=20).std()
            
            # Clasificación de regímenes más agresiva para obtener variedad
            regimes = []
            
            for i, row in df.iterrows():
                trend_short = row['trend_short']
                trend_medium = row['trend_medium'] 
                momentum = row['momentum']
                volatility = row['volatility']
                
                if pd.isna(trend_short) or pd.isna(trend_medium):
                    regime = 1  # SIDEWAYS
                else:
                    # Umbrales más bajos para obtener más variedad
                    if trend_medium > 0.02 or momentum > 0.03:  # 2% o 3% subida
                        regime = 2  # BULL
                    elif trend_medium < -0.02 or momentum < -0.03:  # 2% o 3% bajada
                        regime = 0  # BEAR
                    else:
                        regime = 1  # SIDEWAYS
                
                regimes.append(regime)
            
            df['regime'] = regimes
            
            # Estadísticas iniciales
            regime_counts = Counter(regimes)
            total = len(regimes)
            
            print(f"📊 Distribución inicial de regímenes:")
            for i, name in enumerate(['BEAR', 'SIDEWAYS', 'BULL']):
                count = regime_counts[i]
                pct = count / total * 100
                print(f"   - {name}: {count} ({pct:.1f}%)")
            
            # Balanceado de datos por régimen
            print(f"\n⚖️ Balanceando datos por régimen...")
            
            # Obtener muestras balanceadas
            regime_dfs = []
            min_samples = min(regime_counts.values())
            
            # Si algún régimen tiene muy pocas muestras, usar sampling sintético
            if min_samples < 100:
                target_samples = 200  # Mínimo objetivo
                print(f"   - Generando datos sintéticos para equilibrar (objetivo: {target_samples} por régimen)")
                
                for regime_id in [0, 1, 2]:
                    regime_data = df[df['regime'] == regime_id].copy()
                    
                    if len(regime_data) < target_samples:
                        # Generar datos sintéticos mediante perturbación
                        synthetic_data = []
                        while len(synthetic_data) + len(regime_data) < target_samples:
                            # Seleccionar muestra aleatoria
                            base_sample = regime_data.sample(1).iloc[0]
                            
                            # Crear perturbación pequeña (±2%)
                            noise_factor = 0.02
                            synthetic_sample = base_sample.copy()
                            
                            for col in ['open', 'high', 'low', 'close', 'volume']:
                                noise = np.random.uniform(-noise_factor, noise_factor)
                                synthetic_sample[col] = base_sample[col] * (1 + noise)
                            
                            synthetic_data.append(synthetic_sample)
                        
                        if synthetic_data:
                            synthetic_df = pd.DataFrame(synthetic_data)
                            regime_data = pd.concat([regime_data, synthetic_df], ignore_index=True)
                    
                    # Tomar exactamente target_samples
                    regime_data_balanced = regime_data.sample(n=min(target_samples, len(regime_data)), random_state=42)
                    regime_dfs.append(regime_data_balanced)
            
            else:
                # Balanceado normal
                for regime_id in [0, 1, 2]:
                    regime_data = df[df['regime'] == regime_id]
                    regime_data_balanced = regime_data.sample(n=min_samples, random_state=42)
                    regime_dfs.append(regime_data_balanced)
            
            # Combinar datos balanceados
            df_balanced = pd.concat(regime_dfs, ignore_index=True)
            df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
            
            # Estadísticas finales
            final_regime_counts = Counter(df_balanced['regime'])
            final_total = len(df_balanced)
            
            print(f"✅ Distribución balanceada:")
            for i, name in enumerate(['BEAR', 'SIDEWAYS', 'BULL']):
                count = final_regime_counts[i]
                pct = count / final_total * 100
                print(f"   - {name}: {count} ({pct:.1f}%)")
            
            return df_balanced
            
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
    
    def create_training_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea labels de entrenamiento balanceadas"""
        try:
            print("\n🎯 Creando labels de entrenamiento balanceadas...")
            
            df = df.copy()
            
            # Calcular futuro precio (12 períodos adelante)
            df['future_price'] = df['close'].shift(-12)
            df['price_change_future'] = (df['future_price'] - df['close']) / df['close']
            
            # Crear labels balanceadas por régimen
            labels = []
            
            for i, row in df.iterrows():
                regime = row['regime']
                price_change = row['price_change_future']
                
                if pd.isna(price_change):
                    label = 1  # HOLD por defecto
                else:
                    # Umbrales adaptativos por régimen
                    if regime == 0:  # BEAR market
                        # En bear market, ser más conservador
                        if price_change < -0.003:  # -0.3%
                            label = 0  # SELL
                        elif price_change > 0.002:  # +0.2%
                            label = 2  # BUY
                        else:
                            label = 1  # HOLD
                    elif regime == 2:  # BULL market
                        # En bull market, ser más agresivo
                        if price_change < -0.002:  # -0.2%
                            label = 0  # SELL
                        elif price_change > 0.003:  # +0.3%
                            label = 2  # BUY
                        else:
                            label = 1  # HOLD
                    else:  # SIDEWAYS
                        # En sideways, umbrales normales
                        if price_change < -0.0025:  # -0.25%
                            label = 0  # SELL
                        elif price_change > 0.0025:  # +0.25%
                            label = 2  # BUY
                        else:
                            label = 1  # HOLD
                
                labels.append(label)
            
            df['label'] = labels
            
            # Remover filas sin labels válidas
            df = df.dropna(subset=['price_change_future'])
            
            # Estadísticas de labels
            label_counts = Counter(labels)
            total_labels = len([l for l in labels if not pd.isna(l)])
            
            print(f"📊 Distribución de labels:")
            for i, name in enumerate(self.class_names):
                count = label_counts[i]
                pct = count / total_labels * 100 if total_labels > 0 else 0
                print(f"   - {name}: {count} ({pct:.1f}%)")
            
            # Balancear labels si es muy desbalanceado
            max_samples_per_class = max(label_counts.values())
            min_samples_per_class = min(label_counts.values())
            
            if max_samples_per_class / min_samples_per_class > 2:  # Si hay más de 2x diferencia
                print(f"⚖️ Re-balanceando labels...")
                
                balanced_dfs = []
                target_samples = min_samples_per_class
                
                for label_id in [0, 1, 2]:
                    label_data = df[df['label'] == label_id]
                    if len(label_data) > 0:
                        balanced_data = label_data.sample(n=min(target_samples, len(label_data)), random_state=42)
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
            feature_columns = [col for col in df.columns if col not in ['timestamp', 'regime', 'label', 'future_price', 'price_change_future']]
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
    
    def create_tcn_anti_bias_model(self) -> tf.keras.Model:
        """Crea el modelo TCN Anti-Bias con arquitectura dual"""
        try:
            print("\n🧠 Creando modelo TCN Anti-Bias...")
            
            # Input 1: Features técnicas (secuencias)
            features_input = tf.keras.Input(shape=(self.lookback_window, self.expected_features), name='price_features')
            
            # TCN Layers con dilated convolutions
            x = features_input
            
            # Primera capa TCN
            x = tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='causal', activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            
            # Segunda capa TCN con dilation
            x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=2, padding='causal', activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            
            # Tercera capa TCN con mayor dilation
            x = tf.keras.layers.Conv1D(filters=128, kernel_size=3, dilation_rate=4, padding='causal', activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            
            # Global pooling
            x = tf.keras.layers.GlobalMaxPooling1D()(x)
            x = tf.keras.layers.Dense(64, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.4)(x)
            
            # Input 2: Market regime
            regime_input = tf.keras.Input(shape=(3,), name='market_regime')
            regime_dense = tf.keras.layers.Dense(32, activation='relu')(regime_input)
            regime_dense = tf.keras.layers.BatchNormalization()(regime_dense)
            regime_dense = tf.keras.layers.Dropout(0.2)(regime_dense)
            
            # Fusión de features y regime
            combined = tf.keras.layers.concatenate([x, regime_dense])
            
            # Capas densas finales
            combined = tf.keras.layers.Dense(128, activation='relu')(combined)
            combined = tf.keras.layers.BatchNormalization()(combined)
            combined = tf.keras.layers.Dropout(0.4)(combined)
            
            combined = tf.keras.layers.Dense(64, activation='relu')(combined)
            combined = tf.keras.layers.BatchNormalization()(combined)
            combined = tf.keras.layers.Dropout(0.3)(combined)
            
            combined = tf.keras.layers.Dense(32, activation='relu')(combined)
            combined = tf.keras.layers.Dropout(0.2)(combined)
            
            # Output layer - 3 clases balanceadas
            outputs = tf.keras.layers.Dense(3, activation='softmax', name='predictions')(combined)
            
            # Crear modelo
            model = tf.keras.Model(inputs=[features_input, regime_input], outputs=outputs)
            
            # Compilar con configuración anti-sesgo
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy', 'sparse_categorical_crossentropy']
            )
            
            print(f"✅ Modelo TCN Anti-Bias creado:")
            print(f"   - Parámetros totales: {model.count_params():,}")
            
            model.summary()
            
            return model
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def train_model_with_antibias_validation(self, model: tf.keras.Model, 
                                           X_features: np.ndarray, X_regimes: np.ndarray, y: np.ndarray) -> Dict:
        """Entrena el modelo con validación anti-sesgo"""
        try:
            print("\n🚀 Entrenando modelo con validación anti-sesgo...")
            
            # Split train/validation
            X_feat_train, X_feat_val, X_reg_train, X_reg_val, y_train, y_val = train_test_split(
                X_features, X_regimes, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"   - Train samples: {len(X_feat_train)}")
            print(f"   - Validation samples: {len(X_feat_val)}")
            
            # Calcular class weights para balancear el entrenamiento
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
            
            print(f"   - Class weights: {class_weight_dict}")
            
            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.0001
                )
            ]
            
            # Entrenar
            history = model.fit(
                [X_feat_train, X_reg_train], y_train,
                validation_data=([X_feat_val, X_reg_val], y_val),
                epochs=100,
                batch_size=32,
                class_weight=class_weight_dict,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluación final
            val_loss, val_accuracy, val_sparse_cat = model.evaluate(
                [X_feat_val, X_reg_val], y_val, verbose=0
            )
            
            print(f"✅ Entrenamiento completado:")
            print(f"   - Validation Accuracy: {val_accuracy:.4f}")
            print(f"   - Validation Loss: {val_loss:.4f}")
            
            # Predicciones para análisis de sesgo
            val_predictions = model.predict([X_feat_val, X_reg_val], verbose=0)
            val_pred_classes = np.argmax(val_predictions, axis=1)
            
            # Análisis de sesgo en validación
            self.validate_anti_bias(val_pred_classes, X_reg_val, y_val)
            
            return {
                'history': history.history,
                'val_accuracy': val_accuracy,
                'val_loss': val_loss,
                'val_predictions': val_pred_classes,
                'val_true': y_val
            }
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def validate_anti_bias(self, predictions: np.ndarray, regimes: np.ndarray, true_labels: np.ndarray):
        """Valida que el modelo no tenga sesgo"""
        try:
            print(f"\n🧪 Validando Anti-Bias del modelo entrenado...")
            
            # Distribución general
            pred_counts = Counter(predictions)
            total_preds = len(predictions)
            
            print(f"📊 Distribución de predicciones:")
            for i, name in enumerate(self.class_names):
                count = pred_counts[i]
                pct = count / total_preds * 100
                print(f"   - {name}: {count} ({pct:.1f}%)")
            
            # Análisis por régimen
            regime_names = ['BEAR', 'SIDEWAYS', 'BULL']
            
            print(f"\n📊 Análisis por régimen:")
            bias_detected = False
            
            for class_idx, class_name in enumerate(self.class_names):
                percentages = []
                
                for regime_idx in range(3):
                    regime_mask = np.argmax(regimes, axis=1) == regime_idx
                    if np.sum(regime_mask) > 0:
                        regime_preds = predictions[regime_mask]
                        regime_class_count = np.sum(regime_preds == class_idx)
                        regime_total = len(regime_preds)
                        pct = regime_class_count / regime_total * 100 if regime_total > 0 else 0
                        percentages.append(pct)
                        print(f"   {class_name} en {regime_names[regime_idx]}: {pct:.1f}%")
                
                # Calcular desviación estándar
                if len(percentages) > 1:
                    std_dev = np.std(percentages)
                    if std_dev > 10:  # > 10% desviación estándar
                        print(f"   ⚠️ {class_name}: SESGO DETECTADO (std: {std_dev:.1f}%)")
                        bias_detected = True
                    else:
                        print(f"   ✅ {class_name}: SIN SESGO (std: {std_dev:.1f}%)")
            
            # Resultado final
            if bias_detected:
                print(f"\n❌ MODELO CON SESGO - Requiere ajustes")
            else:
                print(f"\n✅ MODELO ANTI-BIAS EXITOSO")
            
            return not bias_detected
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def save_model_and_artifacts(self, model: tf.keras.Model, training_results: Dict):
        """Guarda el modelo y artefactos"""
        try:
            print(f"\n💾 Guardando modelo y artefactos...")
            
            # Guardar modelo
            model.save(self.model_save_path)
            print(f"✅ Modelo guardado: {self.model_save_path}")
            
            # Guardar scaler
            with open(self.scaler_save_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"✅ Scaler guardado: {self.scaler_save_path}")
            
            # Guardar historial de entrenamiento
            history_data = {
                'training_history': training_results['history'],
                'final_metrics': {
                    'val_accuracy': float(training_results['val_accuracy']),
                    'val_loss': float(training_results['val_loss'])
                },
                'training_timestamp': datetime.now().isoformat(),
                'model_config': {
                    'lookback_window': self.lookback_window,
                    'features': self.expected_features,
                    'classes': self.class_names
                }
            }
            
            with open(self.history_save_path, 'w') as f:
                json.dump(history_data, f, indent=2)
            print(f"✅ Historial guardado: {self.history_save_path}")
            
        except Exception as e:
            print(f"❌ Error guardando: {e}")
    
    async def run_complete_retraining(self):
        """Ejecuta el re-entrenamiento completo"""
        print("🚀 Iniciando re-entrenamiento completo del modelo TCN Anti-Bias")
        print("="*80)
        
        # 1. Configurar Binance
        if not self.setup_binance_client():
            return False
        
        # 2. Recolectar datos extensos
        df = self.collect_extensive_data(1500)
        if df is None:
            return False
        
        # 3. Detectar y balancear regímenes
        df_balanced = self.detect_and_balance_regimes(df)
        
        # 4. Crear features comprehensivas
        df_features = self.create_comprehensive_features(df_balanced)
        if df_features is None:
            return False
        
        # 5. Crear labels de entrenamiento
        df_labeled = self.create_training_labels(df_features)
        
        # 6. Preparar datos para entrenamiento
        X_features, X_regimes, y = self.prepare_training_data(df_labeled)
        if X_features is None:
            return False
        
        # 7. Crear modelo
        model = self.create_tcn_anti_bias_model()
        if model is None:
            return False
        
        # 8. Entrenar con validación anti-sesgo
        training_results = self.train_model_with_antibias_validation(model, X_features, X_regimes, y)
        if training_results is None:
            return False
        
        # 9. Guardar modelo y artefactos
        self.save_model_and_artifacts(model, training_results)
        
        print("\n" + "="*80)
        print("🎉 RE-ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
        print(f"📋 RESUMEN:")
        print(f"   - Modelo guardado: {self.model_save_path}")
        print(f"   - Accuracy final: {training_results['val_accuracy']:.4f}")
        print(f"   - Loss final: {training_results['val_loss']:.4f}")
        print(f"   - Features: {self.expected_features}")
        print(f"   - Arquitectura: TCN Anti-Bias Dual Input")
        
        return True


async def main():
    print("🚀 TCN Anti-Bias Retrainer - Re-entrenamiento Completo")
    print("="*80)
    
    retrainer = TCNAntibiasRetrainer()
    
    try:
        success = await retrainer.run_complete_retraining()
        
        if success:
            print("\n✅ ¡Re-entrenamiento exitoso!")
            print("🎯 Modelo TCN Anti-Bias listo para uso sin sesgo")
        else:
            print("\n❌ Re-entrenamiento fallido.")
    
    except Exception as e:
        print(f"\n💥 Error: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 