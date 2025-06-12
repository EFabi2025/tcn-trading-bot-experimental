#!/usr/bin/env python3
"""
🔷 ETH TCN MODEL - Modelo específico para Ethereum
Enfoque paso a paso y simple
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

class EthereumTCN:
    """Modelo TCN específico para Ethereum"""
    
    def __init__(self):
        self.pair_symbol = "ETHUSDT"
        self.pair_name = "Ethereum"
        self.lookback_window = 60
        self.expected_features = 30  # Simplificado
        self.class_names = ['SELL', 'HOLD', 'BUY']
        
        # Parámetros específicos para ETH
        self.volatility_threshold = 0.008  # 0.8% para ETH
        
        print("🔷 ETHEREUM TCN MODEL")
        print("="*50)
        print(f"📊 Par: {self.pair_symbol}")
        print(f"🎯 Threshold: {self.volatility_threshold*100:.1f}%")
        print(f"📈 Features: {self.expected_features}")
        print(f"⏰ Ventana: {self.lookback_window} períodos")
    
    def step1_connect_binance(self):
        """Paso 1: Conectar a Binance"""
        try:
            print("\n📋 PASO 1: CONECTANDO A BINANCE")
            print("-" * 40)
            
            self.client = BinanceClient()
            
            # Test de conectividad
            ticker = self.client.get_symbol_ticker(symbol=self.pair_symbol)
            price = float(ticker['price'])
            
            print(f"✅ Conectado exitosamente")
            print(f"💰 Precio actual ETH: ${price:,.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error conectando: {e}")
            return False
    
    def step2_collect_data(self):
        """Paso 2: Recopilar datos de ETH"""
        try:
            print("\n📋 PASO 2: RECOPILANDO DATOS ETH")
            print("-" * 40)
            
            # Descargar datos históricos
            print("📥 Descargando datos históricos...")
            klines = self.client.get_historical_klines(
                symbol=self.pair_symbol,
                interval='5m',
                limit=800  # ~66 horas de datos
            )
            
            if not klines:
                print("❌ No se obtuvieron datos")
                return None
            
            # Crear DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convertir tipos
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()
            
            print(f"✅ Datos obtenidos: {len(df)} períodos")
            print(f"📊 Rango ETH: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
            print(f"📊 Volumen promedio: {df['volume'].mean():,.0f}")
            
            # Guardar datos para debugging
            self.raw_data = df
            
            return df
            
        except Exception as e:
            print(f"❌ Error obteniendo datos: {e}")
            return None
    
    def step3_create_features(self, df):
        """Paso 3: Crear features técnicas para ETH"""
        try:
            print("\n📋 PASO 3: CREANDO FEATURES ETH")
            print("-" * 40)
            
            df = df.copy()
            features = []
            
            print("🔧 Creando indicadores técnicos...")
            
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
            
            # Asegurar exactamente 30 features
            features = features[:30]
            
            # Limpiar datos
            df = df.dropna()
            
            print(f"✅ Features creadas: {len(features)}")
            print(f"📊 Datos limpios: {len(df)} períodos")
            
            # Verificar integridad
            for feature in features:
                if feature not in df.columns:
                    print(f"⚠️ Feature faltante: {feature}")
                    df[feature] = 0.0
            
            self.feature_list = features
            return df[['timestamp'] + features]
            
        except Exception as e:
            print(f"❌ Error creando features: {e}")
            return None
    
    def step4_create_labels(self, df):
        """Paso 4: Crear labels para ETH"""
        try:
            print("\n📋 PASO 4: CREANDO LABELS ETH")
            print("-" * 40)
            
            df = df.copy()
            
            print(f"🎯 Usando threshold: {self.volatility_threshold*100:.1f}%")
            
            labels = []
            for i in range(len(df)):
                if i >= len(df) - 6:  # No hay datos futuros
                    labels.append(1)  # HOLD
                    continue
                
                current_price = df['close'].iloc[i]
                future_price = df['close'].iloc[i + 6]  # 30 min después (6 períodos de 5m)
                
                price_change = (future_price - current_price) / current_price
                
                if price_change < -self.volatility_threshold:
                    labels.append(0)  # SELL
                elif price_change > self.volatility_threshold:
                    labels.append(2)  # BUY
                else:
                    labels.append(1)  # HOLD
            
            df['label'] = labels
            
            # Verificar distribución
            label_counts = Counter(labels)
            total = len(labels)
            
            print("📊 Distribución de labels:")
            for i, name in enumerate(self.class_names):
                count = label_counts[i]
                pct = count / total * 100
                print(f"   - {name}: {count} ({pct:.1f}%)")
            
            # Calcular bias score
            percentages = [label_counts[i]/total for i in range(3)]
            bias_score = (max(percentages) - min(percentages)) * 10
            
            print(f"📏 Bias Score: {bias_score:.1f}/10")
            
            return df
            
        except Exception as e:
            print(f"❌ Error creando labels: {e}")
            return None
    
    def step5_prepare_training_data(self, df):
        """Paso 5: Preparar datos para entrenamiento"""
        try:
            print("\n📋 PASO 5: PREPARANDO DATOS DE ENTRENAMIENTO")
            print("-" * 40)
            
            feature_columns = self.feature_list
            
            print("🔄 Normalizando datos...")
            # Normalizar features
            scaler = MinMaxScaler()
            df_scaled = df.copy()
            df_scaled[feature_columns] = scaler.fit_transform(df[feature_columns])
            
            # Crear directorio
            os.makedirs('models/eth', exist_ok=True)
            
            # Guardar scaler
            with open('models/eth/scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            
            print("🔢 Creando secuencias temporales...")
            # Crear secuencias
            X = []
            y = []
            
            for i in range(self.lookback_window, len(df_scaled)):
                # Secuencia de features
                feature_seq = df_scaled[feature_columns].iloc[i-self.lookback_window:i].values
                X.append(feature_seq)
                
                # Label
                y.append(int(df_scaled['label'].iloc[i]))
            
            X = np.array(X)
            y = np.array(y)
            
            print(f"✅ Datos preparados:")
            print(f"   📊 X shape: {X.shape}")
            print(f"   📊 y shape: {y.shape}")
            
            # Verificar distribución final
            final_counts = Counter(y)
            total = len(y)
            print("📊 Distribución final:")
            for i, name in enumerate(self.class_names):
                count = final_counts[i]
                pct = count / total * 100
                print(f"   - {name}: {count} ({pct:.1f}%)")
            
            return X, y
            
        except Exception as e:
            print(f"❌ Error preparando datos: {e}")
            return None, None
    
    def step6_create_model(self):
        """Paso 6: Crear modelo TCN simple"""
        try:
            print("\n📋 PASO 6: CREANDO MODELO TCN ETH")
            print("-" * 40)
            
            print("🧠 Arquitectura TCN simple...")
            
            model = tf.keras.Sequential([
                # Input
                tf.keras.layers.Input(shape=(self.lookback_window, self.expected_features)),
                
                # TCN layers simples
                tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='causal', activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.2),
                
                tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=2, padding='causal', activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                
                tf.keras.layers.Conv1D(filters=32, kernel_size=3, dilation_rate=4, padding='causal', activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                
                # Pooling
                tf.keras.layers.GlobalAveragePooling1D(),
                
                # Dense layers
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.4),
                
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                
                # Output
                tf.keras.layers.Dense(3, activation='softmax')
            ])
            
            # Compilar
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print(f"✅ Modelo creado: {model.count_params():,} parámetros")
            
            return model
            
        except Exception as e:
            print(f"❌ Error creando modelo: {e}")
            return None
    
    def step7_train_model(self, model, X, y):
        """Paso 7: Entrenar modelo"""
        try:
            print("\n📋 PASO 7: ENTRENANDO MODELO ETH")
            print("-" * 40)
            
            # Split datos
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            print(f"📊 Train: {len(X_train)}, Val: {len(X_val)}")
            
            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.00001,
                    verbose=1
                )
            ]
            
            print("🔥 Iniciando entrenamiento...")
            
            # Entrenar
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            print("✅ Entrenamiento completado")
            
            return history, X_val, y_val
            
        except Exception as e:
            print(f"❌ Error entrenando: {e}")
            return None, None, None
    
    def step8_evaluate_model(self, model, X_val, y_val):
        """Paso 8: Evaluar modelo"""
        try:
            print("\n📋 PASO 8: EVALUANDO MODELO ETH")
            print("-" * 40)
            
            # Predicciones
            predictions = model.predict(X_val, verbose=0)
            pred_classes = np.argmax(predictions, axis=1)
            
            # Distribución
            pred_counts = Counter(pred_classes)
            total_preds = len(pred_classes)
            
            print("📊 Distribución de predicciones:")
            for i, name in enumerate(self.class_names):
                count = pred_counts[i]
                pct = count / total_preds * 100
                print(f"   - {name}: {count} ({pct:.1f}%)")
            
            # Bias score
            percentages = [pred_counts[i]/total_preds for i in range(3)]
            bias_score = (max(percentages) - min(percentages)) * 10
            
            print(f"\n🎯 BIAS SCORE ETH: {bias_score:.1f}/10")
            
            if bias_score < 2.0:
                print("🏆 ¡BALANCE PERFECTO!")
            elif bias_score < 4.0:
                print("✅ Balance excelente")
            else:
                print("⚠️ Requiere mejora")
            
            return bias_score, pred_counts
            
        except Exception as e:
            print(f"❌ Error evaluando: {e}")
            return None, None
    
    def step9_save_model(self, model, bias_score, pred_counts):
        """Paso 9: Guardar modelo"""
        try:
            print("\n📋 PASO 9: GUARDANDO MODELO ETH")
            print("-" * 40)
            
            # Guardar modelo
            model_path = 'models/eth/eth_tcn_model.h5'
            model.save(model_path)
            
            # Metadata
            metadata = {
                'pair': self.pair_symbol,
                'pair_name': self.pair_name,
                'model_type': 'ETH Specific TCN',
                'features': len(self.feature_list),
                'bias_score': float(bias_score),
                'volatility_threshold': self.volatility_threshold,
                'final_distribution': {
                    'SELL': float(pred_counts[0]/sum(pred_counts.values())),
                    'HOLD': float(pred_counts[1]/sum(pred_counts.values())),
                    'BUY': float(pred_counts[2]/sum(pred_counts.values()))
                },
                'training_date': datetime.now().isoformat(),
                'features_list': self.feature_list
            }
            
            with open('models/eth/metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Modelo guardado: {model_path}")
            print(f"✅ Metadata guardado: models/eth/metadata.json")
            
            return True
            
        except Exception as e:
            print(f"❌ Error guardando: {e}")
            return False
    
    def train_eth_model(self):
        """Método principal que ejecuta todos los pasos"""
        print("🔷 ENTRENAMIENTO COMPLETO ETH TCN")
        print("="*60)
        
        # Ejecutar todos los pasos
        if not self.step1_connect_binance():
            return False
        
        df = self.step2_collect_data()
        if df is None:
            return False
        
        df = self.step3_create_features(df)
        if df is None:
            return False
        
        df = self.step4_create_labels(df)
        if df is None:
            return False
        
        X, y = self.step5_prepare_training_data(df)
        if X is None:
            return False
        
        model = self.step6_create_model()
        if model is None:
            return False
        
        history, X_val, y_val = self.step7_train_model(model, X, y)
        if history is None:
            return False
        
        bias_score, pred_counts = self.step8_evaluate_model(model, X_val, y_val)
        if bias_score is None:
            return False
        
        if not self.step9_save_model(model, bias_score, pred_counts):
            return False
        
        print("\n🏆 ¡MODELO ETH COMPLETADO EXITOSAMENTE!")
        print("="*60)
        return True


if __name__ == "__main__":
    # Entrenar modelo ETH
    eth_trainer = EthereumTCN()
    success = eth_trainer.train_eth_model()
    
    if success:
        print("\n✅ Entrenamiento ETH exitoso")
    else:
        print("\n❌ Error en entrenamiento ETH") 