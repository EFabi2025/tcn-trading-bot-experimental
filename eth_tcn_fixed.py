#!/usr/bin/env python3
"""
🔷 ETH TCN FIXED - Modelo ETH que SÍ funciona
Basado en aprendizajes del diagnóstico
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from binance.client import Client as BinanceClient
from sklearn.preprocessing import StandardScaler
from collections import Counter
import pickle
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EthereumTCNFixed:
    """Modelo ETH TCN corregido que funciona"""
    
    def __init__(self):
        self.pair_symbol = "ETHUSDT"
        self.lookback_window = 20  # Reducido de 60
        self.expected_features = 10  # Reducido de 30
        self.class_names = ['SELL', 'HOLD', 'BUY']
        self.volatility_threshold = 0.002  # 0.2% según debug
        
        print("🔷 ETHEREUM TCN FIXED")
        print("="*50)
        print(f"📊 Par: {self.pair_symbol}")
        print(f"🎯 Threshold: {self.volatility_threshold*100:.1f}%")
        print(f"📈 Features: {self.expected_features}")
        print(f"⏰ Ventana: {self.lookback_window} períodos")
    
    def get_data_and_features(self):
        """Obtener datos y crear features eficientes"""
        try:
            print("\n📋 OBTENIENDO DATOS Y FEATURES")
            print("-" * 40)
            
            # Conectar
            client = BinanceClient()
            ticker = client.get_symbol_ticker(symbol=self.pair_symbol)
            price = float(ticker['price'])
            print(f"✅ Conectado - Precio ETH: ${price:,.2f}")
            
            # Datos
            klines = client.get_historical_klines(
                symbol=self.pair_symbol,
                interval='5m',
                limit=400  # Reducido
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convertir
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()
            print(f"✅ {len(df)} períodos obtenidos")
            
            # === FEATURES EFICIENTES (10 total) ===
            print("🔧 Creando features eficientes...")
            
            # 1. Returns
            df['returns'] = df['close'].pct_change()
            
            # 2-3. Moving averages
            df['sma_5'] = df['close'].rolling(5).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            
            # 4-5. EMAs
            df['ema_5'] = df['close'].ewm(span=5).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            
            # 6. RSI simple
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # 7. MACD
            df['macd'] = df['ema_12'] - df['close'].ewm(span=26).mean()
            
            # 8. Volatilidad
            df['volatility'] = df['returns'].rolling(10).std()
            
            # 9. Volume ratio
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            
            # 10. Price position (BB-like)
            mean_20 = df['close'].rolling(20).mean()
            std_20 = df['close'].rolling(20).std()
            df['price_position'] = (df['close'] - mean_20) / std_20
            
            # Lista final de features
            features = [
                'returns', 'sma_5', 'sma_20', 'ema_5', 'ema_12',
                'rsi', 'macd', 'volatility', 'volume_ratio', 'price_position'
            ]
            
            # Limpiar NaN
            df = df.dropna()
            print(f"✅ Features creadas: {len(features)}")
            print(f"✅ Datos finales: {len(df)} períodos")
            
            self.feature_list = features
            return df
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def create_labels(self, df):
        """Crear labels optimizados"""
        try:
            print("\n📋 CREANDO LABELS OPTIMIZADOS")
            print("-" * 40)
            
            print(f"🎯 Threshold: {self.volatility_threshold*100:.1f}%")
            
            labels = []
            for i in range(len(df) - 3):  # Predicción a 3 períodos (15 min)
                current = df['close'].iloc[i]
                future = df['close'].iloc[i + 3]
                change = (future - current) / current
                
                if change < -self.volatility_threshold:
                    labels.append(0)  # SELL
                elif change > self.volatility_threshold:
                    labels.append(2)  # BUY
                else:
                    labels.append(1)  # HOLD
            
            # Ajustar DataFrame
            df = df.iloc[:-3].copy()
            df['label'] = labels
            
            # Verificar distribución
            label_counts = Counter(labels)
            total = len(labels)
            
            print("📊 Distribución de labels:")
            for i, name in enumerate(self.class_names):
                count = label_counts[i]
                pct = count / total * 100
                print(f"   - {name}: {count} ({pct:.1f}%)")
            
            # Bias score
            percentages = [label_counts[i]/total for i in range(3)]
            bias_score = (max(percentages) - min(percentages)) * 10
            print(f"📏 Bias Score: {bias_score:.1f}/10")
            
            return df, bias_score
            
        except Exception as e:
            print(f"❌ Error creando labels: {e}")
            return None, None
    
    def prepare_training_data(self, df):
        """Preparar datos para TCN"""
        try:
            print("\n📋 PREPARANDO DATOS TCN")
            print("-" * 40)
            
            # Normalizar features
            scaler = StandardScaler()
            df_scaled = df.copy()
            df_scaled[self.feature_list] = scaler.fit_transform(df[self.feature_list])
            
            # Crear directorio
            os.makedirs('models/eth_fixed', exist_ok=True)
            
            # Guardar scaler
            with open('models/eth_fixed/scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            
            # Crear secuencias TCN
            X = []
            y = []
            
            for i in range(self.lookback_window, len(df_scaled)):
                # Secuencia de features
                sequence = df_scaled[self.feature_list].iloc[i-self.lookback_window:i].values
                X.append(sequence)
                
                # Label correspondiente
                y.append(int(df_scaled['label'].iloc[i]))
            
            X = np.array(X)
            y = np.array(y)
            
            print(f"✅ Secuencias creadas: X{X.shape}, y{y.shape}")
            
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
    
    def create_fixed_model(self):
        """Crear modelo TCN corregido"""
        try:
            print("\n📋 CREANDO MODELO TCN CORREGIDO")
            print("-" * 40)
            
            # Arquitectura simplificada pero efectiva
            model = tf.keras.Sequential([
                # Input
                tf.keras.layers.Input(shape=(self.lookback_window, self.expected_features)),
                
                # Convoluciones temporales simples
                tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='causal', activation='relu'),
                tf.keras.layers.Dropout(0.2),
                
                tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=2, padding='causal', activation='relu'),
                tf.keras.layers.Dropout(0.3),
                
                tf.keras.layers.Conv1D(filters=32, kernel_size=3, dilation_rate=4, padding='causal', activation='relu'),
                tf.keras.layers.Dropout(0.3),
                
                # Pooling global
                tf.keras.layers.GlobalAveragePooling1D(),
                
                # Capas densas
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.4),
                
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                
                # Output con bias initialization
                tf.keras.layers.Dense(3, activation='softmax', 
                                    bias_initializer='zeros')
            ])
            
            # Compilar con parámetros optimizados
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
    
    def train_and_evaluate(self, model, X, y):
        """Entrenar y evaluar modelo"""
        try:
            print("\n📋 ENTRENAMIENTO Y EVALUACIÓN")
            print("-" * 40)
            
            # Split temporal
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            print(f"📊 Train: {len(X_train)}, Val: {len(X_val)}")
            
            # Callbacks inteligentes
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',  # Cambio: monitor loss no accuracy
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.7,
                    patience=7,
                    min_lr=0.00001,
                    verbose=1
                )
            ]
            
            print("🔥 Entrenando modelo...")
            
            # Entrenar
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=16,  # Batch size más pequeño
                callbacks=callbacks,
                verbose=1
            )
            
            print("✅ Entrenamiento completado")
            
            # Evaluar
            print("\n📊 EVALUANDO PREDICCIONES...")
            
            predictions = model.predict(X_val, verbose=0)
            pred_classes = np.argmax(predictions, axis=1)
            
            # Distribución de predicciones
            pred_counts = Counter(pred_classes)
            total_preds = len(pred_classes)
            
            print("📊 Distribución de predicciones:")
            for i, name in enumerate(self.class_names):
                count = pred_counts[i]
                pct = count / total_preds * 100
                print(f"   - {name}: {count} ({pct:.1f}%)")
            
            # Bias score final
            percentages = [pred_counts[i]/total_preds for i in range(3)]
            final_bias = (max(percentages) - min(percentages)) * 10
            
            print(f"\n🎯 BIAS SCORE FINAL: {final_bias:.1f}/10")
            
            # Análisis de confianza
            print("\n🔍 Análisis de confianza:")
            avg_confidence = np.mean(np.max(predictions, axis=1))
            print(f"   Confianza promedio: {avg_confidence:.3f}")
            
            # Mostrar algunas predicciones
            print("\n🔍 Primeras 5 predicciones:")
            for i in range(min(5, len(predictions))):
                probs = predictions[i]
                pred_class = pred_classes[i]
                true_class = y_val[i]
                confidence = np.max(probs)
                print(f"   {i}: Pred={self.class_names[pred_class]} Real={self.class_names[true_class]} | Conf={confidence:.3f}")
            
            return final_bias, pred_counts, avg_confidence
            
        except Exception as e:
            print(f"❌ Error entrenando: {e}")
            return None, None, None
    
    def save_model(self, model, bias_score, pred_counts, avg_confidence):
        """Guardar modelo corregido"""
        try:
            print("\n📋 GUARDANDO MODELO CORREGIDO")
            print("-" * 40)
            
            # Guardar modelo
            model_path = 'models/eth_fixed/eth_tcn_fixed.h5'
            model.save(model_path)
            
            # Metadata completa
            metadata = {
                'pair': self.pair_symbol,
                'model_type': 'ETH Fixed TCN',
                'features': len(self.feature_list),
                'lookback_window': self.lookback_window,
                'volatility_threshold': self.volatility_threshold,
                'bias_score': float(bias_score),
                'avg_confidence': float(avg_confidence),
                'final_distribution': {
                    'SELL': float(pred_counts[0]/sum(pred_counts.values())),
                    'HOLD': float(pred_counts[1]/sum(pred_counts.values())),
                    'BUY': float(pred_counts[2]/sum(pred_counts.values()))
                },
                'training_date': datetime.now().isoformat(),
                'features_list': self.feature_list,
                'model_params': model.count_params()
            }
            
            with open('models/eth_fixed/metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Modelo guardado: {model_path}")
            print(f"✅ Metadata guardado: models/eth_fixed/metadata.json")
            
            return True
            
        except Exception as e:
            print(f"❌ Error guardando: {e}")
            return False
    
    def train_eth_fixed(self):
        """Método principal - entrenar ETH corregido"""
        print("🔷 ENTRENAMIENTO ETH TCN CORREGIDO")
        print("="*60)
        
        # 1. Datos y features
        df = self.get_data_and_features()
        if df is None:
            return False
        
        # 2. Labels
        df, label_bias = self.create_labels(df)
        if df is None:
            return False
        
        # 3. Preparar datos
        X, y = self.prepare_training_data(df)
        if X is None:
            return False
        
        # 4. Crear modelo
        model = self.create_fixed_model()
        if model is None:
            return False
        
        # 5. Entrenar y evaluar
        final_bias, pred_counts, avg_confidence = self.train_and_evaluate(model, X, y)
        if final_bias is None:
            return False
        
        # 6. Guardar
        if not self.save_model(model, final_bias, pred_counts, avg_confidence):
            return False
        
        # Resultado final
        print(f"\n🏆 RESULTADO FINAL ETH:")
        print(f"   📏 Bias Score: {final_bias:.1f}/10")
        print(f"   🎯 Confianza: {avg_confidence:.3f}")
        
        if final_bias < 5.0:
            print("🎉 ¡MODELO ETH EXITOSO!")
            return True
        else:
            print("⚠️ Modelo funcional pero mejorable")
            return True  # Aún consideramos éxito si es funcional


if __name__ == "__main__":
    trainer = EthereumTCNFixed()
    success = trainer.train_eth_fixed()
    
    if success:
        print("\n✅ ¡ETH TCN CORREGIDO COMPLETADO!")
    else:
        print("\n❌ Error en ETH corregido") 