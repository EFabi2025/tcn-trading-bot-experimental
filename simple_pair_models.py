#!/usr/bin/env python3
"""
✅ SIMPLE PAIR MODELS - Modelos que SÍ funcionan
Basado en el éxito de eth_simple_debug.py
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from binance.client import Client as BinanceClient
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import pickle
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SimplePairTrainer:
    """Entrenador de modelos simples por par que funcionan"""
    
    def __init__(self, pair_symbol):
        self.pair_symbol = pair_symbol
        self.pair_name = pair_symbol.replace('USDT', '')
        self.class_names = ['SELL', 'HOLD', 'BUY']
        
        # Configuraciones específicas por par
        self.configs = {
            'ETHUSDT': {
                'threshold': 0.002,  # 0.2% - probado que funciona
                'features': ['returns', 'sma_5', 'sma_20', 'rsi', 'macd', 'volatility'],
                'prediction_periods': 3
            },
            'BTCUSDT': {
                'threshold': 0.0015,  # 0.15% - BTC menos volátil
                'features': ['returns', 'sma_5', 'sma_20', 'rsi', 'macd', 'volatility'],
                'prediction_periods': 4
            }
        }
        
        self.config = self.configs.get(pair_symbol, self.configs['ETHUSDT'])
        
        print(f"✅ SIMPLE {self.pair_name} MODEL")
        print("="*50)
        print(f"📊 Par: {self.pair_symbol}")
        print(f"🎯 Threshold: {self.config['threshold']*100:.1f}%")
        print(f"📈 Features: {len(self.config['features'])}")
        print(f"⏰ Predicción: {self.config['prediction_periods']} períodos")
    
    def get_data_and_features(self):
        """Obtener datos y crear features"""
        try:
            print(f"\n📋 OBTENIENDO DATOS {self.pair_name}")
            print("-" * 40)
            
            # Conectar
            client = BinanceClient()
            ticker = client.get_symbol_ticker(symbol=self.pair_symbol)
            price = float(ticker['price'])
            print(f"✅ Conectado - Precio {self.pair_name}: ${price:,.2f}")
            
            # Datos
            klines = client.get_historical_klines(
                symbol=self.pair_symbol,
                interval='5m',
                limit=250
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
            print(f"✅ {len(df)} períodos obtenidos")
            
            # === FEATURES SIMPLES QUE FUNCIONAN ===
            print("🔧 Creando features...")
            
            # Returns
            df['returns'] = df['close'].pct_change()
            
            # Moving averages
            df['sma_5'] = df['close'].rolling(5).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            
            # Volatilidad
            df['volatility'] = df['returns'].rolling(10).std()
            
            # Limpiar
            df = df.dropna()
            print(f"✅ Features creadas: {self.config['features']}")
            print(f"✅ Datos finales: {len(df)} períodos")
            
            return df
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def create_labels_and_prepare(self, df):
        """Crear labels y preparar datos"""
        try:
            print(f"\n📋 CREANDO LABELS {self.pair_name}")
            print("-" * 40)
            
            threshold = self.config['threshold']
            pred_periods = self.config['prediction_periods']
            
            print(f"🎯 Threshold: {threshold*100:.1f}%")
            print(f"⏰ Predicción: {pred_periods} períodos")
            
            # Crear labels
            labels = []
            for i in range(len(df) - pred_periods):
                current = df['close'].iloc[i]
                future = df['close'].iloc[i + pred_periods]
                change = (future - current) / current
                
                if change < -threshold:
                    labels.append(0)  # SELL
                elif change > threshold:
                    labels.append(2)  # BUY
                else:
                    labels.append(1)  # HOLD
            
            # Ajustar DataFrame
            df = df.iloc[:-pred_periods].copy()
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
            label_bias = (max(percentages) - min(percentages)) * 10
            print(f"📏 Label Bias: {label_bias:.1f}/10")
            
            # Preparar datos de entrenamiento
            features = self.config['features']
            
            # Normalizar
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df[features])
            y = np.array(labels)
            
            # Crear directorio
            pair_dir = f'models/{self.pair_name.lower()}_simple'
            os.makedirs(pair_dir, exist_ok=True)
            
            # Guardar scaler
            with open(f'{pair_dir}/scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            
            print(f"✅ Datos preparados: X{X_scaled.shape}, y{y.shape}")
            
            return X_scaled, y, label_bias, pair_dir
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None, None, None, None
    
    def create_simple_model(self):
        """Crear modelo simple que funciona"""
        try:
            print(f"\n📋 CREANDO MODELO SIMPLE {self.pair_name}")
            print("-" * 40)
            
            n_features = len(self.config['features'])
            
            # Arquitectura simple y efectiva
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(n_features,)),
                
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                
                tf.keras.layers.Dense(8, activation='relu'),
                tf.keras.layers.Dropout(0.1),
                
                tf.keras.layers.Dense(3, activation='softmax')
            ])
            
            model.compile(
                optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print(f"✅ Modelo simple: {model.count_params()} parámetros")
            
            return model
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def train_with_class_weights(self, model, X, y):
        """Entrenar con class weights"""
        try:
            print(f"\n📋 ENTRENANDO {self.pair_name} CON CLASS WEIGHTS")
            print("-" * 40)
            
            # Calcular class weights
            unique_classes = np.unique(y)
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=unique_classes,
                y=y
            )
            class_weight_dict = dict(zip(unique_classes, class_weights))
            
            print("⚖️ Class weights:")
            for cls, weight in class_weight_dict.items():
                print(f"   - {self.class_names[cls]}: {weight:.3f}")
            
            # Split
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            print(f"📊 Train: {len(X_train)}, Val: {len(X_val)}")
            
            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True,
                    verbose=1
                )
            ]
            
            # Entrenar
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=16,
                class_weight=class_weight_dict,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluar
            predictions = model.predict(X_val, verbose=0)
            pred_classes = np.argmax(predictions, axis=1)
            
            # Distribución
            pred_counts = Counter(pred_classes)
            total_preds = len(pred_classes)
            
            print(f"\n📊 PREDICCIONES {self.pair_name}:")
            for i, name in enumerate(self.class_names):
                count = pred_counts[i]
                pct = count / total_preds * 100
                print(f"   - {name}: {count} ({pct:.1f}%)")
            
            # Métricas
            percentages = [pred_counts[i]/total_preds for i in range(3)]
            final_bias = (max(percentages) - min(percentages)) * 10
            avg_confidence = np.mean(np.max(predictions, axis=1))
            
            print(f"\n🎯 MÉTRICAS {self.pair_name}:")
            print(f"   📏 Bias Score: {final_bias:.1f}/10")
            print(f"   🎯 Confianza: {avg_confidence:.3f}")
            
            # Accuracy por clase
            print(f"\n🔍 ACCURACY POR CLASE:")
            for i, name in enumerate(self.class_names):
                mask = y_val == i
                if np.sum(mask) > 0:
                    class_pred = pred_classes[mask]
                    class_acc = np.mean(class_pred == i)
                    print(f"   - {name}: {class_acc:.3f}")
            
            return final_bias, pred_counts, avg_confidence
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None, None, None
    
    def save_model(self, model, pair_dir, bias_score, pred_counts, avg_confidence):
        """Guardar modelo"""
        try:
            print(f"\n📋 GUARDANDO MODELO {self.pair_name}")
            print("-" * 40)
            
            model_path = f'{pair_dir}/{self.pair_name.lower()}_simple_model.h5'
            model.save(model_path)
            
            metadata = {
                'pair': self.pair_symbol,
                'pair_name': self.pair_name,
                'model_type': f'{self.pair_name} Simple Model',
                'features': self.config['features'],
                'threshold': self.config['threshold'],
                'prediction_periods': self.config['prediction_periods'],
                'bias_score': float(bias_score),
                'avg_confidence': float(avg_confidence),
                'final_distribution': {
                    'SELL': float(pred_counts[0]/sum(pred_counts.values())),
                    'HOLD': float(pred_counts[1]/sum(pred_counts.values())),
                    'BUY': float(pred_counts[2]/sum(pred_counts.values()))
                },
                'training_date': datetime.now().isoformat(),
                'model_params': model.count_params()
            }
            
            with open(f'{pair_dir}/metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Modelo guardado: {model_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def train_complete(self):
        """Entrenamiento completo"""
        print(f"✅ ENTRENAMIENTO COMPLETO {self.pair_name}")
        print("="*60)
        
        # 1. Datos y features
        df = self.get_data_and_features()
        if df is None:
            return False
        
        # 2. Labels y preparación
        X, y, label_bias, pair_dir = self.create_labels_and_prepare(df)
        if X is None:
            return False
        
        # 3. Modelo
        model = self.create_simple_model()
        if model is None:
            return False
        
        # 4. Entrenar
        final_bias, pred_counts, avg_confidence = self.train_with_class_weights(model, X, y)
        if final_bias is None:
            return False
        
        # 5. Guardar
        if not self.save_model(model, pair_dir, final_bias, pred_counts, avg_confidence):
            return False
        
        # Resultado
        print(f"\n🏆 RESULTADO {self.pair_name}:")
        print(f"   📏 Bias Score: {final_bias:.1f}/10")
        print(f"   🎯 Confianza: {avg_confidence:.3f}")
        
        if final_bias < 6.0 and avg_confidence > 0.4:
            print(f"🎉 ¡MODELO {self.pair_name} EXITOSO!")
            return True
        else:
            print(f"⚠️ Modelo {self.pair_name} mejorable pero funcional")
            return True


def train_pair(pair_symbol):
    """Entrenar modelo para un par específico"""
    trainer = SimplePairTrainer(pair_symbol)
    return trainer.train_complete()


if __name__ == "__main__":
    # Entrenar ETH primero
    print("🚀 INICIANDO ENTRENAMIENTO DE MODELOS SIMPLES")
    print("="*60)
    
    success_eth = train_pair('ETHUSDT')
    
    if success_eth:
        print("\n✅ ¡ETH SIMPLE COMPLETADO!")
        print("📝 Listo para entrenar otros pares...")
    else:
        print("\n❌ Error en ETH simple") 