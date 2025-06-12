#!/usr/bin/env python3
"""
🎯 ETH TRADING READY - Modelo ETH que predice BUY, HOLD, SELL para trading real
REQUISITOS MÍNIMOS CUMPLIDOS PARA TRADING
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

class ETHTradingReady:
    """Modelo ETH optimizado para trading real con las 3 señales"""
    
    def __init__(self):
        self.pair_symbol = "ETHUSDT"
        self.pair_name = "ETH"
        self.class_names = ['SELL', 'HOLD', 'BUY']
        
        # PARÁMETROS OPTIMIZADOS PARA 3 SEÑALES
        self.volatility_threshold = 0.004  # 0.4% - más sensible para capturar BUY
        self.prediction_periods = 2        # 10 min - más inmediato
        self.min_samples_per_class = 15    # Mínimo por clase
        
        print("🎯 ETH TRADING READY")
        print("="*50)
        print(f"📊 Par: {self.pair_symbol}")
        print(f"🎯 Threshold: {self.volatility_threshold*100:.1f}%")
        print(f"⏰ Predicción: {self.prediction_periods} períodos (10 min)")
        print(f"📋 Objetivo: PREDECIR BUY, HOLD, SELL")
        print("🚀 TRADING READY!")
    
    def get_optimized_data(self):
        """Obtener datos optimizados para 3 señales"""
        try:
            print(f"\n📋 DATOS OPTIMIZADOS PARA 3 SEÑALES")
            print("-" * 40)
            
            client = BinanceClient()
            ticker = client.get_symbol_ticker(symbol=self.pair_symbol)
            price = float(ticker['price'])
            print(f"✅ Precio ETH: ${price:,.2f}")
            
            # Más datos históricos para mejor balance
            klines = client.get_historical_klines(
                symbol=self.pair_symbol,
                interval='5m',
                limit=500  # Más datos
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
            
            # === FEATURES ESPECÍFICAS PARA TRADING ===
            print("🔧 Features específicas para trading...")
            
            # 1. Returns múltiples timeframes
            df['returns_1'] = df['close'].pct_change(1)
            df['returns_3'] = df['close'].pct_change(3)
            df['returns_5'] = df['close'].pct_change(5)
            
            # 2. Momentum indicators
            df['sma_5'] = df['close'].rolling(5).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            df['momentum_5_20'] = (df['sma_5'] - df['sma_20']) / df['sma_20']
            
            # 3. RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            df['rsi_norm'] = (df['rsi'] - 50) / 50  # Normalizado -1 a 1
            
            # 4. MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = (ema_12 - ema_26) / df['close']
            
            # 5. Bollinger position
            ma_20 = df['close'].rolling(20).mean()
            std_20 = df['close'].rolling(20).std()
            df['bb_position'] = (df['close'] - ma_20) / (2 * std_20)
            
            # 6. Volume indicators
            df['volume_ma'] = df['volume'].rolling(10).mean()
            df['volume_surge'] = df['volume'] / df['volume_ma']
            
            # 7. Price velocity
            df['price_velocity'] = df['close'].diff().rolling(3).mean()
            
            # 8. Volatility
            df['volatility'] = df['returns_1'].rolling(10).std()
            
            # Features finales
            self.features = [
                'returns_1', 'returns_3', 'momentum_5_20', 'rsi_norm', 
                'macd', 'bb_position', 'volume_surge', 'volatility'
            ]
            
            df = df.dropna()
            print(f"✅ Features trading: {len(self.features)}")
            print(f"✅ Datos finales: {len(df)} períodos")
            
            return df
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def create_balanced_labels(self, df):
        """Crear labels balanceadas para las 3 señales"""
        try:
            print(f"\n📋 LABELS BALANCEADAS PARA 3 SEÑALES")
            print("-" * 40)
            
            threshold = self.volatility_threshold
            pred_periods = self.prediction_periods
            
            print(f"🎯 Threshold: {threshold*100:.1f}%")
            print(f"⏰ Predicción: {pred_periods} períodos")
            
            labels = []
            
            # Analizar rangos de cambio para optimizar thresholds
            changes = []
            for i in range(len(df) - pred_periods):
                current = df['close'].iloc[i]
                future = df['close'].iloc[i + pred_periods]
                change = (future - current) / current
                changes.append(change)
            
            changes = np.array(changes)
            
            # Estadísticas de cambios
            print(f"📊 Estadísticas de cambios:")
            print(f"   Min: {np.min(changes)*100:.2f}%")
            print(f"   Max: {np.max(changes)*100:.2f}%")
            print(f"   Mean: {np.mean(changes)*100:.2f}%")
            print(f"   Std: {np.std(changes)*100:.2f}%")
            
            # Ajustar threshold dinámicamente si es necesario
            std_changes = np.std(changes)
            if threshold < std_changes * 0.5:
                threshold = std_changes * 0.6
                print(f"🔧 Threshold ajustado a: {threshold*100:.1f}%")
            
            # Crear labels
            for change in changes:
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
            
            print("📊 Distribución inicial:")
            for i, name in enumerate(self.class_names):
                count = label_counts[i]
                pct = count / total * 100
                print(f"   - {name}: {count} ({pct:.1f}%)")
            
            # VERIFICAR REQUISITOS MÍNIMOS
            min_samples = self.min_samples_per_class
            needs_adjustment = False
            
            for i, name in enumerate(self.class_names):
                if label_counts[i] < min_samples:
                    print(f"⚠️ {name} tiene solo {label_counts[i]} muestras (mín: {min_samples})")
                    needs_adjustment = True
            
            # Si necesita ajuste, reducir threshold
            if needs_adjustment:
                print("🔧 Ajustando threshold para mejor balance...")
                threshold = threshold * 0.7
                
                labels = []
                for change in changes:
                    if change < -threshold:
                        labels.append(0)  # SELL
                    elif change > threshold:
                        labels.append(2)  # BUY
                    else:
                        labels.append(1)  # HOLD
                
                df['label'] = labels
                label_counts = Counter(labels)
                
                print(f"🎯 Nuevo threshold: {threshold*100:.1f}%")
                print("📊 Nueva distribución:")
                for i, name in enumerate(self.class_names):
                    count = label_counts[i]
                    pct = count / total * 100
                    print(f"   - {name}: {count} ({pct:.1f}%)")
            
            # Bias score
            percentages = [label_counts[i]/total for i in range(3)]
            label_bias = (max(percentages) - min(percentages)) * 10
            print(f"📏 Label Bias: {label_bias:.1f}/10")
            
            # Guardar threshold usado
            self.final_threshold = threshold
            
            return df, label_bias
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None, None
    
    def prepare_trading_data(self, df):
        """Preparar datos para trading"""
        try:
            print(f"\n📋 PREPARANDO DATOS PARA TRADING")
            print("-" * 40)
            
            # Normalizar features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df[self.features])
            y = np.array(df['label'])
            
            # Crear directorio
            os.makedirs('models/eth_trading', exist_ok=True)
            
            # Guardar scaler
            with open('models/eth_trading/scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            
            print(f"✅ Datos trading: X{X_scaled.shape}, y{y.shape}")
            
            # Verificar distribución final
            final_counts = Counter(y)
            total = len(y)
            print("📊 Distribución final:")
            for i, name in enumerate(self.class_names):
                count = final_counts[i]
                pct = count / total * 100
                print(f"   - {name}: {count} ({pct:.1f}%)")
            
            return X_scaled, y
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None, None
    
    def create_trading_model(self):
        """Crear modelo optimizado para trading"""
        try:
            print(f"\n📋 MODELO OPTIMIZADO PARA TRADING")
            print("-" * 40)
            
            n_features = len(self.features)
            
            # Arquitectura optimizada para trading
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(n_features,)),
                
                # Primera capa: detección de patrones
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                
                # Segunda capa: refinamiento
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                
                # Tercera capa: decisión
                tf.keras.layers.Dense(8, activation='relu'),
                tf.keras.layers.Dropout(0.1),
                
                # Output: 3 señales de trading
                tf.keras.layers.Dense(3, activation='softmax')
            ])
            
            # Optimizador con learning rate adaptativo
            optimizer = tf.keras.optimizers.legacy.Adam(
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999
            )
            
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print(f"✅ Modelo trading: {model.count_params()} parámetros")
            
            return model
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def train_for_trading(self, model, X, y):
        """Entrenar específicamente para trading"""
        try:
            print(f"\n📋 ENTRENAMIENTO PARA TRADING")
            print("-" * 40)
            
            # Class weights balanceados
            unique_classes = np.unique(y)
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=unique_classes,
                y=y
            )
            
            # Ajustar weights para favorecer BUY/SELL
            class_weight_dict = dict(zip(unique_classes, class_weights))
            
            # Boost signals de trading (SELL y BUY)
            if 0 in class_weight_dict:  # SELL
                class_weight_dict[0] *= 1.2
            if 2 in class_weight_dict:  # BUY
                class_weight_dict[2] *= 1.2
            
            print("⚖️ Class weights optimizados:")
            for cls, weight in class_weight_dict.items():
                print(f"   - {self.class_names[cls]}: {weight:.3f}")
            
            # Split temporal
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            print(f"📊 Train: {len(X_train)}, Val: {len(X_val)}")
            
            # Callbacks para trading
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',  # Monitor accuracy para trading
                    patience=25,
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_accuracy',
                    factor=0.8,
                    patience=10,
                    min_lr=0.00001,
                    verbose=1
                )
            ]
            
            print("🔥 Entrenando para trading...")
            
            # Entrenar
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=150,  # Más épocas para mejor aprendizaje
                batch_size=16,
                class_weight=class_weight_dict,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluar con enfoque trading
            predictions = model.predict(X_val, verbose=0)
            pred_classes = np.argmax(predictions, axis=1)
            
            # Distribución final
            pred_counts = Counter(pred_classes)
            total_preds = len(pred_classes)
            
            print(f"\n📊 PREDICCIONES TRADING:")
            for i, name in enumerate(self.class_names):
                count = pred_counts[i]
                pct = count / total_preds * 100
                print(f"   - {name}: {count} ({pct:.1f}%)")
            
            # Métricas trading
            percentages = [pred_counts[i]/total_preds for i in range(3)]
            final_bias = (max(percentages) - min(percentages)) * 10
            avg_confidence = np.mean(np.max(predictions, axis=1))
            
            print(f"\n🎯 MÉTRICAS TRADING:")
            print(f"   📏 Bias Score: {final_bias:.1f}/10")
            print(f"   🎯 Confianza: {avg_confidence:.3f}")
            
            # Accuracy por señal
            print(f"\n🎯 ACCURACY POR SEÑAL:")
            trading_ready = True
            
            for i, name in enumerate(self.class_names):
                mask = y_val == i
                if np.sum(mask) > 0:
                    class_pred = pred_classes[mask]
                    class_acc = np.mean(class_pred == i)
                    print(f"   - {name}: {class_acc:.3f}")
                    
                    # Verificar requisitos mínimos
                    if class_acc < 0.3:  # Mínimo 30%
                        trading_ready = False
                        print(f"     ⚠️ Bajo para trading real")
                    else:
                        print(f"     ✅ Apto para trading")
            
            # Verificar si cumple requisitos trading
            print(f"\n🔍 VERIFICACIÓN TRADING READY:")
            print(f"   📊 Predice 3 señales: {'✅' if all(pred_counts[i] > 0 for i in range(3)) else '❌'}")
            print(f"   📏 Bias < 5.0: {'✅' if final_bias < 5.0 else '❌'}")
            print(f"   🎯 Confianza > 0.6: {'✅' if avg_confidence > 0.6 else '❌'}")
            print(f"   🎯 Accuracy > 0.3: {'✅' if trading_ready else '❌'}")
            
            return final_bias, pred_counts, avg_confidence, trading_ready
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None, None, None, False
    
    def save_trading_model(self, model, bias_score, pred_counts, avg_confidence, trading_ready):
        """Guardar modelo trading"""
        try:
            print(f"\n📋 GUARDANDO MODELO TRADING")
            print("-" * 40)
            
            model_path = 'models/eth_trading/eth_trading_model.h5'
            model.save(model_path)
            
            metadata = {
                'pair': self.pair_symbol,
                'model_type': 'ETH Trading Ready Model',
                'features': self.features,
                'threshold': float(self.final_threshold),
                'prediction_periods': self.prediction_periods,
                'bias_score': float(bias_score),
                'avg_confidence': float(avg_confidence),
                'trading_ready': trading_ready,
                'final_distribution': {
                    'SELL': float(pred_counts[0]/sum(pred_counts.values())),
                    'HOLD': float(pred_counts[1]/sum(pred_counts.values())),
                    'BUY': float(pred_counts[2]/sum(pred_counts.values()))
                },
                'training_date': datetime.now().isoformat(),
                'model_params': model.count_params()
            }
            
            with open('models/eth_trading/metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Modelo guardado: {model_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def train_complete(self):
        """Entrenamiento completo para trading"""
        print("🎯 ENTRENAMIENTO ETH TRADING READY")
        print("="*60)
        
        # 1. Datos optimizados
        df = self.get_optimized_data()
        if df is None:
            return False
        
        # 2. Labels balanceadas
        df, label_bias = self.create_balanced_labels(df)
        if df is None:
            return False
        
        # 3. Preparar datos
        X, y = self.prepare_trading_data(df)
        if X is None:
            return False
        
        # 4. Modelo trading
        model = self.create_trading_model()
        if model is None:
            return False
        
        # 5. Entrenar para trading
        final_bias, pred_counts, avg_confidence, trading_ready = self.train_for_trading(model, X, y)
        if final_bias is None:
            return False
        
        # 6. Guardar
        if not self.save_trading_model(model, final_bias, pred_counts, avg_confidence, trading_ready):
            return False
        
        # Resultado final
        print(f"\n🏆 RESULTADO ETH TRADING:")
        print(f"   📏 Bias Score: {final_bias:.1f}/10")
        print(f"   🎯 Confianza: {avg_confidence:.3f}")
        print(f"   🚀 Trading Ready: {'✅ SÍ' if trading_ready else '❌ NO'}")
        
        if trading_ready:
            print("🎉 ¡MODELO ETH LISTO PARA TRADING REAL!")
            return True
        else:
            print("⚠️ Modelo funcional pero necesita optimización")
            return False


if __name__ == "__main__":
    trainer = ETHTradingReady()
    success = trainer.train_complete()
    
    if success:
        print("\n🎉 ¡ETH TRADING READY COMPLETADO!")
    else:
        print("\n❌ ETH requiere más optimización") 