#!/usr/bin/env python3
"""
🏆 ETH TCN FINAL - Modelo ETH con class weights para solucionar el bias definitivamente
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

class EthereumTCNFinal:
    """Modelo ETH TCN final con solución definitiva"""
    
    def __init__(self):
        self.pair_symbol = "ETHUSDT"
        self.lookback_window = 15  # Optimizado
        self.expected_features = 8   # Optimizado
        self.class_names = ['SELL', 'HOLD', 'BUY']
        self.volatility_threshold = 0.003  # 0.3% optimizado
        
        print("🏆 ETHEREUM TCN FINAL")
        print("="*50)
        print(f"📊 Par: {self.pair_symbol}")
        print(f"🎯 Threshold: {self.volatility_threshold*100:.1f}%")
        print(f"📈 Features: {self.expected_features}")
        print(f"⏰ Ventana: {self.lookback_window} períodos")
        print("🔧 Con CLASS WEIGHTS balanceadas")
    
    def prepare_data(self):
        """Preparar datos completos"""
        try:
            print("\n📋 PREPARANDO DATOS COMPLETOS")
            print("-" * 40)
            
            # Conectar
            client = BinanceClient()
            ticker = client.get_symbol_ticker(symbol=self.pair_symbol)
            price = float(ticker['price'])
            print(f"✅ Conectado - Precio ETH: ${price:,.2f}")
            
            # Datos históricos
            klines = client.get_historical_klines(
                symbol=self.pair_symbol,
                interval='5m',
                limit=300
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
            
            # === FEATURES OPTIMIZADAS (8 total) ===
            print("🔧 Creando features optimizadas...")
            
            # 1. Returns
            df['returns'] = df['close'].pct_change()
            
            # 2. SMA ratio
            df['sma_5'] = df['close'].rolling(5).mean()
            df['sma_ratio'] = df['close'] / df['sma_5']
            
            # 3. EMA ratio  
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_ratio'] = df['close'] / df['ema_12']
            
            # 4. RSI normalizado
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi_norm'] = (100 - (100 / (1 + rs))) / 100  # Normalizado 0-1
            
            # 5. MACD normalizado
            macd = df['ema_12'] - df['close'].ewm(span=26).mean()
            df['macd_norm'] = macd / df['close']
            
            # 6. Volatilidad
            df['volatility'] = df['returns'].rolling(8).std()
            
            # 7. Volume momentum
            df['volume_ma'] = df['volume'].rolling(10).mean()
            df['volume_momentum'] = df['volume'] / df['volume_ma']
            
            # 8. Price momentum
            df['price_momentum'] = df['close'] / df['close'].shift(5)
            
            # Lista de features
            features = [
                'returns', 'sma_ratio', 'ema_ratio', 'rsi_norm',
                'macd_norm', 'volatility', 'volume_momentum', 'price_momentum'
            ]
            
            # Limpiar datos
            df = df.dropna()
            
            # === LABELS BALANCEADAS ===
            print("🎯 Creando labels balanceadas...")
            
            labels = []
            for i in range(len(df) - 2):  # Predicción a 2 períodos
                current = df['close'].iloc[i]
                future = df['close'].iloc[i + 2]
                change = (future - current) / current
                
                if change < -self.volatility_threshold:
                    labels.append(0)  # SELL
                elif change > self.volatility_threshold:
                    labels.append(2)  # BUY
                else:
                    labels.append(1)  # HOLD
            
            df = df.iloc[:-2].copy()
            df['label'] = labels
            
            # Verificar distribución
            label_counts = Counter(labels)
            total = len(labels)
            
            print("📊 Distribución inicial:")
            for i, name in enumerate(self.class_names):
                count = label_counts[i]
                pct = count / total * 100
                print(f"   - {name}: {count} ({pct:.1f}%)")
            
            self.feature_list = features
            return df
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def create_balanced_data(self, df):
        """Crear datos balanceados con class weights"""
        try:
            print("\n📋 CREANDO DATOS BALANCEADOS")
            print("-" * 40)
            
            # Normalizar features
            scaler = StandardScaler()
            df_scaled = df.copy()
            df_scaled[self.feature_list] = scaler.fit_transform(df[self.feature_list])
            
            # Crear directorio
            os.makedirs('models/eth_final', exist_ok=True)
            
            # Guardar scaler
            with open('models/eth_final/scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            
            # Crear secuencias
            X = []
            y = []
            
            for i in range(self.lookback_window, len(df_scaled)):
                sequence = df_scaled[self.feature_list].iloc[i-self.lookback_window:i].values
                X.append(sequence)
                y.append(int(df_scaled['label'].iloc[i]))
            
            X = np.array(X)
            y = np.array(y)
            
            print(f"✅ Secuencias: X{X.shape}, y{y.shape}")
            
            # === CALCULAR CLASS WEIGHTS ===
            print("⚖️ Calculando class weights...")
            
            # Obtener todas las clases únicas
            unique_classes = np.unique(y)
            
            # Calcular class weights
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=unique_classes,
                y=y
            )
            
            # Crear diccionario de class weights
            class_weight_dict = dict(zip(unique_classes, class_weights))
            
            print(f"📊 Class weights calculados:")
            for cls, weight in class_weight_dict.items():
                print(f"   - {self.class_names[cls]}: {weight:.3f}")
            
            # Verificar distribución
            final_counts = Counter(y)
            total = len(y)
            print("\n📊 Distribución de entrenamiento:")
            for i, name in enumerate(self.class_names):
                count = final_counts[i]
                pct = count / total * 100
                weight = class_weight_dict.get(i, 1.0)
                print(f"   - {name}: {count} ({pct:.1f}%) | Weight: {weight:.3f}")
            
            return X, y, class_weight_dict
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None, None, None
    
    def create_final_model(self):
        """Crear modelo final optimizado"""
        try:
            print("\n📋 CREANDO MODELO FINAL")
            print("-" * 40)
            
            # Arquitectura final optimizada
            model = tf.keras.Sequential([
                # Input
                tf.keras.layers.Input(shape=(self.lookback_window, self.expected_features)),
                
                # TCN layers optimizadas
                tf.keras.layers.Conv1D(filters=16, kernel_size=2, padding='causal', activation='relu'),
                tf.keras.layers.Dropout(0.1),
                
                tf.keras.layers.Conv1D(filters=32, kernel_size=2, dilation_rate=2, padding='causal', activation='relu'),
                tf.keras.layers.Dropout(0.2),
                
                tf.keras.layers.Conv1D(filters=16, kernel_size=2, dilation_rate=4, padding='causal', activation='relu'),
                tf.keras.layers.Dropout(0.2),
                
                # Global pooling
                tf.keras.layers.GlobalAveragePooling1D(),
                
                # Dense layers balanceadas
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                
                tf.keras.layers.Dense(16, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                
                # Output balanceado
                tf.keras.layers.Dense(3, activation='softmax')
            ])
            
            # Compilar con optimizador suave
            model.compile(
                optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0005),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print(f"✅ Modelo final: {model.count_params():,} parámetros")
            
            return model
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def train_final_model(self, model, X, y, class_weights):
        """Entrenar modelo final con class weights"""
        try:
            print("\n📋 ENTRENAMIENTO FINAL CON CLASS WEIGHTS")
            print("-" * 40)
            
            # Split temporal
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            print(f"📊 Train: {len(X_train)}, Val: {len(X_val)}")
            
            # Callbacks conservadores
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.8,
                    patience=10,
                    min_lr=0.00001,
                    verbose=1
                )
            ]
            
            print("🔥 Entrenando con class weights...")
            
            # Entrenar con class weights
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=80,
                batch_size=8,  # Batch pequeño
                class_weight=class_weights,  # ← CLAVE PARA BALANCE
                callbacks=callbacks,
                verbose=1
            )
            
            print("✅ Entrenamiento completado")
            
            # Evaluar con detalle
            print("\n📊 EVALUACIÓN FINAL...")
            
            predictions = model.predict(X_val, verbose=0)
            pred_classes = np.argmax(predictions, axis=1)
            
            # Distribución final
            pred_counts = Counter(pred_classes)
            total_preds = len(pred_classes)
            
            print("📊 PREDICCIONES FINALES:")
            for i, name in enumerate(self.class_names):
                count = pred_counts[i]
                pct = count / total_preds * 100
                print(f"   - {name}: {count} ({pct:.1f}%)")
            
            # Bias score final
            percentages = [pred_counts[i]/total_preds for i in range(3)]
            final_bias = (max(percentages) - min(percentages)) * 10
            
            # Confianza promedio
            avg_confidence = np.mean(np.max(predictions, axis=1))
            
            print(f"\n🎯 MÉTRICAS FINALES:")
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
            
            # Mostrar predicciones ejemplo
            print(f"\n🔍 EJEMPLOS DE PREDICCIONES:")
            for i in range(min(8, len(predictions))):
                probs = predictions[i]
                pred_class = pred_classes[i]
                true_class = y_val[i]
                confidence = np.max(probs)
                
                pred_name = self.class_names[pred_class]
                true_name = self.class_names[true_class]
                correct = "✅" if pred_class == true_class else "❌"
                
                print(f"   {i}: {correct} Pred={pred_name} Real={true_name} | Conf={confidence:.3f}")
            
            return final_bias, pred_counts, avg_confidence, predictions
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None, None, None, None
    
    def save_final_model(self, model, bias_score, pred_counts, avg_confidence):
        """Guardar modelo final"""
        try:
            print("\n📋 GUARDANDO MODELO FINAL")
            print("-" * 40)
            
            model_path = 'models/eth_final/eth_tcn_final.h5'
            model.save(model_path)
            
            metadata = {
                'pair': self.pair_symbol,
                'model_type': 'ETH Final TCN with Class Weights',
                'features': len(self.feature_list),
                'lookback_window': self.lookback_window,
                'volatility_threshold': self.volatility_threshold,
                'bias_score': float(bias_score),
                'avg_confidence': float(avg_confidence),
                'uses_class_weights': True,
                'final_distribution': {
                    'SELL': float(pred_counts[0]/sum(pred_counts.values())),
                    'HOLD': float(pred_counts[1]/sum(pred_counts.values())),
                    'BUY': float(pred_counts[2]/sum(pred_counts.values()))
                },
                'training_date': datetime.now().isoformat(),
                'features_list': self.feature_list,
                'model_params': model.count_params()
            }
            
            with open('models/eth_final/metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Modelo guardado: {model_path}")
            print(f"✅ Metadata guardado: models/eth_final/metadata.json")
            
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def train_complete(self):
        """Entrenamiento completo del modelo final"""
        print("🏆 ENTRENAMIENTO COMPLETO ETH FINAL")
        print("="*60)
        
        # 1. Preparar datos
        df = self.prepare_data()
        if df is None:
            return False
        
        # 2. Crear datos balanceados
        X, y, class_weights = self.create_balanced_data(df)
        if X is None:
            return False
        
        # 3. Crear modelo
        model = self.create_final_model()
        if model is None:
            return False
        
        # 4. Entrenar
        final_bias, pred_counts, avg_confidence, predictions = self.train_final_model(model, X, y, class_weights)
        if final_bias is None:
            return False
        
        # 5. Guardar
        if not self.save_final_model(model, final_bias, pred_counts, avg_confidence):
            return False
        
        # Resultado definitivo
        print(f"\n🏆 RESULTADO DEFINITIVO ETH:")
        print(f"   📏 Bias Score: {final_bias:.1f}/10")
        print(f"   🎯 Confianza: {avg_confidence:.3f}")
        
        if final_bias < 4.0 and avg_confidence > 0.5:
            print("🎉 ¡MODELO ETH FINAL EXITOSO!")
            return True
        elif final_bias < 7.0:
            print("✅ Modelo ETH mejorado significativamente")
            return True
        else:
            print("⚠️ Modelo ETH requiere más ajustes")
            return False


if __name__ == "__main__":
    trainer = EthereumTCNFinal()
    success = trainer.train_complete()
    
    if success:
        print("\n🏆 ¡MODELO ETH FINAL COMPLETADO!")
        print("📝 Listo para crear modelo BTC...")
    else:
        print("\n❌ Error en modelo ETH final") 