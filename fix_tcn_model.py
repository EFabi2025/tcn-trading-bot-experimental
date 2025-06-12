#!/usr/bin/env python3
"""
🔧 Fix y Test del Modelo TCN Real

Script para resolver problemas de compatibilidad del modelo TCN 
y probarlo con datos reales de Binance.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from binance.client import Client as BinanceClient
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle


class TCNModelFixer:
    """
    🔧 Clase para arreglar y probar el modelo TCN real
    """
    
    def __init__(self):
        self.model_path = "models/tcn_anti_bias_fixed.h5"
        self.scaler_path = "models/feature_scalers_fixed.pkl"
        self.symbol = "BTCUSDT"
        self.interval = "5m"
        self.lookback_window = 60
        self.prediction_steps = 12
        
        print("🔧 TCN Model Fixer inicializado")
        print(f"   - TensorFlow version: {tf.__version__}")
    
    def check_model_structure(self):
        """Inspecciona la estructura del modelo"""
        try:
            print(f"\n🔍 Inspeccionando modelo: {self.model_path}")
            
            # Intentar cargar con diferentes métodos
            print("   - Método 1: load_model directo...")
            try:
                model = tf.keras.models.load_model(self.model_path)
                print("   ✅ Modelo cargado exitosamente")
                print(f"      - Input shape: {model.input_shape}")
                print(f"      - Output shape: {model.output_shape}")
                print(f"      - Parámetros: {model.count_params():,}")
                return model
            except Exception as e:
                print(f"   ❌ Error: {e}")
            
            # Método 2: Cargar pesos solamente
            print("   - Método 2: Intentando cargar solo pesos...")
            try:
                # Crear modelo básico con arquitectura similar
                model = self.create_fallback_model()
                model.load_weights(self.model_path)
                print("   ✅ Pesos cargados en modelo base")
                return model
            except Exception as e:
                print(f"   ❌ Error: {e}")
            
            return None
            
        except Exception as e:
            print(f"❌ Error inspeccionando modelo: {e}")
            return None
    
    def create_fallback_model(self):
        """Crea un modelo TCN básico de respaldo"""
        try:
            print("   🏗️ Creando modelo TCN de respaldo...")
            
            # Arquitectura TCN simplificada
            inputs = tf.keras.Input(shape=(self.lookback_window, 15))  # 15 features
            
            # Capas TCN básicas
            x = tf.keras.layers.Conv1D(32, 3, padding='causal', activation='relu')(inputs)
            x = tf.keras.layers.Dropout(0.2)(x)
            
            x = tf.keras.layers.Conv1D(64, 3, padding='causal', activation='relu')(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            
            x = tf.keras.layers.Conv1D(32, 3, padding='causal', activation='relu')(x)
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            
            # Capas densas
            x = tf.keras.layers.Dense(50, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            outputs = tf.keras.layers.Dense(1)(x)
            
            model = tf.keras.Model(inputs, outputs)
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            print("   ✅ Modelo de respaldo creado")
            return model
            
        except Exception as e:
            print(f"   ❌ Error creando modelo de respaldo: {e}")
            return None
    
    def load_scalers(self):
        """Carga los scalers guardados"""
        try:
            if Path(self.scaler_path).exists():
                print(f"📥 Cargando scalers desde: {self.scaler_path}")
                with open(self.scaler_path, 'rb') as f:
                    scalers = pickle.load(f)
                print("✅ Scalers cargados exitosamente")
                return scalers
            else:
                print("⚠️ Scalers no encontrados, creando nuevos...")
                return None
        except Exception as e:
            print(f"❌ Error cargando scalers: {e}")
            return None
    
    def get_binance_data(self, limit=1000):
        """Obtiene datos reales de Binance"""
        try:
            print(f"\n📊 Obteniendo datos de {self.symbol}...")
            
            client = BinanceClient()
            klines = client.get_historical_klines(
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
            
            print(f"✅ Datos obtenidos: {len(df)} períodos")
            print(f"   - Precio actual: ${float(df['close'].iloc[-1]):,.2f}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error obteniendo datos: {e}")
            return None
    
    def calculate_features(self, df):
        """Calcula las 15 features del modelo original"""
        try:
            print("🔧 Calculando features originales...")
            
            df = df.copy()
            
            # Las 15 features exactas del modelo original
            df['sma_7'] = df['close'].rolling(window=7).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Momentum y volatilidad
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['roc'] = df['close'].pct_change(periods=10)
            df['volatility'] = df['close'].pct_change().rolling(window=20).std()
            
            # Features finales (15 exactas)
            features = [
                'sma_7', 'sma_20', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal',
                'macd_histogram', 'bb_position', 'volume_ratio', 'momentum_5',
                'roc', 'volatility', 'bb_upper', 'bb_lower'
            ]
            
            df = df.dropna()
            
            print(f"✅ Features calculadas: {len(features)}")
            print(f"   - Datos limpios: {len(df)} períodos")
            
            return df, features
            
        except Exception as e:
            print(f"❌ Error calculando features: {e}")
            return None, None
    
    def test_model_prediction(self, model, df, features):
        """Prueba el modelo con datos reales"""
        try:
            print("\n🧪 Probando modelo con datos reales...")
            
            # Preparar datos
            feature_data = df[features].values
            prices = df['close'].values
            
            # Normalizar (crear scaler si no existe)
            scaler = MinMaxScaler(feature_range=(0, 1))
            feature_data_scaled = scaler.fit_transform(feature_data)
            
            # Crear secuencias
            X, y = [], []
            for i in range(self.lookback_window, len(feature_data_scaled) - self.prediction_steps):
                X.append(feature_data_scaled[i-self.lookback_window:i])
                y.append(prices[i + self.prediction_steps])
            
            X = np.array(X)
            y = np.array(y)
            
            if len(X) == 0:
                print("❌ No hay suficientes datos para crear secuencias")
                return None
            
            # Dividir train/test
            split_idx = int(len(X) * 0.8)
            X_test = X[split_idx:]
            y_test = y[split_idx:]
            
            if len(X_test) == 0:
                print("❌ No hay datos de test")
                return None
            
            print(f"   - Datos de prueba: {len(X_test)} muestras")
            print(f"   - Shape de entrada: {X_test.shape}")
            
            # Hacer predicciones
            y_pred = model.predict(X_test, verbose=0)
            
            if y_pred.ndim > 1:
                y_pred = y_pred.flatten()
            
            # Métricas
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            corr = np.corrcoef(y_test, y_pred)[0,1]
            
            # Precisión direccional
            direction_acc = np.mean(
                np.sign(np.diff(y_test)) == np.sign(np.diff(y_pred))
            ) * 100
            
            # Predicción futura
            last_sequence = feature_data_scaled[-self.lookback_window:].reshape(1, self.lookback_window, -1)
            future_pred = model.predict(last_sequence, verbose=0)[0]
            if hasattr(future_pred, '__len__'):
                future_pred = future_pred[0]
            
            current_price = prices[-1]
            price_change_pct = ((future_pred - current_price) / current_price) * 100
            
            results = {
                'mae': mae,
                'rmse': rmse,
                'correlation': corr,
                'direction_accuracy': direction_acc,
                'current_price': current_price,
                'predicted_price': future_pred,
                'price_change_pct': price_change_pct,
                'test_samples': len(X_test)
            }
            
            print("✅ Evaluación completada:")
            print(f"   - MAE: ${mae:.2f}")
            print(f"   - RMSE: ${rmse:.2f}")
            print(f"   - Correlación: {corr:.3f}")
            print(f"   - Precisión direccional: {direction_acc:.1f}%")
            print(f"   - Predicción futura: {price_change_pct:+.2f}%")
            
            return results
            
        except Exception as e:
            print(f"❌ Error probando modelo: {e}")
            return None
    
    def run_full_test(self):
        """Ejecuta la prueba completa"""
        print("🚀 Iniciando test completo del modelo TCN real")
        print("="*60)
        
        # 1. Cargar modelo
        model = self.check_model_structure()
        if model is None:
            print("❌ No se pudo cargar el modelo")
            return False
        
        # 2. Obtener datos
        df = self.get_binance_data()
        if df is None:
            return False
        
        # 3. Calcular features
        df_features, features = self.calculate_features(df)
        if df_features is None:
            return False
        
        # 4. Probar modelo
        results = self.test_model_prediction(model, df_features, features)
        if results is None:
            return False
        
        print("\n" + "="*60)
        print("🎉 Test del modelo TCN completado!")
        
        # Resumen final
        print(f"\n📋 RESUMEN MODELO TCN REAL:")
        print(f"   - Modelo: {self.model_path}")
        print(f"   - Muestras test: {results['test_samples']}")
        print(f"   - Error promedio: ${results['mae']:.2f}")
        print(f"   - Correlación: {results['correlation']:.3f}")
        print(f"   - Precisión direccional: {results['direction_accuracy']:.1f}%")
        print(f"   - Precio actual: ${results['current_price']:,.2f}")
        print(f"   - Predicción (1h): {results['price_change_pct']:+.2f}%")
        
        # Interpretación
        print(f"\n🔍 INTERPRETACIÓN:")
        if results['correlation'] > 0.5:
            print("   ✅ El modelo TCN muestra buena correlación")
        elif results['correlation'] > 0.3:
            print("   ⚠️ El modelo TCN muestra correlación moderada")
        else:
            print("   ❌ El modelo TCN necesita mejoras")
        
        if results['direction_accuracy'] > 55:
            print("   ✅ Buena precisión en dirección de precios")
        else:
            print("   ⚠️ Precisión direccional mejorable")
        
        return True


def main():
    """Función principal"""
    print("🔧 TCN Model Fixer y Tester")
    print("="*60)
    
    fixer = TCNModelFixer()
    
    try:
        success = fixer.run_full_test()
        
        if success:
            print("\n✅ ¡Test del modelo TCN real completado exitosamente!")
            print("📊 El modelo está funcionando con datos reales de Binance")
        else:
            print("\n❌ El test falló. Revisa los errores anteriores.")
    
    except KeyboardInterrupt:
        print("\n⚠️ Test interrumpido por el usuario")
    except Exception as e:
        print(f"\n💥 Error crítico: {e}")


if __name__ == "__main__":
    main() 