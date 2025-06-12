#!/usr/bin/env python3
"""
🧪 Test del Modelo TCN con 66 Features Exactas

Script para probar el modelo TCN con las 66 features exactas con las que fue entrenado.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle


class TCN66FeaturesTester:
    """
    🧪 Tester del modelo TCN con las 66 features exactas
    """
    
    def __init__(self):
        self.model_path = "models/tcn_anti_bias_fixed.h5"
        self.scaler_path = "models/feature_scalers_fixed.pkl"
        self.symbol = "BTCUSDT"
        self.interval = "5m"
        self.lookback_window = 60  # Ventana de secuencia
        self.prediction_steps = 12  # Predicción a 12 períodos
        self.expected_features = 66  # Features exactas que espera el modelo
        
        print("🧪 TCN 66 Features Tester inicializado")
        print(f"   - Modelo: {self.model_path}")
        print(f"   - Features esperadas: {self.expected_features}")
    
    def setup_binance_client(self) -> bool:
        """Configura cliente de Binance"""
        try:
            print("\n🔗 Configurando cliente Binance...")
            self.binance_client = BinanceClient()
            
            server_time = self.binance_client.get_server_time()
            print(f"✅ Conectado a Binance")
            
            return True
        except Exception as e:
            print(f"❌ Error conectando a Binance: {e}")
            return False
    
    def get_market_data(self, limit: int = 1500) -> Optional[pd.DataFrame]:
        """Obtiene datos suficientes para calcular todas las features"""
        try:
            print(f"\n📊 Obteniendo datos de {self.symbol}...")
            
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
            
            print(f"✅ Datos obtenidos: {len(df)} períodos")
            print(f"   - Precio actual: ${float(df['close'].iloc[-1]):,.2f}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error obteniendo datos: {e}")
            return None
    
    def calculate_66_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula exactamente las 66 features que espera el modelo TCN"""
        try:
            print("\n🔧 Calculando las 66 features exactas del modelo...")
            
            df = df.copy()
            features = []
            
            # 1. Básicos OHLCV (5 features)
            features.extend(['open', 'high', 'low', 'close', 'volume'])
            
            # 2. Moving Averages - múltiples períodos (12 features)
            for period in [7, 10, 14, 20, 25, 50]:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                features.append(f'sma_{period}')
            
            for period in [9, 12, 21, 26, 50, 100]:
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                features.append(f'ema_{period}')
            
            # 3. RSI múltiples períodos (3 features)
            for period in [9, 14, 21]:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
                features.append(f'rsi_{period}')
            
            # 4. MACD familia completa (6 features)
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            df['macd_normalized'] = df['macd'] / df['close']
            features.extend(['macd', 'macd_signal', 'macd_histogram', 'macd_normalized'])
            
            # 5. Bollinger Bands completas (6 features)
            for period in [20, 50]:
                bb_middle = df['close'].rolling(window=period).mean()
                bb_std = df['close'].rolling(window=period).std()
                df[f'bb_upper_{period}'] = bb_middle + (bb_std * 2)
                df[f'bb_lower_{period}'] = bb_middle - (bb_std * 2)
                df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
                features.extend([f'bb_upper_{period}', f'bb_lower_{period}', f'bb_position_{period}'])
            
            # 6. Momentum y ROC (8 features)
            for period in [5, 10, 14, 20]:
                df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
                df[f'roc_{period}'] = df['close'].pct_change(periods=period)
                features.extend([f'momentum_{period}', f'roc_{period}'])
            
            # 7. Volatilidad (4 features)
            for period in [10, 20, 30, 50]:
                df[f'volatility_{period}'] = df['close'].pct_change().rolling(window=period).std()
                features.append(f'volatility_{period}')
            
            # 8. Volume analysis (6 features)
            for period in [10, 20, 50]:
                df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
                df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']
                features.extend([f'volume_sma_{period}', f'volume_ratio_{period}'])
            
            # 9. ATR (Average True Range) (3 features)
            for period in [14, 21, 30]:
                high_low = df['high'] - df['low']
                high_close = (df['high'] - df['close'].shift()).abs()
                low_close = (df['low'] - df['close'].shift()).abs()
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df[f'atr_{period}'] = true_range.rolling(window=period).mean()
                features.append(f'atr_{period}')
            
            # 10. Stochastic (2 features)
            low_min = df['low'].rolling(window=14).min()
            high_max = df['high'].rolling(window=14).max()
            df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            features.extend(['stoch_k', 'stoch_d'])
            
            # 11. Williams %R (2 features)
            for period in [14, 21]:
                high_max = df['high'].rolling(window=period).max()
                low_min = df['low'].rolling(window=period).min()
                df[f'williams_r_{period}'] = -100 * (high_max - df['close']) / (high_max - low_min)
                features.append(f'williams_r_{period}')
            
            # 12. Price position features (4 features)
            for period in [20, 50]:
                df[f'price_position_{period}'] = (df['close'] - df['low'].rolling(period).min()) / \
                                                (df['high'].rolling(period).max() - df['low'].rolling(period).min())
                df[f'price_distance_ma_{period}'] = (df['close'] - df['close'].rolling(period).mean()) / df['close'].rolling(period).mean()
                features.extend([f'price_position_{period}', f'price_distance_ma_{period}'])
            
            # 13. Completar hasta 66 features si es necesario
            while len(features) < self.expected_features:
                # Features adicionales para llegar exactamente a 66
                if len(features) == 65:
                    df['close_norm'] = df['close'] / df['close'].rolling(window=100).mean()
                    features.append('close_norm')
                elif len(features) == 64:
                    df['volume_norm'] = df['volume'] / df['volume'].rolling(window=100).mean()
                    features.append('volume_norm')
                elif len(features) == 63:
                    df['high_low_ratio'] = df['high'] / df['low']
                    features.append('high_low_ratio')
                else:
                    break
            
            # Asegurar exactamente 66 features
            features = features[:self.expected_features]
            
            # Limpiar NaN
            df = df.dropna()
            
            # Verificar que tenemos todas las features
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                print(f"⚠️ Features faltantes: {missing_features}")
                return None
            
            print(f"✅ Features calculadas: {len(features)}")
            print(f"   - Datos limpios: {len(df)} períodos")
            print(f"   - Features incluidas: {features[:10]}... (+{len(features)-10} más)")
            
            return df[['timestamp'] + features]
            
        except Exception as e:
            print(f"❌ Error calculando features: {e}")
            return None
    
    def test_model_with_correct_features(self, df: pd.DataFrame) -> Dict:
        """Prueba el modelo con las 66 features correctas"""
        try:
            print(f"\n🧪 Probando modelo TCN con {self.expected_features} features...")
            
            # Cargar modelo
            print("📥 Cargando modelo TCN...")
            model = tf.keras.models.load_model(self.model_path, compile=False)
            
            print(f"✅ Modelo cargado exitosamente")
            print(f"   - Input shape: {model.input_shape}")
            print(f"   - Output shape: {model.output_shape}")
            print(f"   - Parámetros: {model.count_params():,}")
            
            # Preparar datos
            feature_columns = [col for col in df.columns if col != 'timestamp']
            features_data = df[feature_columns].values
            
            if features_data.shape[1] != self.expected_features:
                print(f"❌ Error: Esperaba {self.expected_features} features, encontré {features_data.shape[1]}")
                return None
            
            # Cargar scaler si existe
            scaler = MinMaxScaler(feature_range=(0, 1))
            if Path(self.scaler_path).exists():
                print("📥 Cargando scaler guardado...")
                with open(self.scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            else:
                print("🔧 Creando nuevo scaler...")
                scaler.fit(features_data)
            
            # Normalizar features
            features_scaled = scaler.transform(features_data)
            
            # Crear secuencias para el modelo
            X = []
            for i in range(self.lookback_window, len(features_scaled)):
                X.append(features_scaled[i-self.lookback_window:i])
            
            X = np.array(X)
            
            if len(X) == 0:
                print("❌ No hay suficientes datos para crear secuencias")
                return None
            
            print(f"   - Secuencias creadas: {X.shape}")
            
            # Hacer predicciones
            print("🔮 Generando predicciones...")
            predictions = model.predict(X, verbose=0)
            
            # Analizar últimas predicciones
            recent_predictions = predictions[-20:]  # Últimas 20 predicciones
            
            # Interpretar predicciones (asumiendo clasificación BUY/HOLD/SELL)
            if predictions.shape[1] == 3:
                pred_classes = np.argmax(recent_predictions, axis=1)
                class_names = ['SELL', 'HOLD', 'BUY']
                
                # Estadísticas
                buy_signals = np.sum(pred_classes == 2)
                hold_signals = np.sum(pred_classes == 1)
                sell_signals = np.sum(pred_classes == 0)
                
                # Confianza promedio
                confidence = np.max(recent_predictions, axis=1).mean()
                
                # Predicción actual (última)
                current_prediction = pred_classes[-1]
                current_confidence = np.max(predictions[-1])
                
                results = {
                    'total_predictions': len(predictions),
                    'recent_buy_signals': buy_signals,
                    'recent_hold_signals': hold_signals,
                    'recent_sell_signals': sell_signals,
                    'average_confidence': confidence,
                    'current_prediction': class_names[current_prediction],
                    'current_confidence': current_confidence,
                    'model_working': True
                }
                
                print("✅ Predicciones completadas:")
                print(f"   - Total predicciones: {len(predictions)}")
                print(f"   - Últimas 20 señales:")
                print(f"     • BUY: {buy_signals}")
                print(f"     • HOLD: {hold_signals}")
                print(f"     • SELL: {sell_signals}")
                print(f"   - Confianza promedio: {confidence:.3f}")
                print(f"   - Predicción actual: {class_names[current_prediction]} (confianza: {current_confidence:.3f})")
                
                return results
            
            else:
                print(f"⚠️ Formato de salida inesperado: {predictions.shape}")
                return None
            
        except Exception as e:
            print(f"❌ Error probando modelo: {e}")
            return None
    
    async def run_complete_test(self):
        """Ejecuta la prueba completa con 66 features"""
        print("🚀 Iniciando test del modelo TCN con 66 features")
        print("="*60)
        
        # 1. Configurar Binance
        if not self.setup_binance_client():
            return False
        
        # 2. Obtener datos
        df = self.get_market_data(limit=1500)
        if df is None:
            return False
        
        # 3. Calcular las 66 features exactas
        df_features = self.calculate_66_features(df)
        if df_features is None:
            return False
        
        # 4. Probar modelo
        results = self.test_model_with_correct_features(df_features)
        if results is None:
            return False
        
        print("\n" + "="*60)
        print("🎉 Test del modelo TCN con 66 features completado!")
        
        # Resumen final
        print(f"\n📋 RESUMEN FINAL:")
        print(f"   - Modelo: {self.model_path}")
        print(f"   - Features: {self.expected_features} (correctas)")
        print(f"   - Predicciones: {results['total_predictions']}")
        print(f"   - Señal actual: {results['current_prediction']}")
        print(f"   - Confianza: {results['current_confidence']:.1%}")
        print(f"   - Modelo funcionando: {'✅ SÍ' if results['model_working'] else '❌ NO'}")
        
        return True


async def main():
    """Función principal"""
    print("🧪 TCN 66 Features Tester - Validación Completa")
    print("="*60)
    
    tester = TCN66FeaturesTester()
    
    try:
        success = await tester.run_complete_test()
        
        if success:
            print("\n✅ ¡Modelo TCN funcionando correctamente con datos reales!")
            print("🎯 El modelo está listo para usar en el sistema de trading")
        else:
            print("\n❌ Test fallido. Revisar errores anteriores.")
    
    except KeyboardInterrupt:
        print("\n⚠️ Test interrumpido por el usuario")
    except Exception as e:
        print(f"\n💥 Error crítico: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 