#!/usr/bin/env python3
"""
🧪 TEST MODELO DEFINITIVO BTCUSDT
Prueba el modelo definitivo entrenado con predicciones en tiempo real
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
import talib
import warnings
import pickle
import os
from collections import Counter
warnings.filterwarnings('ignore')

class DefinitivoBTCTester:
    """🧪 Tester del modelo definitivo BTCUSDT"""

    def __init__(self):
        self.symbol = "BTCUSDT"
        self.lookback_window = 48
        self.model_path = "models/definitivo_btcusdt/best_model.h5"
        self.model = None
        self.scaler = None

        # Cargar modelo
        self.load_model()

    def load_model(self):
        """📥 Cargar modelo definitivo"""
        try:
            print(f"📥 Cargando modelo definitivo: {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"✅ Modelo cargado: {self.model.count_params():,} parámetros")

            # Crear scaler (necesitaremos reentrenarlo con datos actuales)
            self.scaler = RobustScaler()

        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            raise

    async def get_real_time_data(self, hours: int = 3) -> pd.DataFrame:
        """📊 Obtener datos en tiempo real para predicción"""

        print(f"📊 Obteniendo {hours} horas de datos en tiempo real para {self.symbol}...")

        base_url = "https://api.binance.com"
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(hours=hours)).timestamp() * 1000)

        async with aiohttp.ClientSession() as session:
            url = f"{base_url}/api/v3/klines"
            params = {
                'symbol': self.symbol,
                'interval': '1m',
                'startTime': start_time,
                'endTime': end_time,
                'limit': 1000
            }

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                else:
                    raise Exception(f"Error API: {response.status}")

        # Convertir a DataFrame
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        # Convertir tipos
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp').sort_index()

        print(f"✅ Obtenidos {len(df)} registros en tiempo real")
        return df

    def create_66_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """🔧 Crear 66 features técnicos (misma función del trainer)"""

        print("🔧 Creando 66 features técnicos...")

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        features = pd.DataFrame(index=df.index)

        try:
            # === MOMENTUM INDICATORS (15 features) ===
            features['rsi_14'] = talib.RSI(close, timeperiod=14)
            features['rsi_21'] = talib.RSI(close, timeperiod=21)
            features['rsi_7'] = talib.RSI(close, timeperiod=7)

            # MACD family
            macd, macd_signal, macd_hist = talib.MACD(close)
            features['macd'] = macd
            features['macd_signal'] = macd_signal
            features['macd_histogram'] = macd_hist

            # Stochastic
            slowk, slowd = talib.STOCH(high, low, close)
            features['stoch_k'] = slowk
            features['stoch_d'] = slowd

            # Williams %R
            features['williams_r'] = talib.WILLR(high, low, close)

            # Rate of Change
            features['roc_10'] = talib.ROC(close, timeperiod=10)
            features['roc_20'] = talib.ROC(close, timeperiod=20)

            # Momentum
            features['momentum_10'] = talib.MOM(close, timeperiod=10)
            features['momentum_20'] = talib.MOM(close, timeperiod=20)

            # CCI
            features['cci_14'] = talib.CCI(high, low, close, timeperiod=14)
            features['cci_20'] = talib.CCI(high, low, close, timeperiod=20)

            # === TREND INDICATORS (12 features) ===
            # Moving Averages
            features['sma_10'] = talib.SMA(close, timeperiod=10)
            features['sma_20'] = talib.SMA(close, timeperiod=20)
            features['sma_50'] = talib.SMA(close, timeperiod=50)
            features['ema_10'] = talib.EMA(close, timeperiod=10)
            features['ema_20'] = talib.EMA(close, timeperiod=20)
            features['ema_50'] = talib.EMA(close, timeperiod=50)

            # ADX family
            features['adx_14'] = talib.ADX(high, low, close, timeperiod=14)
            features['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
            features['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)

            # PSAR
            features['psar'] = talib.SAR(high, low)

            # Aroon
            aroon_down, aroon_up = talib.AROON(high, low, timeperiod=14)
            features['aroon_up'] = aroon_up
            features['aroon_down'] = aroon_down

            # === VOLATILITY INDICATORS (10 features) ===
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close)
            features['bb_upper'] = bb_upper
            features['bb_middle'] = bb_middle
            features['bb_lower'] = bb_lower
            features['bb_width'] = (bb_upper - bb_lower) / bb_middle
            features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)

            # ATR
            features['atr_14'] = talib.ATR(high, low, close, timeperiod=14)
            features['atr_20'] = talib.ATR(high, low, close, timeperiod=20)

            # True Range
            features['true_range'] = talib.TRANGE(high, low, close)

            # Normalized ATR
            features['natr_14'] = talib.NATR(high, low, close, timeperiod=14)
            features['natr_20'] = talib.NATR(high, low, close, timeperiod=20)

            # === VOLUME INDICATORS (8 features) ===
            features['ad'] = talib.AD(high, low, close, volume)
            features['adosc'] = talib.ADOSC(high, low, close, volume)
            features['obv'] = talib.OBV(close, volume)

            # Volume SMA
            features['volume_sma_10'] = talib.SMA(volume, timeperiod=10)
            features['volume_sma_20'] = talib.SMA(volume, timeperiod=20)

            # Volume ratios
            features['volume_ratio'] = volume / features['volume_sma_20']
            features['price_volume'] = close * volume
            features['vwap'] = (features['price_volume'].rolling(20).sum() /
                              features['volume'].rolling(20).sum())

            # === PRICE ACTION FEATURES (10 features) ===
            # Returns
            features['returns_1'] = close / np.roll(close, 1) - 1
            features['returns_5'] = close / np.roll(close, 5) - 1
            features['returns_10'] = close / np.roll(close, 10) - 1

            # High-Low ratios
            features['hl_ratio'] = (high - low) / close
            features['oc_ratio'] = (close - df['open'].values) / df['open'].values

            # Price position in range
            features['price_position'] = (close - low) / (high - low)

            # Gaps
            features['gap'] = (df['open'].values - np.roll(close, 1)) / np.roll(close, 1)

            # Rolling statistics
            features['close_std_10'] = pd.Series(close).rolling(10).std()
            features['close_mean_10'] = pd.Series(close).rolling(10).mean()
            features['close_zscore'] = (close - features['close_mean_10']) / features['close_std_10']

            # === CUSTOM FEATURES (11 features) ===
            # Trend strength
            features['trend_strength'] = (features['ema_10'] - features['ema_50']) / features['ema_50']

            # Volatility regime
            features['vol_regime'] = features['atr_14'] / features['atr_14'].rolling(50).mean()

            # Momentum divergence
            features['mom_divergence'] = features['rsi_14'] - features['rsi_14'].rolling(10).mean()

            # Volume momentum
            features['vol_momentum'] = (features['volume_sma_10'] - features['volume_sma_20']) / features['volume_sma_20']

            # Price acceleration
            features['price_acceleration'] = features['returns_1'] - features['returns_1'].shift(1)

            # Support/Resistance levels
            features['support_level'] = pd.Series(low).rolling(20).min()
            features['resistance_level'] = pd.Series(high).rolling(20).max()
            features['support_distance'] = (close - features['support_level']) / close
            features['resistance_distance'] = (features['resistance_level'] - close) / close

            # Market regime
            features['market_regime'] = np.where(
                features['ema_10'] > features['ema_20'], 1,
                np.where(features['ema_10'] < features['ema_20'], -1, 0)
            )

            # Fractal dimension (simplified)
            features['fractal_dimension'] = self._calculate_fractal_dimension(pd.Series(close))

            # Limpiar infinitos y NaN
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(method='ffill').fillna(0)

            print(f"✅ {len(features.columns)} features creados")
            return features

        except Exception as e:
            print(f"❌ Error creando features: {e}")
            raise

    def _calculate_fractal_dimension(self, series: pd.Series, window: int = 20) -> pd.Series:
        """📊 Calcular dimensión fractal simplificada"""
        def hurst_exponent(ts):
            try:
                if len(ts) < 10:
                    return 0.5
                lags = range(2, min(len(ts)//2, 20))
                tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                return max(0.1, min(0.9, poly[0] * 2.0))
            except:
                return 0.5

        return series.rolling(window).apply(hurst_exponent, raw=False)

    async def make_prediction(self) -> dict:
        """🎯 Hacer predicción en tiempo real"""

        print(f"\n🎯 PREDICCIÓN EN TIEMPO REAL - {self.symbol}")
        print("=" * 60)

        try:
            # 1. Obtener datos actuales
            df = await self.get_real_time_data(hours=3)
            current_price = df['close'].iloc[-1]

            # 2. Crear features
            features = self.create_66_features(df)

            # 3. Preparar datos para predicción
            feature_columns = [col for col in features.columns if features[col].dtype in ['float64', 'int64']]

            # Reentrenar scaler con datos actuales
            self.scaler.fit(features[feature_columns].fillna(0))
            features_scaled = self.scaler.transform(features[feature_columns].fillna(0))

            # 4. Crear secuencia para predicción
            if len(features_scaled) >= self.lookback_window:
                sequence = features_scaled[-self.lookback_window:].reshape(1, self.lookback_window, len(feature_columns))

                # 5. Hacer predicción
                prediction = self.model.predict(sequence, verbose=0)
                predicted_class = np.argmax(prediction[0])
                confidence = np.max(prediction[0])

                # 6. Interpretar resultado
                class_names = ['SELL', 'HOLD', 'BUY']
                action = class_names[predicted_class]

                # 7. Análisis de distribución
                sell_prob = prediction[0][0]
                hold_prob = prediction[0][1]
                buy_prob = prediction[0][2]

                result = {
                    'timestamp': datetime.now(),
                    'symbol': self.symbol,
                    'current_price': current_price,
                    'prediction': action,
                    'confidence': confidence,
                    'probabilities': {
                        'SELL': sell_prob,
                        'HOLD': hold_prob,
                        'BUY': buy_prob
                    },
                    'features_count': len(feature_columns)
                }

                # 8. Mostrar resultados
                print(f"💰 Precio actual: ${current_price:,.2f}")
                print(f"🎯 Predicción: {action}")
                print(f"🔥 Confianza: {confidence:.1%}")
                print(f"\n📊 Distribución de probabilidades:")
                print(f"   🔴 SELL: {sell_prob:.1%}")
                print(f"   ⚪ HOLD: {hold_prob:.1%}")
                print(f"   🟢 BUY:  {buy_prob:.1%}")
                print(f"\n🔧 Features utilizados: {len(feature_columns)}")

                return result

            else:
                print(f"❌ Datos insuficientes: {len(features_scaled)} < {self.lookback_window}")
                return None

        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            import traceback
            traceback.print_exc()
            return None

async def main():
    """🚀 Función principal"""

    print("🧪 TESTING MODELO DEFINITIVO BTCUSDT")
    print("=" * 70)

    try:
        # Crear tester
        tester = DefinitivoBTCTester()

        # Hacer predicción
        result = await tester.make_prediction()

        if result:
            print(f"\n✅ Predicción completada exitosamente")
            print(f"🕐 Timestamp: {result['timestamp']}")
        else:
            print(f"\n❌ No se pudo completar la predicción")

    except Exception as e:
        print(f"❌ Error general: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
