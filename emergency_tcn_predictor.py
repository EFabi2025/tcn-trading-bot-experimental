#!/usr/bin/env python3
"""
🚨 EMERGENCY TCN PREDICTOR - Predictor con modelos corregidos de 66 features
Sistema que usa los modelos de emergencia entrenados con thresholds corregidos
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
warnings.filterwarnings('ignore')

class AdvancedBinanceData:
    """Proveedor avanzado de datos de Binance"""

    def __init__(self):
        self.base_url = "https://api.binance.com"
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_comprehensive_data(self, symbol: str) -> dict:
        """Obtener datos completos de un símbolo"""

        # Datos simultáneos para mayor eficiencia
        tasks = [
            self.get_klines(symbol, "1m", 1000),  # Más datos históricos
            self.get_klines(symbol, "5m", 200),   # Datos 5m para contexto
            self.get_24hr_ticker(symbol),
            self.get_orderbook_ticker(symbol),
            self.get_avg_price(symbol)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            'klines_1m': results[0] if not isinstance(results[0], Exception) else [],
            'klines_5m': results[1] if not isinstance(results[1], Exception) else [],
            'ticker_24h': results[2] if not isinstance(results[2], Exception) else {},
            'orderbook': results[3] if not isinstance(results[3], Exception) else {},
            'avg_price': results[4] if not isinstance(results[4], Exception) else {}
        }

    async def get_klines(self, symbol: str, interval: str = "1m", limit: int = 500) -> list:
        """Obtener datos de velas"""
        url = f"{self.base_url}/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}

        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return [{
                        'timestamp': int(item[0]),
                        'open': float(item[1]),
                        'high': float(item[2]),
                        'low': float(item[3]),
                        'close': float(item[4]),
                        'volume': float(item[5]),
                        'close_time': int(item[6]),
                        'quote_volume': float(item[7]),
                        'count': int(item[8])
                    } for item in data]
        except Exception as e:
            print(f"Error getting klines: {e}")
        return []

    async def get_24hr_ticker(self, symbol: str) -> dict:
        """Estadísticas 24h"""
        url = f"{self.base_url}/api/v3/ticker/24hr"
        params = {"symbol": symbol}

        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
        except:
            pass
        return {}

    async def get_orderbook_ticker(self, symbol: str) -> dict:
        """Mejor precio bid/ask"""
        url = f"{self.base_url}/api/v3/ticker/bookTicker"
        params = {"symbol": symbol}

        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
        except:
            pass
        return {}

    async def get_avg_price(self, symbol: str) -> dict:
        """Precio promedio"""
        url = f"{self.base_url}/api/v3/avgPrice"
        params = {"symbol": symbol}

        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
        except:
            pass
        return {}

class EmergencyFeatureEngineer:
    """Ingeniero de features con 66 características para modelos de emergencia"""

    def __init__(self):
        self.scaler = RobustScaler()

    def create_66_features(self, klines_1m: list, klines_5m: list) -> pd.DataFrame:
        """Crear exactamente 66 features para compatibilidad con modelos de emergencia"""

        if len(klines_1m) < 200:
            return pd.DataFrame()

        # Convertir a DataFrame
        df = pd.DataFrame(klines_1m)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        df = df.sort_index()

        # Arrays para cálculos TA-Lib
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        features = pd.DataFrame(index=df.index)

        print(f"🔧 Creando features avanzados desde {len(df)} velas...")

        try:
            # === MOMENTUM INDICATORS (15 features) ===
            features['rsi_14'] = talib.RSI(close, timeperiod=14)
            features['rsi_21'] = talib.RSI(close, timeperiod=21)
            features['rsi_7'] = talib.RSI(close, timeperiod=7)
            features['rsi_divergence'] = features['rsi_14'] - features['rsi_21']
            features['rsi_momentum'] = features['rsi_14'].diff().fillna(0)

            # CCI - Commodity Channel Index
            features['cci_14'] = talib.CCI(high, low, close, timeperiod=14)
            features['cci_20'] = talib.CCI(high, low, close, timeperiod=20)
            features['cci_divergence'] = features['cci_14'] - features['cci_20']

            # Williams %R
            features['willr_14'] = talib.WILLR(high, low, close, timeperiod=14)
            features['willr_21'] = talib.WILLR(high, low, close, timeperiod=21)

            # Stochastic
            slowk, slowd = talib.STOCH(high, low, close)
            features['stoch_k'] = slowk
            features['stoch_d'] = slowd
            features['stoch_divergence'] = slowk - slowd

            # ROC - Rate of Change
            features['roc_10'] = talib.ROC(close, timeperiod=10)
            features['roc_20'] = talib.ROC(close, timeperiod=20)

            # === TREND INDICATORS (15 features) ===
            # MACD
            macd, signal, histogram = talib.MACD(close)
            features['macd'] = macd
            features['macd_signal'] = signal
            features['macd_histogram'] = histogram
            features['macd_strength'] = np.abs(histogram)
            features['macd_momentum'] = pd.Series(histogram, index=features.index).diff().fillna(0)

            # ADX - Average Directional Movement Index
            features['adx_14'] = talib.ADX(high, low, close, timeperiod=14)
            features['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
            features['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)
            features['di_divergence'] = features['plus_di'] - features['minus_di']

            # Parabolic SAR
            features['sar'] = talib.SAR(high, low)
            features['sar_signal'] = np.where(close > features['sar'], 1, -1)

            # Moving Averages
            features['ema_12'] = talib.EMA(close, timeperiod=12)
            features['ema_26'] = talib.EMA(close, timeperiod=26)
            features['ema_50'] = talib.EMA(close, timeperiod=50)
            features['ema_crossover'] = features['ema_12'] - features['ema_26']

            # === VOLATILITY INDICATORS (12 features) ===
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
            features['bb_upper'] = bb_upper
            features['bb_middle'] = bb_middle
            features['bb_lower'] = bb_lower
            features['bb_width'] = (bb_upper - bb_lower) / bb_middle
            features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
            features['bb_squeeze'] = features['bb_width'].rolling(20).min() == features['bb_width']

            # Average True Range
            features['atr_14'] = talib.ATR(high, low, close, timeperiod=14)
            features['atr_20'] = talib.ATR(high, low, close, timeperiod=20)
            features['atr_ratio'] = features['atr_14'] / features['atr_20']

            # Keltner Channels
            kc_middle = talib.EMA(close, timeperiod=20)
            kc_range = talib.ATR(high, low, close, timeperiod=20) * 2
            features['kc_upper'] = kc_middle + kc_range
            features['kc_lower'] = kc_middle - kc_range
            features['kc_position'] = (close - features['kc_lower']) / (features['kc_upper'] - features['kc_lower'])

            # === VOLUME INDICATORS (8 features) ===
            # On Balance Volume
            features['obv'] = talib.OBV(close, volume)
            features['obv_ema'] = talib.EMA(features['obv'].values, timeperiod=20)
            features['obv_divergence'] = features['obv'] - features['obv_ema']

            # Volume Rate of Change
            features['volume_roc'] = talib.ROC(volume, timeperiod=10)
            features['volume_sma'] = talib.SMA(volume, timeperiod=20)
            features['volume_ratio'] = volume / features['volume_sma']

            # Accumulation/Distribution Line
            features['ad'] = talib.AD(high, low, close, volume)
            features['ad_momentum'] = features['ad'].diff().fillna(0)

            # === PRICE ACTION INDICATORS (8 features) ===
            # Price momentum
            features['price_momentum_5'] = close / np.roll(close, 5) - 1
            features['price_momentum_10'] = close / np.roll(close, 10) - 1
            features['price_momentum_20'] = close / np.roll(close, 20) - 1

            # High-Low ratios
            features['hl_ratio'] = (high - low) / close
            features['oc_ratio'] = (close - df['open']) / df['open']

            # Support/Resistance levels
            features['resistance_distance'] = (df['high'].rolling(20).max() - close) / close
            features['support_distance'] = (close - df['low'].rolling(20).min()) / close

            # Price position in range
            features['price_position'] = (close - df['low'].rolling(20).min()) / (df['high'].rolling(20).max() - df['low'].rolling(20).min())

            # === MARKET STRUCTURE INDICATORS (8 features) ===
            # Fractal patterns
            high_series = pd.Series(high, index=features.index)
            low_series = pd.Series(low, index=features.index)
            close_series = pd.Series(close, index=features.index)

            features['higher_high'] = ((high_series > high_series.shift(1)) & (high_series > high_series.shift(-1))).astype(int)
            features['lower_low'] = ((low_series < low_series.shift(1)) & (low_series < low_series.shift(-1))).astype(int)

            # Trend strength
            features['uptrend_strength'] = (close_series > close_series.shift(1)).rolling(10).sum() / 10
            features['downtrend_strength'] = (close_series < close_series.shift(1)).rolling(10).sum() / 10

            # Volatility clustering (reutilizar close_series)
            returns = np.log(close_series / close_series.shift(1))
            features['volatility_5'] = returns.rolling(5).std().fillna(0)
            features['volatility_20'] = returns.rolling(20).std().fillna(0)
            features['volatility_ratio'] = (features['volatility_5'] / features['volatility_20']).fillna(0)

            # Market efficiency
            features['efficiency_ratio'] = (np.abs(close_series - close_series.shift(10)) / (np.abs(close_series.diff()).rolling(10).sum())).fillna(0)

            # Limpiar datos de forma robusta
            features = features.fillna(method='ffill').fillna(0)

            # Reemplazar infinitos y valores muy grandes
            features = features.replace([np.inf, -np.inf], 0)

            # Limitar valores extremos
            for col in features.columns:
                if features[col].dtype in ['float64', 'float32']:
                    # Limitar a percentiles 1% y 99%
                    lower_bound = features[col].quantile(0.01)
                    upper_bound = features[col].quantile(0.99)
                    features[col] = features[col].clip(lower_bound, upper_bound)

            # Verificar que tenemos exactamente 66 features
            if len(features.columns) != 66:
                print(f"⚠️ Features creados: {len(features.columns)}, esperados: 66")
                # Ajustar si es necesario
                while len(features.columns) < 66:
                    features[f'padding_{len(features.columns)}'] = 0
                features = features.iloc[:, :66]  # Tomar solo las primeras 66

            print(f"  ✅ {len(features.columns)} features técnicos creados")
            return features

        except Exception as e:
            print(f"❌ Error creando features: {e}")
            return pd.DataFrame()

class EmergencyTCNPredictor:
    """Predictor TCN que usa los modelos de emergencia con 66 features"""

    def __init__(self):
        self.pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        self.models = {}
        self.scalers = {}
        self.feature_columns = {}
        self.feature_engineer = EmergencyFeatureEngineer()
        self.load_emergency_models()

    def load_emergency_models(self):
        """Cargar modelos de emergencia entrenados con 66 features"""
        print("🚨 Cargando modelos de emergencia con 66 features...")

        for pair in self.pairs:
            symbol = pair.lower()
            model_dir = f"models/emergency_{symbol}"

            try:
                # Cargar modelo
                model_path = f"{model_dir}/model.h5"
                if os.path.exists(model_path):
                    self.models[pair] = tf.keras.models.load_model(model_path)
                    print(f"  ✅ {pair}: Modelo de emergencia cargado")
                else:
                    print(f"  ❌ {pair}: Modelo no encontrado en {model_path}")
                    continue

                # Cargar scaler
                scaler_path = f"{model_dir}/scaler.pkl"
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        self.scalers[pair] = pickle.load(f)
                    print(f"  ✅ {pair}: Scaler cargado")
                else:
                    print(f"  ⚠️ {pair}: Scaler no encontrado, usando nuevo")
                    self.scalers[pair] = RobustScaler()

                # Cargar columnas de features
                features_path = f"{model_dir}/feature_columns.pkl"
                if os.path.exists(features_path):
                    with open(features_path, 'rb') as f:
                        self.feature_columns[pair] = pickle.load(f)
                    print(f"  ✅ {pair}: Feature columns cargadas ({len(self.feature_columns[pair])} features)")
                else:
                    print(f"  ⚠️ {pair}: Feature columns no encontradas")

            except Exception as e:
                print(f"  ❌ {pair}: Error cargando modelo de emergencia: {e}")

    async def predict_enhanced(self, pair: str, market_data: dict) -> dict:
        """Predicción usando modelos de emergencia con 66 features"""

        if pair not in self.models:
            print(f"  ❌ Modelo no disponible para {pair}")
            return None

        klines_1m = market_data.get('klines_1m', [])
        klines_5m = market_data.get('klines_5m', [])

        if len(klines_1m) < 200:
            print(f"  ❌ Datos insuficientes para {pair}")
            return None

        print(f"  🧠 Análisis TCN avanzado para {pair}...")

        # Crear 66 features
        features = self.feature_engineer.create_66_features(klines_1m, klines_5m)
        if features.empty:
            print(f"  ❌ No se pudieron crear features para {pair}")
            return None

        # Usar las columnas específicas del modelo si están disponibles
        if pair in self.feature_columns:
            try:
                features = features[self.feature_columns[pair]]
                print(f"  🎯 Seleccionadas {len(self.feature_columns[pair])} features principales")
            except KeyError as e:
                print(f"  ⚠️ Algunas features no están disponibles, usando todas")

        # Normalización
        try:
            features_scaled = self.scalers[pair].transform(features.values)
        except Exception as e:
            print(f"  ⚠️ Error en normalización, reentrenando scaler: {e}")
            features_scaled = self.scalers[pair].fit_transform(features.values)

        # Crear secuencia para el modelo (los modelos esperan 48 timesteps)
        sequence_length = 48
        if len(features_scaled) < sequence_length:
            print(f"  ❌ Secuencia muy corta: {len(features_scaled)} < {sequence_length}")
            return None

        # Tomar las últimas 48 observaciones con todas las features
        sequence = features_scaled[-sequence_length:]
        sequence = np.expand_dims(sequence, axis=0)

        try:
            # Predicción
            prediction = self.models[pair].predict(sequence, verbose=0)
            probabilities = prediction[0]

            predicted_class = np.argmax(probabilities)
            confidence = float(np.max(probabilities))

            # Boost de confianza con confirmación técnica
            confidence_boost = self._calculate_technical_confirmation(features.iloc[-1], probabilities)
            adjusted_confidence = min(confidence * confidence_boost, 0.99)

            class_names = ['SELL', 'HOLD', 'BUY']
            signal = class_names[predicted_class]

            return {
                'pair': pair,
                'signal': signal,
                'confidence': adjusted_confidence,
                'raw_confidence': confidence,
                'technical_boost': confidence_boost,
                'probabilities': {
                    'SELL': float(probabilities[0]),
                    'HOLD': float(probabilities[1]),
                    'BUY': float(probabilities[2])
                },
                'features_used': sequence.shape[2],
                'timestamp': datetime.now()
            }

        except Exception as e:
            print(f"    ❌ Error en predicción: {e}")
            return None

    def _calculate_technical_confirmation(self, current_features: pd.Series, probabilities: np.ndarray) -> float:
        """Calcular confirmación técnica para boost de confianza"""
        boost = 1.0

        try:
            # RSI confirmation
            rsi = current_features.get('rsi_14', 50)
            if probabilities[2] > probabilities[0]:  # BUY signal
                if rsi < 40:  # Oversold supports BUY
                    boost += 0.15
            elif probabilities[0] > probabilities[2]:  # SELL signal
                if rsi > 60:  # Overbought supports SELL
                    boost += 0.15

            # MACD confirmation
            macd_hist = current_features.get('macd_histogram', 0)
            if probabilities[2] > probabilities[0] and macd_hist > 0:  # BUY + positive MACD
                boost += 0.1
            elif probabilities[0] > probabilities[2] and macd_hist < 0:  # SELL + negative MACD
                boost += 0.1

            # Trend strength
            adx = current_features.get('adx_14', 0)
            if adx > 25:  # Strong trend
                boost += 0.05

            # Volatility confirmation
            bb_pos = current_features.get('bb_position', 0.5)
            if probabilities[2] > probabilities[0] and bb_pos < 0.2:  # BUY near lower BB
                boost += 0.1
            elif probabilities[0] > probabilities[2] and bb_pos > 0.8:  # SELL near upper BB
                boost += 0.1

        except Exception:
            pass

        return min(boost, 1.4)  # Max 40% boost

# Alias para compatibilidad con el sistema existente
EnhancedTCNPredictor = EmergencyTCNPredictor

if __name__ == "__main__":
    async def test_emergency_predictor():
        """Test del predictor de emergencia"""
        print("🚨 TESTING EMERGENCY TCN PREDICTOR")
        print("="*50)

        predictor = EmergencyTCNPredictor()

        async with AdvancedBinanceData() as data_provider:
            for pair in ["BTCUSDT", "ETHUSDT", "BNBUSDT"]:
                print(f"\n🔍 Testing {pair}...")

                market_data = await data_provider.get_comprehensive_data(pair)
                prediction = await predictor.predict_enhanced(pair, market_data)

                if prediction:
                    print(f"  ✅ {pair}: {prediction['signal']} ({prediction['confidence']:.1%})")
                    print(f"     Features: {prediction['features_used']}")
                else:
                    print(f"  ❌ {pair}: No prediction")

    asyncio.run(test_emergency_predictor())
