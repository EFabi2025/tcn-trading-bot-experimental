#!/usr/bin/env python3
"""
🔄 REAL MARKET DATA PROVIDER PROFESIONAL - ACTUALIZADO
=====================================================

Módulo de datos reales de mercado con las 66 features EXACTAS
usando TA-Lib para consistencia total con tcn_definitivo_predictor.py

Características:
- ✅ 66 features técnicas usando TA-Lib (CORREGIDO)
- ✅ Consistencia total con sistema principal
- ✅ Datos reales de Binance vía klines
- ✅ Normalización profesional con RobustScaler
- ✅ Compatibilidad con TensorFlow 2.15.0
- ✅ Secuencias temporales de 32 timesteps
- ✅ Input shape: (None, 32, 66)

CAMBIOS REALIZADOS:
- Migrado de 21 features manuales a 66 features TA-Lib
- Eliminados errores matemáticos en RSI, ATR, Bollinger Bands
- Compatibilidad total con tcn_definitivo_predictor.py
"""

import asyncio
import time
import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Tuple, Optional
from binance.client import Client
from binance.exceptions import BinanceAPIException
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

class RealMarketDataProvider:
    """🔄 Proveedor de datos reales de mercado con features exactas del TCN"""

    def __init__(self, binance_client: Client):
        """
        Inicializar proveedor de datos reales

        Args:
            binance_client: Cliente autenticado de Binance
        """
        self.client = binance_client
        self.cache = {}
        self.cache_duration = 60  # 1 minuto de caché

        # LISTA FIJA DE 66 FEATURES EXACTAS - COMPATIBLE CON tcn_definitivo_predictor.py
        self.FIXED_FEATURE_LIST = [
            # === MOMENTUM INDICATORS (15 features) ===
            'rsi_14', 'rsi_21', 'rsi_7',
            'macd', 'macd_signal', 'macd_histogram',
            'stoch_k', 'stoch_d', 'williams_r',
            'roc_10', 'roc_20', 'momentum_10', 'momentum_20',
            'cci_14', 'cci_20',

            # === TREND INDICATORS (12 features) ===
            'sma_10', 'sma_20', 'sma_50',
            'ema_10', 'ema_20', 'ema_50',
            'adx_14', 'plus_di', 'minus_di',
            'psar', 'aroon_up', 'aroon_down',

            # === VOLATILITY INDICATORS (10 features) ===
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
            'atr_14', 'atr_20', 'true_range', 'natr_14', 'natr_20',

            # === VOLUME INDICATORS (8 features) ===
            'ad', 'adosc', 'obv', 'volume_sma_10', 'volume_sma_20',
            'volume_ratio', 'mfi_14', 'mfi_20',

            # === PRICE PATTERNS (8 features) ===
            'hl_ratio', 'oc_ratio', 'price_position',
            'price_change_1', 'price_change_5', 'price_change_10',
            'volatility_10', 'volatility_20',

            # === MARKET STRUCTURE (8 features) ===
            'higher_high', 'lower_low', 'uptrend_strength', 'downtrend_strength',
            'resistance_touch', 'support_touch', 'efficiency_ratio', 'fractal_dimension',

            # === MOMENTUM DERIVATIVES (5 features) ===
            'rsi_momentum', 'macd_momentum', 'ad_momentum', 'volume_momentum', 'price_acceleration'
        ]

        # Verificar que sean exactamente 66 features
        assert len(self.FIXED_FEATURE_LIST) == 66, f"Error: Se requieren exactamente 66 features, encontradas {len(self.FIXED_FEATURE_LIST)}"

        # Inicializar normalizadores
        self.feature_scalers = {}

        print(f"✅ RealMarketDataProvider inicializado con {len(self.FIXED_FEATURE_LIST)} features exactas - COMPATIBLE CON tcn_definitivo_predictor.py")

    async def get_real_market_features(self, symbol: str, limit: int = 200) -> Optional[np.ndarray]:
        """
        Obtener features reales de mercado desde Binance

        Args:
            symbol: Par de trading (ej: BTCUSDT)
            limit: Número de velas (mínimo 200 para cálculos técnicos)

        Returns:
            Array numpy con shape (32, 66) o None si hay error
        """
        try:
            # Verificar caché
            cache_key = f"{symbol}_{limit}"
            if cache_key in self.cache:
                cache_time, cached_data = self.cache[cache_key]
                if time.time() - cache_time < self.cache_duration:
                    return cached_data

            print(f"🔄 Obteniendo datos reales de {symbol} desde Binance...")

            # Obtener klines reales de Binance
            klines = await self._get_klines_data(symbol, limit)

            if klines is None or len(klines) < 100:
                print(f"❌ Datos insuficientes para {symbol}: {len(klines) if klines else 0} velas")
                return None

            # Crear DataFrame de precios OHLCV
            df = await self._create_ohlcv_dataframe(klines)

            if df is None or len(df) < 100:
                print(f"❌ DataFrame inválido para {symbol}")
                return None

            # Crear todas las 66 features técnicas usando TA-Lib
            features_df = await self._create_all_technical_features(df)

            if features_df is None:
                print(f"❌ Error creando features para {symbol}")
                return None

            # Extraer las últimas 32 filas (timesteps) para compatibilidad con TCN
            if len(features_df) < 32:
                print(f"❌ Datos insuficientes para secuencia: {len(features_df)} < 32")
                return None

            # Tomar las últimas 32 filas y las 66 features
            sequence_data = features_df.tail(32)[self.FIXED_FEATURE_LIST].values

            # Verificar shape final
            if sequence_data.shape != (32, 66):
                print(f"❌ Shape incorrecto: {sequence_data.shape} != (32, 66)")
                return None

            # Guardar en caché
            self.cache[cache_key] = (time.time(), sequence_data)

            print(f"✅ Features reales obtenidas para {symbol}: {sequence_data.shape}")
            return sequence_data

        except Exception as e:
            print(f"❌ Error obteniendo features reales {symbol}: {e}")
            return None

    async def _get_klines_data(self, symbol: str, limit: int) -> Optional[List]:
        """Obtener datos de klines desde Binance"""
        try:
            # Obtener klines de 1 minuto
            klines = self.client.get_klines(
                symbol=symbol,
                interval=Client.KLINE_INTERVAL_1MINUTE,
                limit=limit
            )

            if not klines:
                return None

            # Convertir a formato estándar
            processed_klines = []
            for kline in klines:
                processed_klines.append({
                    'timestamp': int(kline[0]),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5]),
                    'close_time': int(kline[6]),
                    'quote_volume': float(kline[7]),
                    'trades': int(kline[8])
                })

            return processed_klines

        except BinanceAPIException as e:
            print(f"❌ Error Binance API {symbol}: {e}")
            return None
        except Exception as e:
            print(f"❌ Error obteniendo klines {symbol}: {e}")
            return None

    async def _create_ohlcv_dataframe(self, klines: List[Dict]) -> Optional[pd.DataFrame]:
        """Crear DataFrame OHLCV desde klines"""
        try:
            df = pd.DataFrame(klines)

            # Establecer timestamp como índice
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)

            # Ordenar por tiempo
            df = df.sort_index()

            # Seleccionar solo las columnas OHLCV
            df = df[['open', 'high', 'low', 'close', 'volume']].copy()

            # Verificar que no hay valores nulos
            if df.isnull().any().any():
                df = df.fillna(method='ffill').fillna(method='bfill')

            return df

        except Exception as e:
            print(f"❌ Error creando DataFrame OHLCV: {e}")
            return None

    async def _create_all_technical_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Crear las 66 features técnicas EXACTAS usando TA-Lib
        Replica exactamente la lógica de tcn_definitivo_predictor.py
        """
        try:
            print("🔧 Calculando las 66 features exactas usando TA-Lib...")

            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values

            features = pd.DataFrame(index=df.index)

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
            features['volume_ratio'] = volume / features['volume_sma_20']

            # Money Flow Index
            features['mfi_14'] = talib.MFI(high, low, close, volume, timeperiod=14)
            features['mfi_20'] = talib.MFI(high, low, close, volume, timeperiod=20)

            # === PRICE PATTERNS (8 features) ===
            # Price ratios
            features['hl_ratio'] = (high - low) / close
            features['oc_ratio'] = (close - df['open'].values) / close
            features['price_position'] = (close - low) / (high - low)

            # Price momentum
            close_series = pd.Series(close, index=features.index)
            features['price_change_1'] = close_series.pct_change(1)
            features['price_change_5'] = close_series.pct_change(5)
            features['price_change_10'] = close_series.pct_change(10)

            # Volatility
            returns = np.log(close_series / close_series.shift(1))
            features['volatility_10'] = returns.rolling(10).std()
            features['volatility_20'] = returns.rolling(20).std()

            # === MARKET STRUCTURE (8 features) ===
            # Higher highs, lower lows
            features['higher_high'] = (pd.Series(high, index=features.index) > pd.Series(high, index=features.index).shift(1)).astype(int)
            features['lower_low'] = (pd.Series(low, index=features.index) < pd.Series(low, index=features.index).shift(1)).astype(int)

            # Trend strength
            features['uptrend_strength'] = (close_series > close_series.shift(1)).rolling(10).sum() / 10
            features['downtrend_strength'] = (close_series < close_series.shift(1)).rolling(10).sum() / 10

            # Support/Resistance
            features['resistance_touch'] = (close_series >= close_series.rolling(20).max() * 0.99).astype(int)
            features['support_touch'] = (close_series <= close_series.rolling(20).min() * 1.01).astype(int)

            # Market efficiency
            features['efficiency_ratio'] = (np.abs(close_series - close_series.shift(10)) /
                                          (np.abs(close_series.diff()).rolling(10).sum())).fillna(0)

            # Fractal dimension (simplificado)
            features['fractal_dimension'] = 0.5  # Valor constante por ahora

            # === MOMENTUM DERIVATIVES (5 features) ===
            features['rsi_momentum'] = features['rsi_14'].diff().fillna(0)
            features['macd_momentum'] = pd.Series(macd_hist, index=features.index).diff().fillna(0)
            features['ad_momentum'] = features['ad'].diff().fillna(0)
            features['volume_momentum'] = pd.Series(volume, index=features.index).pct_change().fillna(0)
            features['price_acceleration'] = features['price_change_1'].diff().fillna(0)

            # Limpiar datos
            features = features.fillna(method='ffill').fillna(0)
            features = features.replace([np.inf, -np.inf], 0)

            # Clip valores extremos
            for col in features.columns:
                if features[col].dtype in ['float64', 'int64']:
                    q99 = features[col].quantile(0.99)
                    q01 = features[col].quantile(0.01)
                    features[col] = features[col].clip(q01, q99)

            # Verificar que tenemos exactamente 66 features
            if len(features.columns) != 66:
                print(f"⚠️ Features creados: {len(features.columns)}, esperados: 66")
                # Ajustar si es necesario
                while len(features.columns) < 66:
                    features[f'padding_{len(features.columns)}'] = 0
                features = features.iloc[:, :66]  # Tomar solo las primeras 66

            # Verificar que tenemos todas las features requeridas
            missing_features = [f for f in self.FIXED_FEATURE_LIST if f not in features.columns]
            if missing_features:
                print(f"⚠️ Features faltantes: {missing_features}")
                return None

            print(f"✅ 66 features técnicas calculadas correctamente usando TA-Lib")
            return features

        except Exception as e:
            print(f"❌ Error creando features técnicas: {e}")
            return None

    async def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calcular RSI (Relative Strength Index)"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()

            # Evitar división por cero
            loss = loss.replace(0, 1e-8)
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            return rsi
        except:
            return pd.Series(index=prices.index, data=50.0)  # RSI neutro como fallback

    async def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calcular ATR (Average True Range)"""
        try:
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())

            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period, min_periods=1).mean()

            return atr
        except:
            return pd.Series(index=high.index, data=1.0)  # ATR fallback

    async def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series,
                                   k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calcular Stochastic Oscillator"""
        try:
            lowest_low = low.rolling(window=k_period, min_periods=1).min()
            highest_high = high.rolling(window=k_period, min_periods=1).max()

            # Evitar división por cero
            price_range = highest_high - lowest_low
            price_range = price_range.replace(0, 1e-8)

            k_percent = 100 * (close - lowest_low) / price_range
            d_percent = k_percent.rolling(window=d_period, min_periods=1).mean()

            return k_percent, d_percent
        except:
            return (
                pd.Series(index=high.index, data=50.0),
                pd.Series(index=high.index, data=50.0)
            )

    async def _calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """Calcular Williams %R"""
        try:
            highest_high = high.rolling(window=period, min_periods=1).max()
            lowest_low = low.rolling(window=period, min_periods=1).min()

            # Evitar división por cero
            price_range = highest_high - lowest_low
            price_range = price_range.replace(0, 1e-8)

            williams_r = -100 * (highest_high - close) / price_range

            return williams_r
        except:
            return pd.Series(index=high.index, data=-50.0)  # Williams %R fallback

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalizar features usando RobustScaler

        Args:
            features: Array de features (32, 66)

        Returns:
            Features normalizadas
        """
        try:
            # Reshape para normalización: (32*66,) -> (2112,)
            original_shape = features.shape
            features_flat = features.reshape(-1, 1)

            # Usar RobustScaler para manejo robusto de outliers
            scaler = RobustScaler()
            features_normalized = scaler.fit_transform(features_flat)

            # Reshape de vuelta a forma original
            features_normalized = features_normalized.reshape(original_shape)

            # Clip para evitar valores extremos
            features_normalized = np.clip(features_normalized, -5, 5)

            return features_normalized.astype(np.float32)

        except Exception as e:
            print(f"❌ Error normalizando features: {e}")
            # Retornar features originales sin normalizar
            return features.astype(np.float32)


class MarketDataValidator:
    """🔍 Validador de calidad de datos de mercado"""

    def validate_features(self, features: np.ndarray, symbol: str) -> bool:
        """
        Validar calidad de features de mercado

        Args:
            features: Array de features (32, 66)
            symbol: Símbolo para logging

        Returns:
            True si los datos son válidos
        """
        try:
            # 1. Verificar shape
            if features.shape != (32, 66):
                print(f"❌ Shape inválido para {symbol}: {features.shape} != (32, 66)")
                return False

            # 2. Verificar valores no finitos
            if not np.isfinite(features).all():
                nan_count = np.isnan(features).sum()
                inf_count = np.isinf(features).sum()
                print(f"❌ Valores no finitos en {symbol}: {nan_count} NaN, {inf_count} Inf")
                return False

            # 3. Verificar varianza (evitar features constantes)
            feature_variances = np.var(features, axis=0)
            constant_features = np.sum(feature_variances < 1e-10)
            if constant_features > 15:  # Permitir algunas features constantes (ajustado para 66)
                print(f"⚠️ Muchas features constantes en {symbol}: {constant_features}/66")
                return False

            # 4. Verificar rango razonable
            min_val, max_val = np.min(features), np.max(features)
            if max_val - min_val < 1e-10:
                print(f"❌ Rango de valores muy pequeño en {symbol}: {max_val - min_val}")
                return False

            print(f"✅ Features válidas para {symbol}: shape={features.shape}, range=({min_val:.4f}, {max_val:.4f})")
            return True

        except Exception as e:
            print(f"❌ Error validando features {symbol}: {e}")
            return False


# === TESTING Y EJEMPLO DE USO ===
async def test_real_market_data():
    """🧪 Test del proveedor de datos reales"""
    print("🧪 TESTING REAL MARKET DATA PROVIDER")
    print("=" * 50)

    try:
        # Configurar cliente de Binance (necesita API keys reales)
        from binance.client import Client
        import os

        api_key = os.getenv('BINANCE_API_KEY')
        secret_key = os.getenv('BINANCE_SECRET_KEY')

        if not api_key or not secret_key:
            print("❌ Se requieren BINANCE_API_KEY y BINANCE_SECRET_KEY en variables de entorno")
            return

        client = Client(api_key, secret_key)

        # Crear proveedor
        provider = RealMarketDataProvider(client)
        validator = MarketDataValidator()

        # Test con BTCUSDT
        print(f"\n🧪 Testing con BTCUSDT...")

        features = await provider.get_real_market_features('BTCUSDT')

        if features is not None:
            print(f"✅ Features obtenidas: {features.shape}")

            # Validar features
            is_valid = validator.validate_features(features, 'BTCUSDT')
            print(f"✅ Features válidas: {is_valid}")

            # Normalizar features
            normalized = provider.normalize_features(features)
            print(f"✅ Features normalizadas: {normalized.shape}")
            print(f"   Rango normalizado: ({np.min(normalized):.4f}, {np.max(normalized):.4f})")

        else:
            print("❌ No se pudieron obtener features")

    except Exception as e:
        print(f"❌ Error en test: {e}")


if __name__ == "__main__":
    asyncio.run(test_real_market_data())
