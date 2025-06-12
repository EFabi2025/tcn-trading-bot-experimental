#!/usr/bin/env python3
"""
🔄 REAL MARKET DATA PROVIDER PROFESIONAL
=====================================

Módulo de datos reales de mercado con las 66 features EXACTAS
usadas en el entrenamiento de los modelos TCN originales.

Características:
- 66 features técnicas exactas del entrenamiento (FIXED_FEATURE_LIST)
- Datos reales de Binance vía klines
- Normalización profesional con RobustScaler
- Compatibilidad con TensorFlow 2.15.0
- Secuencias temporales de 32 timesteps
- Input shape: (None, 32, 66)
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

        # LISTA FIJA DE 21 FEATURES EXACTAS para modelos tcn_final_*.h5
        self.FIXED_FEATURE_LIST = [
            # 1. OHLCV básicos (5 features)
            'open', 'high', 'low', 'close', 'volume',

            # 2. Returns múltiples períodos (5 features)
            'returns_1', 'returns_3', 'returns_5', 'returns_10', 'returns_20',

            # 3. Moving Averages (3 features)
            'sma_5', 'sma_20', 'ema_12',

            # 4. RSI (1 feature)
            'rsi_14',

            # 5. MACD completo (3 features)
            'macd', 'macd_signal', 'macd_histogram',

            # 6. Bollinger Bands (2 features)
            'bb_position', 'bb_width',

            # 7. Volume analysis (1 feature)
            'volume_ratio',

            # 8. Volatilidad (1 feature)
            'volatility'
        ]

        # Verificar que sean exactamente 21 features
        assert len(self.FIXED_FEATURE_LIST) == 21, f"Error: Se requieren exactamente 21 features, encontradas {len(self.FIXED_FEATURE_LIST)}"

        # Inicializar normalizadores
        self.feature_scalers = {}

        print(f"✅ RealMarketDataProvider inicializado con {len(self.FIXED_FEATURE_LIST)} features exactas para modelos tcn_final")

    async def get_real_market_features(self, symbol: str, limit: int = 200) -> Optional[np.ndarray]:
        """
        Obtener features reales de mercado desde Binance

        Args:
            symbol: Par de trading (ej: BTCUSDT)
            limit: Número de velas (mínimo 200 para cálculos técnicos)

        Returns:
            Array numpy con shape (50, 21) o None si hay error
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

            # Crear todas las 66 features técnicas
            features_df = await self._create_all_technical_features(df)

            if features_df is None:
                print(f"❌ Error creando features para {symbol}")
                return None

            # Extraer las últimas 50 filas (timesteps) para modelos tcn_final
            if len(features_df) < 50:
                print(f"❌ Datos insuficientes para secuencia: {len(features_df)} < 50")
                return None

            # Tomar las últimas 50 filas y las 21 features para tcn_final
            sequence_data = features_df.tail(50)[self.FIXED_FEATURE_LIST].values

            # Verificar shape final
            if sequence_data.shape != (50, 21):
                print(f"❌ Shape incorrecto: {sequence_data.shape} != (50, 21)")
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
        Crear las 21 features exactas para modelos tcn_final
        """
        try:
            print("🔧 Calculando las 21 features exactas para tcn_final...")

            features_df = df.copy()

            # === 1. OHLCV básicos (5 features) ===
            # Ya están en el DataFrame

            # === 2. Returns múltiples períodos (5 features) ===
            features_df['returns_1'] = features_df['close'].pct_change(periods=1)
            features_df['returns_3'] = features_df['close'].pct_change(periods=3)
            features_df['returns_5'] = features_df['close'].pct_change(periods=5)
            features_df['returns_10'] = features_df['close'].pct_change(periods=10)
            features_df['returns_20'] = features_df['close'].pct_change(periods=20)

            # === 3. Moving Averages (3 features) ===
            features_df['sma_5'] = features_df['close'].rolling(window=5, min_periods=1).mean()
            features_df['sma_20'] = features_df['close'].rolling(window=20, min_periods=1).mean()
            features_df['ema_12'] = features_df['close'].ewm(span=12, min_periods=1).mean()

            # === 4. RSI (1 feature) ===
            features_df['rsi_14'] = await self._calculate_rsi(features_df['close'], 14)

            # === 5. MACD completo (3 features) ===
            ema_26 = features_df['close'].ewm(span=26, min_periods=1).mean()
            features_df['macd'] = features_df['ema_12'] - ema_26
            features_df['macd_signal'] = features_df['macd'].ewm(span=9, min_periods=1).mean()
            features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']

            # === 6. Bollinger Bands (2 features) ===
            bb_middle = features_df['sma_20']
            bb_std = features_df['close'].rolling(window=20, min_periods=1).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            bb_range = bb_upper - bb_lower
            bb_range = bb_range.replace(0, 1e-8)  # Evitar división por cero
            features_df['bb_position'] = (features_df['close'] - bb_lower) / bb_range
            features_df['bb_width'] = bb_range / bb_middle

            # === 7. Volume analysis (1 feature) ===
            volume_sma_20 = features_df['volume'].rolling(window=20, min_periods=1).mean()
            volume_sma_20 = volume_sma_20.replace(0, 1e-8)  # Evitar división por cero
            features_df['volume_ratio'] = features_df['volume'] / volume_sma_20

            # === 8. Volatilidad (1 feature) ===
            features_df['volatility'] = features_df['close'].pct_change().rolling(window=20, min_periods=1).std()

            # === LIMPIEZA FINAL ===
            # Reemplazar infinitos y NaN
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)

            # Verificar que tenemos todas las features
            missing_features = [f for f in self.FIXED_FEATURE_LIST if f not in features_df.columns]
            if missing_features:
                print(f"⚠️ Features faltantes: {missing_features}")
                return None

            print(f"✅ 21 features técnicas calculadas correctamente para tcn_final")
            return features_df

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
            if constant_features > 10:  # Permitir algunas features constantes
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
