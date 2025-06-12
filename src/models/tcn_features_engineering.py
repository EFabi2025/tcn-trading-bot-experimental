#!/usr/bin/env python3
"""
🔧 FEATURE ENGINEERING AVANZADO PARA TCN ANTI-SESGO
==================================================

Implementación completa del sistema de features engineering
para eliminar sesgos de mercado en modelos TCN.

Features incluidos:
- Indicadores técnicos multitimeframe
- Features de régimen de mercado
- Features de microestructura
- Features de sentiment
- Normalización robusta
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import RobustScaler, StandardScaler
import talib
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """
    Feature Engineering avanzado para TCN consciente del régimen

    Características clave:
    - Features específicos por régimen de mercado
    - Indicadores multitimeframe
    - Análisis de microestructura
    - Normalización robusta anti-outliers
    """

    def __init__(self):
        self.scalers = {}
        self.feature_columns = []
        self.regime_columns = ['regime_bull', 'regime_bear', 'regime_sideways']

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcular RSI usando TA-Lib para máxima precisión"""
        try:
            return pd.Series(talib.RSI(prices.values, timeperiod=period), index=prices.index)
        except:
            # Fallback manual si TA-Lib no está disponible
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-10)
            return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calcular MACD completo"""
        try:
            macd, signal_line, histogram = talib.MACD(prices.values, fastperiod=fast, slowperiod=slow, signalperiod=signal)
            return {
                'macd': pd.Series(macd, index=prices.index),
                'signal': pd.Series(signal_line, index=prices.index),
                'histogram': pd.Series(histogram, index=prices.index)
            }
        except:
            # Fallback manual
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal).mean()
            histogram = macd - signal_line

            return {
                'macd': macd,
                'signal': signal_line,
                'histogram': histogram
            }

    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Calcular Bollinger Bands"""
        try:
            upper, middle, lower = talib.BBANDS(prices.values, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
            return {
                'upper': pd.Series(upper, index=prices.index),
                'middle': pd.Series(middle, index=prices.index),
                'lower': pd.Series(lower, index=prices.index)
            }
        except:
            # Fallback manual
            middle = prices.rolling(period).mean()
            std = prices.rolling(period).std()
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)

            return {
                'upper': upper,
                'middle': middle,
                'lower': lower
            }

    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calcular Average True Range"""
        try:
            return pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=period), index=close.index)
        except:
            # Fallback manual
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            return true_range.rolling(period).mean()

    def prepare_advanced_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, RobustScaler]]:
        """
        Prepara features técnicos avanzados para el modelo TCN

        Incluye:
        - Indicadores técnicos tradicionales
        - Features de régimen de mercado
        - Features de microestructura
        - Features de sentiment
        """
        print("🔧 Preparing advanced features...")

        df = df.copy()
        feature_df = pd.DataFrame(index=df.index)

        # === FEATURES TÉCNICOS BÁSICOS ===
        print("   📊 Computing basic technical indicators...")

        # RSI en múltiples timeframes
        feature_df['rsi_14'] = self.calculate_rsi(df['close'], 14)
        feature_df['rsi_21'] = self.calculate_rsi(df['close'], 21)
        feature_df['rsi_divergence'] = feature_df['rsi_14'] - feature_df['rsi_21']

        # MACD completo
        macd_data = self.calculate_macd(df['close'])
        feature_df['macd'] = macd_data['macd']
        feature_df['macd_signal'] = macd_data['signal']
        feature_df['macd_histogram'] = macd_data['histogram']
        feature_df['macd_momentum'] = feature_df['macd_histogram'].diff()

        # Bollinger Bands
        bb_data = self.calculate_bollinger_bands(df['close'], 20, 2)
        feature_df['bb_upper'] = bb_data['upper']
        feature_df['bb_middle'] = bb_data['middle']
        feature_df['bb_lower'] = bb_data['lower']
        feature_df['bb_width'] = (bb_data['upper'] - bb_data['lower']) / bb_data['middle']
        feature_df['bb_position'] = (df['close'] - bb_data['lower']) / (bb_data['upper'] - bb_data['lower'] + 1e-10)

        # ATR (Average True Range)
        feature_df['atr'] = self.calculate_atr(df['high'], df['low'], df['close'], 14)
        feature_df['atr_normalized'] = feature_df['atr'] / df['close']

        # === MEDIAS MÓVILES MÚLTIPLES ===
        print("   📈 Computing moving averages...")

        for period in [10, 20, 50, 100]:
            feature_df[f'sma_{period}'] = df['close'].rolling(period).mean()
            feature_df[f'ema_{period}'] = df['close'].ewm(span=period).mean()

            # Distancia del precio a la media móvil
            feature_df[f'price_to_sma_{period}'] = (df['close'] - feature_df[f'sma_{period}']) / feature_df[f'sma_{period}']
            feature_df[f'price_to_ema_{period}'] = (df['close'] - feature_df[f'ema_{period}']) / feature_df[f'ema_{period}']

        # Señales de cruce de medias móviles
        feature_df['ma_signal_10_20'] = (feature_df['sma_10'] - feature_df['sma_20']) / feature_df['sma_20']
        feature_df['ma_signal_20_50'] = (feature_df['sma_20'] - feature_df['sma_50']) / feature_df['sma_50']

        # === FEATURES DE RÉGIMEN ===
        print("   🎯 Computing regime-specific features...")

        # Volatilidad multitimeframe
        feature_df['volatility_5m'] = df['close'].rolling(5).std() / df['close'].rolling(5).mean()
        feature_df['volatility_1h'] = df['close'].rolling(12).std() / df['close'].rolling(12).mean()  # 12 * 5m = 1h
        feature_df['volatility_4h'] = df['close'].rolling(48).std() / df['close'].rolling(48).mean()  # 48 * 5m = 4h
        feature_df['volatility_24h'] = df['close'].rolling(288).std() / df['close'].rolling(288).mean()  # 288 * 5m = 24h

        # Ratios de volatilidad
        feature_df['vol_ratio_1h_4h'] = feature_df['volatility_1h'] / (feature_df['volatility_4h'] + 1e-10)
        feature_df['vol_ratio_4h_24h'] = feature_df['volatility_4h'] / (feature_df['volatility_24h'] + 1e-10)

        # Momentum multitimeframe
        feature_df['momentum_5m'] = df['close'].pct_change(1)
        feature_df['momentum_30m'] = df['close'].pct_change(6)   # 6 * 5m = 30m
        feature_df['momentum_1h'] = df['close'].pct_change(12)   # 12 * 5m = 1h
        feature_df['momentum_4h'] = df['close'].pct_change(48)   # 48 * 5m = 4h
        feature_df['momentum_24h'] = df['close'].pct_change(288) # 288 * 5m = 24h

        # === FEATURES DE MICROESTRUCTURA ===
        print("   🔬 Computing microstructure features...")

        # Análisis de volumen
        feature_df['volume_sma_20'] = df['volume'].rolling(20).mean()
        feature_df['volume_ratio'] = df['volume'] / (feature_df['volume_sma_20'] + 1e-10)
        feature_df['volume_trend'] = (df['volume'].rolling(10).mean() /
                                     (df['volume'].rolling(30).mean() + 1e-10))

        # Volume Price Trend (VPT)
        feature_df['vpt'] = (df['volume'] * df['close'].pct_change()).cumsum()
        feature_df['vpt_sma'] = feature_df['vpt'].rolling(20).mean()
        feature_df['vpt_signal'] = feature_df['vpt'] - feature_df['vpt_sma']

        # Price action patterns
        feature_df['price_range'] = (df['high'] - df['low']) / df['close']
        feature_df['body_size'] = abs(df['close'] - df['open']) / df['close']
        feature_df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
        feature_df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']

        # Gaps
        feature_df['gap_up'] = np.where(df['open'] > df['close'].shift(),
                                       (df['open'] - df['close'].shift()) / df['close'].shift(), 0)
        feature_df['gap_down'] = np.where(df['open'] < df['close'].shift(),
                                         (df['close'].shift() - df['open']) / df['close'].shift(), 0)

        # === FEATURES DE MOMENTUM AVANZADO ===
        print("   ⚡ Computing advanced momentum features...")

        # Williams %R
        highest_high = df['high'].rolling(14).max()
        lowest_low = df['low'].rolling(14).min()
        feature_df['williams_r'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low + 1e-10)

        # Stochastic Oscillator
        feature_df['stoch_k'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low + 1e-10)
        feature_df['stoch_d'] = feature_df['stoch_k'].rolling(3).mean()

        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            feature_df[f'roc_{period}'] = df['close'].pct_change(period) * 100

        # === FEATURES DE TENDENCIA ===
        print("   📈 Computing trend features...")

        # Parabolic SAR simulation (simplified)
        feature_df['trend_strength'] = feature_df['momentum_1h'].rolling(10).mean()
        feature_df['trend_consistency'] = (feature_df['momentum_1h'] > 0).rolling(10).mean()

        # Ichimoku components (simplified)
        high_9 = df['high'].rolling(9).max()
        low_9 = df['low'].rolling(9).min()
        feature_df['tenkan_sen'] = (high_9 + low_9) / 2

        high_26 = df['high'].rolling(26).max()
        low_26 = df['low'].rolling(26).min()
        feature_df['kijun_sen'] = (high_26 + low_26) / 2

        feature_df['ichimoku_signal'] = (feature_df['tenkan_sen'] - feature_df['kijun_sen']) / feature_df['kijun_sen']

        # === ONE-HOT ENCODING PARA RÉGIMEN ===
        if 'regime' in df.columns:
            print("   🎯 Adding regime one-hot encoding...")
            regime_dummies = pd.get_dummies(df['regime'], prefix='regime')
            for col in self.regime_columns:
                if col in regime_dummies.columns:
                    feature_df[col] = regime_dummies[col]
                else:
                    feature_df[col] = 0
        else:
            # Default si no hay régimen
            for col in self.regime_columns:
                feature_df[col] = 0
            feature_df['regime_sideways'] = 1  # Default conservador

        # === LIMPIAR DATOS ===
        print("   🧹 Cleaning data...")

        # Eliminar infinitos y NaN
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

        # Forward fill y luego backward fill para NaN
        feature_df = feature_df.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # CRÍTICO: Asegurar que todos los datos sean float64
        for col in feature_df.columns:
            if not col.startswith('regime_'):
                feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce').astype(np.float64)

        # Verificar que no hay NaN después de la conversión
        feature_df = feature_df.fillna(0.0)

        # === NORMALIZACIÓN ===
        print("   📏 Normalizing features...")

        # Features que necesitan normalización
        features_to_normalize = [col for col in feature_df.columns if not col.startswith('regime_')]

        scalers = {}
        normalized_df = feature_df.copy()

        # Usar RobustScaler para resistir outliers
        scaler = RobustScaler()

        # Asegurar que los datos de entrada sean float64
        data_to_scale = feature_df[features_to_normalize].astype(np.float64)
        normalized_data = scaler.fit_transform(data_to_scale)

        # Asegurar que la salida también sea float64
        normalized_df[features_to_normalize] = normalized_data.astype(np.float64)
        scalers['main'] = scaler

        # Guardar columnas de features para uso posterior
        self.feature_columns = [col for col in normalized_df.columns if not col.startswith('regime_')]

        print(f"✅ Feature engineering completed: {len(normalized_df.columns)} features created")

        return normalized_df, scalers

    def create_training_sequences(self, df: pd.DataFrame, sequence_length: int = 60,
                                prediction_horizon: int = 1, price_threshold: float = 0.005) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Crea secuencias temporales para entrenamiento del TCN

        Args:
            df: DataFrame con features y precios
            sequence_length: Longitud de secuencia temporal
            prediction_horizon: Horizonte de predicción
            price_threshold: Umbral para clasificar BUY/SELL

        Returns:
            X_price: Array de secuencias de precios/features (samples, timesteps, features)
            X_regime: Array de contexto de régimen (samples, 3)
            y: Array de labels (samples, 3) - BUY/HOLD/SELL
        """
        print(f"🔄 Creating training sequences (length={sequence_length}, horizon={prediction_horizon})...")

        # Seleccionar features para el modelo
        price_features = [col for col in df.columns if not col.startswith('regime_') and col not in ['close', 'regime']]
        regime_features = self.regime_columns

        X_price, X_regime, y = [], [], []

        # Necesitamos precios originales para calcular labels
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column for label calculation")

        for i in range(sequence_length, len(df) - prediction_horizon):
            # Secuencia de features de precios
            price_sequence = df[price_features].iloc[i-sequence_length:i].values.astype(np.float32)
            X_price.append(price_sequence)

            # Contexto de régimen (del período actual)
            regime_context = df[regime_features].iloc[i].values.astype(np.float32)
            X_regime.append(regime_context)

            # Label (predicción futura)
            current_price = float(df['close'].iloc[i])
            future_price = float(df['close'].iloc[i + prediction_horizon])
            price_change = (future_price - current_price) / current_price

            # Clasificar en BUY/HOLD/SELL basado en cambio de precio
            if price_change > price_threshold:      # +0.5% default -> BUY
                label = [1.0, 0.0, 0.0]
            elif price_change < -price_threshold:   # -0.5% default -> SELL
                label = [0.0, 0.0, 1.0]
            else:                                   # [-0.5%, +0.5%] -> HOLD
                label = [0.0, 1.0, 0.0]

            y.append(label)

        X_price = np.array(X_price, dtype=np.float32)
        X_regime = np.array(X_regime, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        print(f"✅ Sequences created:")
        print(f"   X_price shape: {X_price.shape}")
        print(f"   X_regime shape: {X_regime.shape}")
        print(f"   y shape: {y.shape}")

        # Estadísticas de labels
        label_counts = np.sum(y, axis=0)
        total = len(y)
        print(f"   Label distribution:")
        print(f"     BUY: {label_counts[0]:,} ({label_counts[0]/total*100:.1f}%)")
        print(f"     HOLD: {label_counts[1]:,} ({label_counts[1]/total*100:.1f}%)")
        print(f"     SELL: {label_counts[2]:,} ({label_counts[2]/total*100:.1f}%)")

        return X_price, X_regime, y

    def regime_stratified_split(self, df: pd.DataFrame, test_size: float = 0.2,
                               validation_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split estratificado que garantiza representación balanceada
        de todos los regímenes en train/validation/test
        """
        print("📊 Creating stratified splits by regime...")

        if 'regime' not in df.columns:
            print("⚠️ No regime column found, using simple temporal split")
            n_total = len(df)
            n_test = int(n_total * test_size)
            n_val = int(n_total * validation_size)

            train_df = df.iloc[:-n_test-n_val]
            val_df = df.iloc[-n_test-n_val:-n_test]
            test_df = df.iloc[-n_test:]

            return train_df, val_df, test_df

        # Separar por régimen para estratificación
        regimes = {}
        for regime_type in ['bull', 'bear', 'sideways']:
            regime_data = df[df['regime'] == regime_type]
            if len(regime_data) > 0:
                regimes[regime_type] = regime_data.copy()
                print(f"   {regime_type}: {len(regime_data):,} samples")

        # Split estratificado por régimen
        train_data, val_data, test_data = [], [], []

        for regime_type, regime_df in regimes.items():
            n_samples = len(regime_df)
            n_test = int(n_samples * test_size)
            n_val = int(n_samples * validation_size)
            n_train = n_samples - n_test - n_val

            # Split temporal para evitar data leakage
            regime_train = regime_df.iloc[:n_train]
            regime_val = regime_df.iloc[n_train:n_train+n_val]
            regime_test = regime_df.iloc[n_train+n_val:]

            train_data.append(regime_train)
            val_data.append(regime_val)
            test_data.append(regime_test)

            print(f"   {regime_type} split: {len(regime_train)} train, {len(regime_val)} val, {len(regime_test)} test")

        # Concatenar y mezclar
        train_df = pd.concat(train_data).sample(frac=1, random_state=42).reset_index(drop=True)
        val_df = pd.concat(val_data).sample(frac=1, random_state=42).reset_index(drop=True)
        test_df = pd.concat(test_data).sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"✅ Stratified split completed:")
        print(f"   Train: {len(train_df):,} samples")
        print(f"   Validation: {len(val_df):,} samples")
        print(f"   Test: {len(test_df):,} samples")

        return train_df, val_df, test_df

def main():
    """Demo del feature engineering avanzado"""
    print("🚀 Starting Advanced Feature Engineering Demo")

    # Para la demo, crear datos de prueba
    dates = pd.date_range('2023-01-01', periods=1000, freq='5T')

    # Simular datos OHLCV
    np.random.seed(42)
    base_price = 50000
    returns = np.random.normal(0, 0.02, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))

    demo_df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, len(dates)),
        'regime': np.random.choice(['bull', 'bear', 'sideways'], len(dates))
    })

    demo_df.set_index('timestamp', inplace=True)

    print(f"📊 Demo data shape: {demo_df.shape}")

    # Crear feature engineer
    engineer = AdvancedFeatureEngineer()

    try:
        # Preparar features
        print("\n1️⃣ Preparing advanced features...")
        features_df, scalers = engineer.prepare_advanced_features(demo_df)

        print(f"✅ Features shape: {features_df.shape}")
        print(f"✅ Features created: {list(features_df.columns[:10])}...")  # Mostrar primeras 10

        # Split estratificado
        print("\n2️⃣ Creating stratified split...")
        features_df['close'] = demo_df['close']  # Añadir close para labels
        features_df['regime'] = demo_df['regime']  # Añadir regime para split

        train_df, val_df, test_df = engineer.regime_stratified_split(features_df)

        # Crear secuencias
        print("\n3️⃣ Creating training sequences...")
        X_price, X_regime, y = engineer.create_training_sequences(train_df, sequence_length=60)

        print(f"✅ Training sequences ready for TCN model!")

    except Exception as e:
        print(f"❌ Error in demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
