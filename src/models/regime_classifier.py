#!/usr/bin/env python3
"""
🎯 CLASIFICADOR DE REGÍMENES DE MERCADO MEJORADO
=============================================

Implementación mejorada del sistema de clasificación automática de regímenes
para el TCN Anti-Sesgo con múltiples criterios de clasificación.

Regímenes detectados:
- Bull Market: Tendencia alcista sostenida
- Bear Market: Tendencia bajista sostenida
- Sideways: Mercado lateral/consolidación
"""

import os
import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
from binance.client import Client
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

class MarketRegimeClassifier:
    """
    Clasificador automático de regímenes de mercado mejorado

    Detecta automáticamente 3 tipos de mercado usando múltiples criterios:
    - Bull markets (tendencia alcista sostenida)
    - Bear markets (tendencia bajista sostenida)
    - Sideways markets (consolidación/lateral)
    """

    def __init__(self, trend_window=20, trend_threshold=0.015, momentum_threshold=0.015):
        self.trend_window = trend_window
        self.trend_threshold = trend_threshold
        self.momentum_threshold = momentum_threshold
        self.client = None
        self.regime_history = []
        self.initialize_binance()

    def initialize_binance(self):
        """Inicializar cliente Binance"""
        try:
            # Cargar variables de entorno si no están cargadas
            from dotenv import load_dotenv
            load_dotenv()

            api_key = os.environ.get("BINANCE_API_KEY")
            api_secret = os.environ.get("BINANCE_SECRET_KEY")  # ✅ Corregido

            if api_key and api_secret:
                self.client = Client(api_key, api_secret, testnet=False)
                print("✅ Binance client initialized with API keys")
            else:
                self.client = Client()
                print("⚠️ Binance client initialized without API keys (limited access)")
        except Exception as e:
            self.client = Client()
            print(f"⚠️ Error initializing Binance client: {e}")

    def download_balanced_data(self, symbols=None, days=730):
        """
        Descarga datos balanceados de múltiples símbolos para entrenamiento anti-sesgo

        Args:
            symbols: Lista de símbolos a descargar
            days: Días de historia a descargar

        Returns:
            DataFrame con datos OHLCV + símbolo
        """
        if symbols is None:
            symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT']

        print(f"📊 Downloading {days} days of data for {len(symbols)} symbols...")

        all_data = []
        for symbol in symbols:
            try:
                print(f"   📈 Downloading {symbol}...")

                # Descargar datos históricos con 5m timeframe
                klines = self.client.get_historical_klines(
                    symbol=symbol,
                    interval='5m',
                    start_str=f"{days} days ago UTC"
                )

                if not klines:
                    print(f"   ❌ No data for {symbol}")
                    continue

                # Crear DataFrame
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])

                # Limpiar datos
                df['symbol'] = symbol
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                df[numeric_cols] = df[numeric_cols].astype(float)

                # Convertir timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)

                # Seleccionar columnas relevantes
                df = df[['symbol', 'open', 'high', 'low', 'close', 'volume']]

                all_data.append(df)
                print(f"   ✅ {symbol}: {len(df):,} samples")

            except Exception as e:
                print(f"   ❌ Error downloading {symbol}: {e}")
                continue

        if not all_data:
            raise Exception("No data downloaded successfully")

        # Concatenar todos los datos
        combined_df = pd.concat(all_data).sort_index()
        print(f"📊 Total dataset: {len(combined_df):,} samples from {len(all_data)} symbols")

        return combined_df

    def classify_market_regimes_improved(self, df):
        """
        Clasifica automáticamente regímenes de mercado usando múltiples criterios

        Mejoras:
        - Múltiples indicadores de tendencia
        - Momentum multitimeframe
        - Contexto de volatilidad
        - Criterios de confirmación
        """
        print("🎯 Classifying market regimes with improved algorithm...")

        df = df.copy()
        df['regime'] = 'sideways'  # Default

        # Clasificar por símbolo individualmente
        for symbol in df['symbol'].unique():
            symbol_mask = df['symbol'] == symbol
            symbol_data = df[symbol_mask].copy()

            if len(symbol_data) < 144:  # Necesitamos al menos 12h de datos
                continue

            # === INDICADORES DE TENDENCIA ===

            # 1. Medias móviles multitimeframe
            symbol_data['sma_20'] = symbol_data['close'].rolling(20).mean()
            symbol_data['sma_50'] = symbol_data['close'].rolling(50).mean()
            symbol_data['ema_20'] = symbol_data['close'].ewm(span=20).mean()

            # 2. Momentum multitimeframe
            symbol_data['momentum_1h'] = symbol_data['close'].pct_change(12)   # 12 * 5m = 1h
            symbol_data['momentum_4h'] = symbol_data['close'].pct_change(48)   # 48 * 5m = 4h
            symbol_data['momentum_12h'] = symbol_data['close'].pct_change(144) # 144 * 5m = 12h

            # 3. Trend strength respecto a MA
            symbol_data['ma_trend'] = (symbol_data['close'] - symbol_data['sma_20']) / symbol_data['sma_20']
            symbol_data['ma_direction'] = symbol_data['sma_20'] > symbol_data['sma_50']

            # 4. Contexto de volatilidad
            symbol_data['volatility_4h'] = (
                symbol_data['close'].rolling(48).std() /
                symbol_data['close'].rolling(48).mean()
            )
            symbol_data['volatility_percentile'] = (
                symbol_data['volatility_4h'].rolling(288).rank(pct=True)  # Percentil en 24h
            )

            # 5. Price action momentum
            symbol_data['price_momentum'] = (
                symbol_data['close'].rolling(10).mean().pct_change(10)
            )

            # === CRITERIOS DE CLASIFICACIÓN MEJORADOS ===

            # BULL MARKET: Múltiples señales alcistas
            bull_signals = (
                # Momentum fuerte positivo
                (symbol_data['momentum_4h'] > self.momentum_threshold) |
                # Precio sobre MA + dirección alcista
                ((symbol_data['ma_trend'] > 0.01) & symbol_data['ma_direction']) |
                # Momentum de precio consistente
                (symbol_data['price_momentum'] > 0.01) |
                # Momentum 12h muy fuerte
                (symbol_data['momentum_12h'] > 0.03)
            )

            # Confirmación adicional para bull
            bull_confirmation = (
                (symbol_data['close'] > symbol_data['sma_20']) &  # Por encima de MA
                (symbol_data['momentum_1h'].rolling(6).mean() > 0)  # Momentum promedio positivo
            )

            bull_condition = bull_signals & bull_confirmation

            # BEAR MARKET: Múltiples señales bajistas
            bear_signals = (
                # Momentum fuerte negativo
                (symbol_data['momentum_4h'] < -self.momentum_threshold) |
                # Precio bajo MA + dirección bajista
                ((symbol_data['ma_trend'] < -0.01) & ~symbol_data['ma_direction']) |
                # Momentum de precio negativo
                (symbol_data['price_momentum'] < -0.01) |
                # Momentum 12h muy negativo
                (symbol_data['momentum_12h'] < -0.03)
            )

            # Confirmación adicional para bear
            bear_confirmation = (
                (symbol_data['close'] < symbol_data['sma_20']) &  # Por debajo de MA
                (symbol_data['momentum_1h'].rolling(6).mean() < 0)  # Momentum promedio negativo
            )

            bear_condition = bear_signals & bear_confirmation

            # SIDEWAYS: Todo lo que no es claramente bull o bear
            # Con filtro adicional para volatilidad normal
            sideways_condition = (
                ~bull_condition & ~bear_condition &
                (symbol_data['volatility_percentile'] < 0.8)  # No alta volatilidad extrema
            )

            # === APLICAR CLASIFICACIÓN ===

            # Inicializar con sideways
            symbol_data['regime'] = 'sideways'

            # Aplicar con prioridad: bull > bear > sideways
            symbol_data.loc[bear_condition, 'regime'] = 'bear'
            symbol_data.loc[bull_condition, 'regime'] = 'bull'

            # Actualizar en el DataFrame principal
            df.loc[symbol_mask, 'regime'] = symbol_data['regime']

            # Estadísticas por símbolo
            regime_counts = symbol_data['regime'].value_counts()
            total = len(symbol_data)

            print(f"   {symbol}:")
            for regime, count in regime_counts.items():
                percentage = count / total * 100
                print(f"     {regime}: {count:,} ({percentage:.1f}%)")

        # Estadísticas generales
        overall_counts = df['regime'].value_counts()
        total_samples = len(df)

        print(f"\n📊 Overall regime distribution (improved algorithm):")
        for regime, count in overall_counts.items():
            percentage = count / total_samples * 100
            print(f"   {regime}: {count:,} ({percentage:.1f}%)")

        return df

    def classify_market_regimes(self, df):
        """
        Wrapper que usa el algoritmo mejorado por defecto
        """
        return self.classify_market_regimes_improved(df)

    def balance_regime_data(self, df, min_samples_per_regime=8000):
        """
        Balancea datos para tener igual representación de cada régimen

        Estrategia mejorada:
        - 33% Bull Market data
        - 33% Bear Market data
        - 33% Sideways Market data
        """
        print(f"⚖️ Balancing regime data...")

        regimes = df['regime'].value_counts()

        # Usar el mínimo entre regímenes disponibles y el target
        target_samples = min(regimes.min(), min_samples_per_regime)

        print(f"🎯 Target: {target_samples:,} samples per regime")

        balanced_dfs = []
        for regime in ['bull', 'bear', 'sideways']:
            regime_df = df[df['regime'] == regime]

            if len(regime_df) < target_samples:
                print(f"⚠️ {regime} has only {len(regime_df):,} samples (need {target_samples:,})")
                # Usar todos los disponibles
                balanced_dfs.append(regime_df)
                continue

            # Sampling estratificado por símbolo para diversidad
            balanced_regime = []
            symbols = df['symbol'].unique()
            samples_per_symbol = target_samples // len(symbols)
            remaining_samples = target_samples % len(symbols)

            for i, symbol in enumerate(symbols):
                symbol_data = regime_df[regime_df['symbol'] == symbol]
                symbol_target = samples_per_symbol + (1 if i < remaining_samples else 0)

                if len(symbol_data) >= symbol_target:
                    # Sampling temporal distribuido (no solo reciente)
                    sampled = symbol_data.sample(n=symbol_target, random_state=42)
                    balanced_regime.append(sampled)
                else:
                    # Usar todos los disponibles si es menor
                    balanced_regime.append(symbol_data)

            if balanced_regime:
                regime_balanced = pd.concat(balanced_regime)
                balanced_dfs.append(regime_balanced)
                print(f"✅ {regime}: {len(regime_balanced):,} samples")

        if not balanced_dfs:
            raise ValueError("No regime data available for balancing")

        balanced_data = pd.concat(balanced_dfs).sort_index()

        # Verificar balance final
        final_counts = balanced_data['regime'].value_counts()
        print(f"\n✅ Balanced dataset: {len(balanced_data):,} total samples")
        for regime, count in final_counts.items():
            percentage = count / len(balanced_data) * 100
            print(f"   {regime}: {count:,} ({percentage:.1f}%)")

        return balanced_data

    def detect_current_regime(self, recent_data):
        """
        Detecta el régimen actual del mercado usando el algoritmo mejorado
        """
        if len(recent_data) < 50:
            return 'sideways'  # Default conservador

        # Usar el mismo algoritmo que para clasificación histórica
        recent_df = recent_data.copy()
        recent_df['symbol'] = 'CURRENT'  # Símbolo temporal para el análisis

        classified_df = self.classify_market_regimes_improved(recent_df)

        # Obtener el régimen más reciente
        current_regime = classified_df['regime'].iloc[-1]

        # Calcular confianza basada en consistencia reciente
        recent_regimes = classified_df['regime'].tail(20)
        regime_consistency = (recent_regimes == current_regime).mean()

        return current_regime, {'confidence': regime_consistency}

    def save_balanced_data(self, df, filename='balanced_training_data.pkl'):
        """Guardar datos balanceados para entrenamiento"""
        try:
            with open(filename, 'wb') as f:
                pickle.dump(df, f)
            print(f"💾 Balanced data saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving data: {e}")

    def load_balanced_data(self, filename='balanced_training_data.pkl'):
        """Cargar datos balanceados previamente guardados"""
        try:
            with open(filename, 'rb') as f:
                df = pickle.load(f)
            print(f"📂 Balanced data loaded from {filename}")
            return df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None

def main():
    """Demo del clasificador de regímenes mejorado"""
    print("🚀 Starting Improved Market Regime Classifier Demo")

    # Crear clasificador
    classifier = MarketRegimeClassifier()

    # Descargar datos
    try:
        print("\n1️⃣ Downloading market data...")
        df = classifier.download_balanced_data(days=365)  # 1 año de datos

        print("\n2️⃣ Classifying market regimes with improved algorithm...")
        df_with_regimes = classifier.classify_market_regimes(df)

        print("\n3️⃣ Balancing regime data...")
        balanced_df = classifier.balance_regime_data(df_with_regimes)

        print("\n4️⃣ Saving balanced data...")
        classifier.save_balanced_data(balanced_df)

        print("\n✅ Improved regime classification completed successfully!")

        # Demo de detección en tiempo real
        print("\n5️⃣ Testing real-time regime detection...")
        recent_btc = balanced_df[balanced_df['symbol'] == 'BTCUSDT'].tail(100)
        current_regime, metrics = classifier.detect_current_regime(recent_btc)

        print(f"Current market regime: {current_regime}")
        print(f"Confidence: {metrics['confidence']:.3f}")

    except Exception as e:
        print(f"❌ Error in demo: {e}")

if __name__ == "__main__":
    main()
