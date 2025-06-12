#!/usr/bin/env python3
"""
ENHANCED REAL PREDICTOR - Predictor mejorado con datos reales optimizado
Sistema avanzado de análisis y predicción con datos reales de Binance
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

class AdvancedFeatureEngineer:
    """Ingeniero de features técnicos avanzados"""

    def __init__(self):
        self.scaler = RobustScaler()

    def create_advanced_features(self, klines_1m: list, klines_5m: list) -> pd.DataFrame:
        """Crear features técnicos avanzados"""

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
            # === MOMENTUM INDICATORS ===
            features['rsi_14'] = talib.RSI(close, timeperiod=14)
            features['rsi_21'] = talib.RSI(close, timeperiod=21)
            features['rsi_divergence'] = features['rsi_14'] - features['rsi_21']

            # CCI - Commodity Channel Index
            features['cci_14'] = talib.CCI(high, low, close, timeperiod=14)
            features['cci_20'] = talib.CCI(high, low, close, timeperiod=20)

            # Williams %R
            features['willr_14'] = talib.WILLR(high, low, close, timeperiod=14)

            # === TREND INDICATORS ===
            # MACD
            macd, signal, histogram = talib.MACD(close)
            features['macd'] = macd
            features['macd_signal'] = signal
            features['macd_histogram'] = histogram
            features['macd_strength'] = np.abs(histogram)

            # ADX - Average Directional Movement Index
            features['adx_14'] = talib.ADX(high, low, close, timeperiod=14)
            features['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
            features['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)

            # Parabolic SAR
            features['sar'] = talib.SAR(high, low)
            features['sar_signal'] = np.where(close > features['sar'], 1, -1)

            # === VOLATILITY INDICATORS ===
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
            features['bb_upper'] = bb_upper
            features['bb_middle'] = bb_middle
            features['bb_lower'] = bb_lower
            features['bb_width'] = (bb_upper - bb_lower) / bb_middle
            features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)

            # Average True Range
            features['atr_14'] = talib.ATR(high, low, close, timeperiod=14)
            features['atr_20'] = talib.ATR(high, low, close, timeperiod=20)

            # === VOLUME INDICATORS ===
            # On Balance Volume
            features['obv'] = talib.OBV(close, volume)
            features['obv_ema'] = talib.EMA(features['obv'].values, timeperiod=20)

            # Volume Rate of Change
            features['volume_roc'] = talib.ROC(volume, timeperiod=10)

            # === PATTERN RECOGNITION ===
            # Moving averages convergence/divergence
            for short, long in [(5, 15), (10, 30), (20, 50)]:
                sma_short = talib.SMA(close, timeperiod=short)
                sma_long = talib.SMA(close, timeperiod=long)
                features[f'ma_conv_{short}_{long}'] = (sma_short - sma_long) / close

            # === PRICE ACTION ===
            # Returns múltiples períodos
            for period in [1, 3, 5, 10, 15, 30]:
                features[f'returns_{period}'] = talib.ROC(close, timeperiod=period)

            # High-Low spread
            features['hl_spread'] = (high - low) / close
            features['hl_position'] = (close - low) / (high - low)

            # === MULTI-TIMEFRAME (usando 5m) ===
            if len(klines_5m) >= 50:
                df_5m = pd.DataFrame(klines_5m)
                close_5m = df_5m['close'].values

                # RSI en 5m
                rsi_5m = talib.RSI(close_5m, timeperiod=14)
                features['rsi_5m'] = np.repeat(rsi_5m, 5)[:len(features)]  # Interpolar

                # Trend 5m
                macd_5m, _, _ = talib.MACD(close_5m)
                features['macd_5m'] = np.repeat(macd_5m, 5)[:len(features)]

            # === MARKET REGIME DETECTION ===
            # Volatility regime
            returns = talib.ROC(close, timeperiod=1)
            vol_short = pd.Series(returns).rolling(20).std()
            vol_long = pd.Series(returns).rolling(60).std()
            features['vol_regime'] = vol_short / vol_long

            # Trend strength
            features['trend_strength'] = np.abs(features['adx_14']) / 100

            # Market state (trending vs sideways)
            features['market_state'] = np.where(
                (features['adx_14'] > 25) & (features['bb_width'] > 0.02), 1,  # Trending
                np.where(features['bb_width'] < 0.01, -1, 0)  # Sideways
            )

        except Exception as e:
            print(f"  ⚠️  Error en algunos indicadores: {e}")

        # Limpiar datos
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # Normalizar features extremos
        for col in features.columns:
            if features[col].std() > 0:
                # Winsorización para outliers
                q95 = features[col].quantile(0.95)
                q05 = features[col].quantile(0.05)
                features[col] = features[col].clip(q05, q95)

        print(f"  ✅ {len(features.columns)} features técnicos creados")
        return features

    def select_best_features(self, features: pd.DataFrame, top_n: int = 21) -> pd.DataFrame:
        """Seleccionar las mejores features (21 para compatibilidad con modelos entrenados)"""
        if features.empty:
            return features

        # Features esenciales que siempre incluir (exactamente 21 como en el modelo original)
        essential_features = [
            'rsi_14', 'macd', 'bb_position', 'adx_14', 'atr_14',
            'returns_1', 'returns_5', 'vol_regime', 'trend_strength',
            'macd_signal', 'macd_histogram', 'bb_width', 'obv',
            'volume_roc', 'hl_spread', 'hl_position', 'plus_di',
            'minus_di', 'cci_14', 'willr_14', 'sar_signal'
        ]

        # Filtrar features existentes y tomar exactamente 21
        available_features = [f for f in essential_features if f in features.columns]

        # Si no tenemos suficientes, agregar las que falten
        if len(available_features) < 21:
            remaining_features = [f for f in features.columns if f not in available_features]
            available_features.extend(remaining_features[:21 - len(available_features)])

        # Tomar exactamente 21 features
        selected_features = available_features[:21]
        selected_df = features[selected_features].copy()

        print(f"  🎯 Seleccionadas {len(selected_df.columns)} features principales")
        return selected_df

class EnhancedTCNPredictor:
    """Predictor TCN mejorado"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_engineer = AdvancedFeatureEngineer()
        self.pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        self.load_models()

    def load_models(self):
        """Cargar modelos entrenados"""
        print("📦 Cargando modelos TCN optimizados...")

        for pair in self.pairs:
            try:
                # Intentar cargar modelo entrenado
                model_path = f"models/tcn_final_{pair.lower()}.h5"
                self.models[pair] = tf.keras.models.load_model(model_path)
                print(f"  ✅ {pair}: Modelo final cargado")
            except Exception as e:
                print(f"  ⚠️  {pair}: Creando modelo optimizado")
                self.models[pair] = self._create_enhanced_model()

    def _create_enhanced_model(self):
        """Crear modelo mejorado optimizado (compatible con modelos entrenados)"""
        model = tf.keras.Sequential([
            # Entrada compatible con modelos entrenados
            tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(50, 21)),
            tf.keras.layers.Dropout(0.2),

            # Capas de procesamiento
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.3),

            # Capas densas con regularización
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),

            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),

            # Salida con alta confianza
            tf.keras.layers.Dense(3, activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    async def predict_enhanced(self, pair: str, market_data: dict) -> dict:
        """Predicción mejorada con análisis completo"""

        klines_1m = market_data.get('klines_1m', [])
        klines_5m = market_data.get('klines_5m', [])

        if len(klines_1m) < 200:
            return None

        print(f"  🧠 Análisis TCN avanzado para {pair}...")

        # Crear features avanzados
        features = self.feature_engineer.create_advanced_features(klines_1m, klines_5m)
        if features.empty:
            return None

        # Seleccionar mejores features (21 para compatibilidad)
        features = self.feature_engineer.select_best_features(features, top_n=21)

        # Normalización avanzada
        if pair not in self.scalers:
            self.scalers[pair] = RobustScaler()
            features_scaled = self.scalers[pair].fit_transform(features.values)
        else:
            features_scaled = self.scalers[pair].transform(features.values)

        # Crear secuencia compatible con modelos entrenados (50 timesteps, 21 features)
        sequence_length = 50
        if len(features_scaled) < sequence_length:
            return None

        sequence = features_scaled[-sequence_length:, :21]  # Exactamente 21 features
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

class EnhancedMarketAnalyzer:
    """Analizador de mercado mejorado"""

    def __init__(self):
        self.data_provider = None
        self.predictor = EnhancedTCNPredictor()

    async def comprehensive_analysis(self, pairs: list = None):
        """Análisis completo del mercado"""
        if pairs is None:
            pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

        print("🚀 ANÁLISIS COMPLETO CON DATOS REALES")
        print("="*60)
        print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Mercados: {', '.join(pairs)}")
        print("="*60)

        async with AdvancedBinanceData() as provider:
            self.data_provider = provider

            results = []
            for pair in pairs:
                result = await self.analyze_enhanced_pair(pair)
                if result:
                    results.append(result)
                print("-" * 50)

            # Resumen final
            await self.show_trading_summary(results)

    async def analyze_enhanced_pair(self, pair: str):
        """Análisis completo de un par"""
        print(f"\\n📈 ANÁLISIS AVANZADO: {pair}")

        try:
            # Obtener datos completos
            print("  🔄 Recopilando datos del mercado...")
            market_data = await self.data_provider.get_comprehensive_data(pair)

            if not market_data['klines_1m']:
                print(f"  ❌ Sin datos para {pair}")
                return None

            # Información del mercado
            current_candle = market_data['klines_1m'][-1]
            current_price = current_candle['close']
            ticker_24h = market_data['ticker_24h']
            orderbook = market_data['orderbook']

            print(f"  💰 Precio: ${current_price:,.4f}")
            print(f"  📊 24h: {float(ticker_24h.get('priceChangePercent', 0)):+.2f}%")
            print(f"  📈 Volumen 24h: {float(ticker_24h.get('volume', 0)):,.0f}")
            if orderbook:
                spread = (float(orderbook.get('askPrice', 0)) - float(orderbook.get('bidPrice', 0))) / float(orderbook.get('bidPrice', 1)) * 100
                print(f"  🔄 Spread: {spread:.4f}%")

            # Predicción avanzada
            prediction = await self.predictor.predict_enhanced(pair, market_data)

            if prediction:
                print(f"  🎯 SEÑAL TCN: {prediction['signal']}")
                print(f"  🔥 Confianza: {prediction['confidence']:.3f}")
                print(f"  📊 Confianza raw: {prediction['raw_confidence']:.3f}")
                print(f"  ⚡ Boost técnico: {prediction['technical_boost']:.2f}x")

                print(f"  📊 Distribución:")
                for signal, prob in prediction['probabilities'].items():
                    bar = "█" * int(prob * 20)
                    print(f"     {signal}: {prob:.3f} {bar}")

                # Análisis de calidad de señal
                signal_quality = self._assess_signal_quality(prediction, market_data)
                print(f"  ⭐ Calidad señal: {signal_quality}")

                return {
                    'pair': pair,
                    'price': current_price,
                    'prediction': prediction,
                    'quality': signal_quality,
                    'market_data': market_data
                }
            else:
                print(f"  ❌ No se pudo generar predicción")

        except Exception as e:
            print(f"  ❌ Error en análisis: {e}")

        return None

    def _assess_signal_quality(self, prediction: dict, market_data: dict) -> str:
        """Evaluar calidad de la señal"""
        confidence = prediction['confidence']
        technical_boost = prediction['technical_boost']

        # Factores de calidad
        quality_score = 0

        # Factor confianza
        if confidence >= 0.80:
            quality_score += 3
        elif confidence >= 0.65:
            quality_score += 2
        elif confidence >= 0.50:
            quality_score += 1

        # Factor boost técnico
        if technical_boost > 1.2:
            quality_score += 2
        elif technical_boost > 1.1:
            quality_score += 1

        # Factor volumen
        ticker = market_data.get('ticker_24h', {})
        volume_change = float(ticker.get('count', 0))
        if volume_change > 100000:  # Alto volumen
            quality_score += 1

        # Clasificación
        if quality_score >= 5:
            return "🟢 EXCELENTE"
        elif quality_score >= 3:
            return "🟡 BUENA"
        elif quality_score >= 1:
            return "🟠 REGULAR"
        else:
            return "🔴 BAJA"

    async def show_trading_summary(self, results: list):
        """Mostrar resumen de trading"""
        if not results:
            return

        print("\\n" + "="*60)
        print("📊 RESUMEN DE TRADING")
        print("="*60)

        high_confidence = [r for r in results if r['prediction']['confidence'] >= 0.75]
        medium_confidence = [r for r in results if 0.60 <= r['prediction']['confidence'] < 0.75]

        print(f"🎯 Señales alta confianza (≥75%): {len(high_confidence)}")
        print(f"🎯 Señales media confianza (60-75%): {len(medium_confidence)}")

        if high_confidence:
            print("\\n🚀 OPORTUNIDADES PRINCIPALES:")
            for result in high_confidence:
                pred = result['prediction']
                print(f"  {result['pair']}: {pred['signal']} - {pred['confidence']:.3f} - {result['quality']}")

        print("\\n✅ Análisis completado con datos reales de Binance")

async def main():
    """Función principal"""
    print("🎯 ENHANCED REAL BINANCE PREDICTOR")
    print("Sistema avanzado de predicción con datos reales")
    print()

    analyzer = EnhancedMarketAnalyzer()
    await analyzer.comprehensive_analysis()

if __name__ == "__main__":
    # Verificar TA-Lib
    try:
        import talib
        asyncio.run(main())
    except ImportError:
        print("❌ TA-Lib no instalado. Instalando...")
        print("pip install TA-Lib")
        print("En macOS: brew install ta-lib && pip install TA-Lib")
        # Usar versión básica
        from real_binance_predictor import main as basic_main
        asyncio.run(basic_main())
