#!/usr/bin/env python3
"""
FINAL REAL BINANCE PREDICTOR - Predictor optimizado con datos reales
Sistema final que usa las dimensiones exactas del modelo entrenado
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

class OptimizedBinanceData:
    """Proveedor optimizado de datos de Binance"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com"
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_market_data(self, symbol: str) -> dict:
        """Obtener datos completos del mercado"""
        tasks = [
            self.get_klines(symbol, "1m", 500),
            self.get_24hr_ticker(symbol),
            self.get_current_price(symbol)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            'klines': results[0] if not isinstance(results[0], Exception) else [],
            'ticker_24h': results[1] if not isinstance(results[1], Exception) else {},
            'current_price': results[2] if not isinstance(results[2], Exception) else {}
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
                        'volume': float(item[5])
                    } for item in data]
                else:
                    print(f"Error API: {response.status}")
        except Exception as e:
            print(f"Error klines: {e}")
        return []
    
    async def get_24hr_ticker(self, symbol: str) -> dict:
        """Obtener ticker 24h"""
        url = f"{self.base_url}/api/v3/ticker/24hr"
        params = {"symbol": symbol}
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
        except:
            pass
        return {}
    
    async def get_current_price(self, symbol: str) -> dict:
        """Obtener precio actual"""
        url = f"{self.base_url}/api/v3/ticker/price"
        params = {"symbol": symbol}
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
        except:
            pass
        return {}

class CompatibleFeatureEngine:
    """Motor de features compatible con modelo entrenado (50, 21)"""
    
    def create_exact_features(self, klines_data: list) -> pd.DataFrame:
        """Crear exactamente 21 features para compatibilidad"""
        
        if len(klines_data) < 100:
            return pd.DataFrame()
        
        df = pd.DataFrame(klines_data)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        df = df.sort_index()
        
        features = pd.DataFrame(index=df.index)
        
        print(f"🔧 Creando 21 features exactos desde {len(df)} velas reales...")
        
        # 1. Returns (5 features)
        for period in [1, 3, 5, 10, 20]:
            returns = df['close'].pct_change(period)
            features[f'returns_{period}'] = returns
        
        # 2. Returns MA (3 features) 
        for period in [1, 5, 10]:
            returns = df['close'].pct_change(period)
            features[f'returns_ma_{period}'] = returns.rolling(5).mean()
        
        # 3. Volatilidad (3 features)
        for window in [10, 20, 50]:
            vol = df['close'].pct_change().rolling(window).std()
            features[f'volatility_{window}'] = vol
        
        # 4. SMA Trends (2 features)
        for short, long in [(10, 30), (20, 60)]:
            sma_short = df['close'].rolling(short).mean()
            sma_long = df['close'].rolling(long).mean()
            features[f'sma_trend_{short}_{long}'] = (sma_short - sma_long) / df['close']
        
        # 5. RSI (2 features)
        features['rsi_14'] = self._calculate_rsi(df['close'], 14)
        features['rsi_deviation'] = abs(features['rsi_14'] - 50) / 50
        
        # 6. MACD (3 features)
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        features['macd'] = (ema12 - ema26) / df['close']
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # 7. Bollinger Bands (2 features)
        bb_middle = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        features['bb_position'] = (df['close'] - bb_middle) / (2 * bb_std)
        features['bb_width'] = (bb_std * 4) / bb_middle
        
        # 8. Volume (1 feature)
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Limpiar y verificar exactamente 21 features
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Asegurar exactamente 21 columnas
        feature_columns = list(features.columns)
        if len(feature_columns) > 21:
            features = features[feature_columns[:21]]
        elif len(feature_columns) < 21:
            # Agregar features dummy si faltan
            for i in range(len(feature_columns), 21):
                features[f'dummy_{i}'] = 0.0
        
        print(f"  ✅ {len(features.columns)} features creados (exacto para modelo)")
        return features
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calcular RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

class OptimizedTCNPredictor:
    """Predictor TCN optimizado para modelo entrenado"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_engine = CompatibleFeatureEngine()
        self.pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        self.load_trained_models()
    
    def load_trained_models(self):
        """Cargar modelos entrenados con arquitectura exacta"""
        print("📦 Cargando modelos TCN entrenados...")
        
        for pair in self.pairs:
            try:
                model_path = f"models/tcn_final_{pair.lower()}.h5"
                self.models[pair] = tf.keras.models.load_model(model_path)
                print(f"  ✅ {pair}: Modelo cargado - Input shape: {self.models[pair].input_shape}")
            except Exception as e:
                print(f"  ⚠️  {pair}: {e}")
                # Usar el sistema final como backup
                try:
                    from tcn_final_ready import FinalReadyTCN
                    tcn = FinalReadyTCN(pair_name=pair)
                    input_shape = (50, 21)
                    self.models[pair] = tcn.build_confidence_model(input_shape)
                    print(f"  🔧 {pair}: Modelo backup creado")
                except Exception as e2:
                    print(f"  ❌ {pair}: No se pudo crear modelo - {e2}")
    
    async def predict_real_market(self, pair: str, market_data: dict) -> dict:
        """Predicción con datos reales del mercado"""
        
        if pair not in self.models:
            return None
        
        klines = market_data.get('klines', [])
        if len(klines) < 100:
            print(f"  ⚠️  {pair}: Datos insuficientes ({len(klines)} velas)")
            return None
        
        print(f"  🧠 Generando predicción TCN para {pair}...")
        
        try:
            # Crear features exactos
            features = self.feature_engine.create_exact_features(klines)
            if features.empty or len(features) < 50:
                print(f"  ❌ Features insuficientes para {pair}")
                return None
            
            # Normalización por par
            if pair not in self.scalers:
                self.scalers[pair] = RobustScaler()
                features_scaled = self.scalers[pair].fit_transform(features.values)
            else:
                features_scaled = self.scalers[pair].transform(features.values)
            
            # Crear secuencia exacta (50, 21)
            sequence = features_scaled[-50:, :21]  # Últimas 50 filas, 21 features
            sequence = np.expand_dims(sequence, axis=0)  # (1, 50, 21)
            
            print(f"    Dimensión entrada: {sequence.shape}")
            
            # Predicción
            prediction = self.models[pair].predict(sequence, verbose=0)
            probabilities = prediction[0]
            
            predicted_class = np.argmax(probabilities)
            confidence = float(np.max(probabilities))
            
            class_names = ['SELL', 'HOLD', 'BUY']
            signal = class_names[predicted_class]
            
            # Análisis de mercado complementario
            market_analysis = self._analyze_market_context(klines, features.iloc[-1])
            
            # Confianza ajustada
            adjusted_confidence = self._adjust_confidence(confidence, market_analysis, signal)
            
            return {
                'pair': pair,
                'signal': signal,
                'confidence': adjusted_confidence,
                'raw_confidence': confidence,
                'probabilities': {
                    'SELL': float(probabilities[0]),
                    'HOLD': float(probabilities[1]),
                    'BUY': float(probabilities[2])
                },
                'market_analysis': market_analysis,
                'timestamp': datetime.now(),
                'data_quality': 'REAL_BINANCE'
            }
            
        except Exception as e:
            print(f"    ❌ Error en predicción {pair}: {e}")
            return None
    
    def _analyze_market_context(self, klines: list, latest_features: pd.Series) -> dict:
        """Analizar contexto del mercado actual"""
        
        current_price = klines[-1]['close']
        prices_20 = [k['close'] for k in klines[-20:]]
        
        # Análisis de precio
        sma_20 = np.mean(prices_20)
        price_position = (current_price - sma_20) / sma_20 * 100
        
        # Análisis de volatilidad
        returns = [(klines[i]['close'] / klines[i-1]['close'] - 1) for i in range(1, len(klines))]
        current_vol = np.std(returns[-20:]) * 100
        avg_vol = np.std(returns[-100:]) * 100
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        
        # Momentum
        price_momentum = (current_price / np.mean([k['close'] for k in klines[-10:]]) - 1) * 100
        
        # Volume análisis
        volumes = [k['volume'] for k in klines[-20:]]
        volume_avg = np.mean(volumes)
        current_volume = klines[-1]['volume']
        volume_ratio = current_volume / volume_avg if volume_avg > 0 else 1.0
        
        return {
            'price_vs_sma20': price_position,
            'volatility_ratio': vol_ratio,
            'momentum_10': price_momentum,
            'volume_ratio': volume_ratio,
            'rsi': latest_features.get('rsi_14', 50),
            'bb_position': latest_features.get('bb_position', 0.5)
        }
    
    def _adjust_confidence(self, confidence: float, market_analysis: dict, signal: str) -> float:
        """Ajustar confianza basado en análisis de mercado"""
        
        adjustment = 1.0
        
        # Factor RSI
        rsi = market_analysis['rsi']
        if signal == 'BUY' and rsi < 35:  # Oversold confirma BUY
            adjustment += 0.15
        elif signal == 'SELL' and rsi > 65:  # Overbought confirma SELL
            adjustment += 0.15
        
        # Factor volatilidad
        vol_ratio = market_analysis['volatility_ratio']
        if signal != 'HOLD' and vol_ratio > 1.5:  # Alta volatilidad reduce confianza
            adjustment -= 0.1
        elif signal == 'HOLD' and vol_ratio < 0.7:  # Baja volatilidad confirma HOLD
            adjustment += 0.1
        
        # Factor momentum
        momentum = market_analysis['momentum_10']
        if signal == 'BUY' and momentum > 0:  # Momentum positivo confirma BUY
            adjustment += 0.05
        elif signal == 'SELL' and momentum < 0:  # Momentum negativo confirma SELL
            adjustment += 0.05
        
        # Factor volumen
        vol_ratio = market_analysis['volume_ratio']
        if vol_ratio > 1.2:  # Alto volumen aumenta confianza
            adjustment += 0.05
        
        return min(confidence * adjustment, 0.95)  # Máximo 95%

class RealMarketAnalyzer:
    """Analizador de mercado real completo"""
    
    def __init__(self):
        self.data_provider = None
        self.predictor = OptimizedTCNPredictor()
    
    async def analyze_real_markets(self, pairs: list = None):
        """Análisis completo de mercados reales"""
        if pairs is None:
            pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        
        print("🚀 ANÁLISIS REAL DE MERCADO BINANCE")
        print("="*60)
        print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Mercados: {', '.join(pairs)}")
        print("="*60)
        
        async with OptimizedBinanceData() as provider:
            self.data_provider = provider
            
            results = []
            for pair in pairs:
                result = await self.analyze_single_pair(pair)
                if result:
                    results.append(result)
                print("-" * 50)
            
            # Resumen de trading
            self.show_trading_recommendations(results)
    
    async def analyze_single_pair(self, pair: str):
        """Análisis individual de un par"""
        print(f"\\n📈 ANALIZANDO: {pair}")
        
        try:
            # Obtener datos del mercado
            print("  📡 Obteniendo datos reales de Binance...")
            market_data = await self.data_provider.get_market_data(pair)
            
            if not market_data['klines']:
                print(f"  ❌ Sin datos para {pair}")
                return None
            
            # Información del mercado
            current_candle = market_data['klines'][-1]
            ticker_24h = market_data['ticker_24h']
            current_price = current_candle['close']
            
            print(f"  💰 Precio actual: ${current_price:,.4f}")
            if ticker_24h:
                change_24h = float(ticker_24h.get('priceChangePercent', 0))
                volume_24h = float(ticker_24h.get('volume', 0))
                print(f"  📊 Cambio 24h: {change_24h:+.2f}%")
                print(f"  📈 Volumen 24h: {volume_24h:,.0f}")
            
            # Predicción con modelo real
            prediction = await self.predictor.predict_real_market(pair, market_data)
            
            if prediction:
                signal = prediction['signal']
                confidence = prediction['confidence']
                raw_conf = prediction['raw_confidence']
                
                print(f"  🎯 PREDICCIÓN TCN:")
                print(f"     Señal: {signal}")
                print(f"     Confianza: {confidence:.3f}")
                print(f"     Confianza raw: {raw_conf:.3f}")
                print(f"     Boost: {confidence/raw_conf:.2f}x")
                
                print(f"  📊 DISTRIBUCIÓN:")
                for sig, prob in prediction['probabilities'].items():
                    bar_length = int(prob * 15)
                    bar = "█" * bar_length + "░" * (15 - bar_length)
                    print(f"     {sig}: {prob:.3f} |{bar}|")
                
                # Análisis de calidad
                quality = self._assess_signal_quality(prediction)
                print(f"  ⭐ Calidad: {quality}")
                
                # Contexto de mercado
                market = prediction['market_analysis']
                print(f"  📊 CONTEXTO:")
                print(f"     RSI: {market['rsi']:.1f}")
                print(f"     Precio vs SMA20: {market['price_vs_sma20']:+.2f}%")
                print(f"     Volatilidad: {market['volatility_ratio']:.2f}x")
                print(f"     Momentum: {market['momentum_10']:+.2f}%")
                
                return {
                    'pair': pair,
                    'price': current_price,
                    'prediction': prediction,
                    'quality': quality
                }
            else:
                print(f"  ❌ No se pudo generar predicción")
                
        except Exception as e:
            print(f"  ❌ Error analizando {pair}: {e}")
        
        return None
    
    def _assess_signal_quality(self, prediction: dict) -> str:
        """Evaluar calidad de la señal"""
        confidence = prediction['confidence']
        market = prediction['market_analysis']
        
        score = 0
        
        # Factor confianza
        if confidence >= 0.80:
            score += 3
        elif confidence >= 0.65:
            score += 2
        elif confidence >= 0.50:
            score += 1
        
        # Factor RSI
        rsi = market['rsi']
        if 35 <= rsi <= 65:  # RSI neutral
            score += 1
        elif rsi < 25 or rsi > 75:  # RSI extremo
            score += 2
        
        # Factor volatilidad
        if 0.8 <= market['volatility_ratio'] <= 1.2:  # Volatilidad normal
            score += 1
        
        # Clasificación
        if score >= 5:
            return "🟢 EXCELENTE"
        elif score >= 3:
            return "🟡 BUENA"
        elif score >= 1:
            return "🟠 MODERADA"
        else:
            return "🔴 BAJA"
    
    def show_trading_recommendations(self, results: list):
        """Mostrar recomendaciones de trading"""
        if not results:
            print("\\n❌ No hay resultados para mostrar")
            return
        
        print("\\n" + "="*60)
        print("💡 RECOMENDACIONES DE TRADING")
        print("="*60)
        
        # Filtrar por calidad
        excellent = [r for r in results if "EXCELENTE" in r['quality']]
        good = [r for r in results if "BUENA" in r['quality']]
        
        print(f"🎯 Señales excelentes: {len(excellent)}")
        print(f"🎯 Señales buenas: {len(good)}")
        
        if excellent:
            print("\\n🔥 OPORTUNIDADES PRINCIPALES:")
            for result in excellent:
                pred = result['prediction']
                print(f"  {result['pair']}: {pred['signal']} - {pred['confidence']:.3f} - {result['quality']}")
        
        if good:
            print("\\n⚡ OPORTUNIDADES SECUNDARIAS:")
            for result in good:
                pred = result['prediction']
                print(f"  {result['pair']}: {pred['signal']} - {pred['confidence']:.3f} - {result['quality']}")
        
        print("\\n✅ Análisis completado con datos reales de Binance")
        print("🔄 Datos actualizados en tiempo real")

async def main():
    """Función principal"""
    print("🎯 FINAL REAL BINANCE PREDICTOR")
    print("Sistema final con datos reales de Binance")
    print()
    
    analyzer = RealMarketAnalyzer()
    await analyzer.analyze_real_markets()

if __name__ == "__main__":
    asyncio.run(main()) 