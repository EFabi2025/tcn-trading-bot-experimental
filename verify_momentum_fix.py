#!/usr/bin/env python3
"""
✅ VERIFICACIÓN FINAL - MOMENTUM BAJISTA CORREGIDO
Confirmar que el TCN ahora detecta correctamente el momentum bajista
"""

import asyncio
import numpy as np
import pandas as pd
from binance.client import Client
import tensorflow as tf
import pickle
import os
from dotenv import load_dotenv
from datetime import datetime

# Cargar variables de entorno
load_dotenv()

class MomentumFixVerifier:
    """✅ Verificador de corrección del momentum bajista"""

    def __init__(self):
        self.client = Client(
            api_key=os.getenv('BINANCE_API_KEY'),
            api_secret=os.getenv('BINANCE_SECRET_KEY')
        )
        self.lookback_window = 48

    def get_current_market_data(self, symbol: str) -> pd.DataFrame:
        """📊 Obtener datos actuales del mercado"""

        print(f"📊 Obteniendo datos actuales de {symbol}...")

        # Obtener klines recientes
        klines = self.client.get_historical_klines(
            symbol,
            Client.KLINE_INTERVAL_1HOUR,
            "3 days ago UTC"
        )

        # Convertir a DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        # Convertir tipos
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    def create_features_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """📈 Crear features para predicción (igual que en entrenamiento)"""

        features = pd.DataFrame(index=df.index)

        # 1. RETORNOS MULTI-TIMEFRAME
        for period in [1, 2, 4, 6, 12, 24]:
            features[f'return_{period}h'] = df['close'].pct_change(period)
            features[f'return_{period}h_abs'] = np.abs(features[f'return_{period}h'])

        # 2. MOMENTUM ACUMULATIVO
        for window in [6, 12, 24, 48]:
            features[f'momentum_{window}h'] = (df['close'] / df['close'].shift(window) - 1)
            features[f'momentum_strength_{window}h'] = np.abs(features[f'momentum_{window}h'])

        # 3. MEDIAS MÓVILES Y POSICIÓN RELATIVA
        for period in [6, 12, 24, 48]:
            sma = df['close'].rolling(period).mean()
            features[f'sma_{period}h'] = sma
            features[f'price_vs_sma_{period}h'] = (df['close'] - sma) / sma
            features[f'sma_slope_{period}h'] = sma.pct_change(6)

        # 4. RSI MULTI-TIMEFRAME
        for period in [6, 14, 24]:
            features[f'rsi_{period}'] = self.calculate_rsi(df['close'], period)
            features[f'rsi_{period}_oversold'] = (features[f'rsi_{period}'] < 30).astype(int)
            features[f'rsi_{period}_overbought'] = (features[f'rsi_{period}'] > 70).astype(int)

        # 5. VOLATILIDAD DIRECCIONAL
        for window in [6, 12, 24]:
            returns = df['close'].pct_change()
            features[f'vol_{window}h'] = returns.rolling(window).std()

            upside_vol = returns[returns > 0].rolling(window).std()
            downside_vol = returns[returns < 0].rolling(window).std()
            features[f'downside_vol_{window}h'] = downside_vol.fillna(0)
            features[f'vol_skew_{window}h'] = (downside_vol - upside_vol).fillna(0)

        # 6. VOLUMEN Y MOMENTUM
        features['volume_sma_20'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_sma_20']
        features['volume_momentum'] = df['volume'].rolling(6).mean() / df['volume'].rolling(24).mean()

        # 7. BANDAS DE BOLLINGER
        bb_period = 20
        bb_middle = df['close'].rolling(bb_period).mean()
        bb_std = df['close'].rolling(bb_period).std()
        features['bb_upper'] = bb_middle + (bb_std * 2)
        features['bb_lower'] = bb_middle - (bb_std * 2)
        features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        features['bb_squeeze'] = (features['bb_upper'] - features['bb_lower']) / bb_middle

        # 8. MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        features['macd_normalized'] = features['macd'] / df['close']

        # 9. DETECCIÓN DE REGÍMENES
        vol_24h = df['close'].pct_change().rolling(24).std()
        momentum_24h = features['momentum_24h']

        features['regime_bearish'] = ((vol_24h > vol_24h.rolling(100).quantile(0.7)) &
                                     (momentum_24h < -0.02)).astype(int)
        features['regime_bullish'] = ((vol_24h > vol_24h.rolling(100).quantile(0.7)) &
                                     (momentum_24h > 0.02)).astype(int)
        features['regime_sideways'] = ((features['regime_bearish'] == 0) &
                                      (features['regime_bullish'] == 0)).astype(int)

        # 10. CONTEXTO TEMPORAL
        features['hour'] = df['timestamp'].dt.hour / 24.0
        features['day_of_week'] = df['timestamp'].dt.dayofweek / 7.0

        return features.fillna(method='ffill').fillna(0)

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """📊 Calcular RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def load_corrected_model(self, symbol: str):
        """🤖 Cargar modelo corregido"""

        # Mapeo de símbolos a directorios
        symbol_to_dir = {
            'BTCUSDT': 'models/btc',
            'ETHUSDT': 'models/eth',
            'BNBUSDT': 'models/bnb'
        }

        model_dir = symbol_to_dir.get(symbol)
        if not model_dir:
            raise ValueError(f"Símbolo {symbol} no soportado")

        # Cargar modelo
        model_path = f'{model_dir}/model.h5'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")

        model = tf.keras.models.load_model(model_path)

        # Cargar scaler
        scaler_path = f'{model_dir}/scaler.pkl'
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        # Cargar columnas de features
        features_path = f'{model_dir}/feature_columns.pkl'
        with open(features_path, 'rb') as f:
            feature_columns = pickle.load(f)

        return model, scaler, feature_columns

    def predict_with_corrected_model(self, symbol: str) -> dict:
        """🔮 Hacer predicción con modelo corregido"""

        print(f"\n🔮 PREDICCIÓN CON MODELO CORREGIDO - {symbol}")
        print("=" * 50)

        try:
            # Obtener datos actuales
            df = self.get_current_market_data(symbol)

            # Crear features
            features = self.create_features_for_prediction(df)

            # Cargar modelo corregido
            model, scaler, feature_columns = self.load_corrected_model(symbol)

            # Preparar datos para predicción
            features_for_prediction = features[feature_columns].iloc[-self.lookback_window:]
            features_scaled = scaler.transform(features_for_prediction)

            # Crear secuencia
            X = features_scaled.reshape(1, self.lookback_window, len(feature_columns))

            # Hacer predicción
            prediction = model.predict(X, verbose=0)[0]

            # Interpretar predicción
            class_names = ['SELL', 'HOLD', 'BUY']
            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class]

            # Analizar contexto actual
            current_price = df['close'].iloc[-1]
            price_24h_ago = df['close'].iloc[-24] if len(df) >= 24 else df['close'].iloc[0]
            change_24h = (current_price - price_24h_ago) / price_24h_ago * 100

            current_rsi = features['rsi_14'].iloc[-1]
            current_momentum_24h = features['momentum_24h'].iloc[-1] * 100
            current_regime = features['regime_bearish'].iloc[-1]

            result = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'current_price': current_price,
                'change_24h': change_24h,
                'predicted_signal': class_names[predicted_class],
                'confidence': confidence,
                'probabilities': {
                    'SELL': prediction[0],
                    'HOLD': prediction[1],
                    'BUY': prediction[2]
                },
                'market_context': {
                    'rsi_14': current_rsi,
                    'momentum_24h': current_momentum_24h,
                    'bearish_regime': bool(current_regime)
                }
            }

            # Mostrar resultados
            print(f"💰 Precio actual: ${current_price:,.2f}")
            print(f"📈 Cambio 24h: {change_24h:+.2f}%")
            print(f"📊 RSI: {current_rsi:.1f}")
            print(f"📊 Momentum 24h: {current_momentum_24h:+.2f}%")
            print(f"🔴 Régimen bajista: {'SÍ' if current_regime else 'NO'}")

            print(f"\n🤖 PREDICCIÓN DEL MODELO CORREGIDO:")
            print(f"  🎯 Señal: {class_names[predicted_class]}")
            print(f"  🎯 Confianza: {confidence:.1%}")
            print(f"  📊 Probabilidades:")
            for i, name in enumerate(class_names):
                print(f"    - {name}: {prediction[i]:.1%}")

            return result

        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            return None

    def analyze_momentum_detection(self, symbol: str, result: dict) -> str:
        """📊 Analizar si el modelo detecta correctamente el momentum"""

        if not result:
            return "❌ ERROR EN PREDICCIÓN"

        change_24h = result['change_24h']
        predicted_signal = result['predicted_signal']
        confidence = result['confidence']
        momentum_24h = result['market_context']['momentum_24h']
        bearish_regime = result['market_context']['bearish_regime']

        # Análisis de corrección
        analysis = []

        # 1. Momentum bajista fuerte
        if change_24h < -5:
            if predicted_signal == 'SELL':
                analysis.append("✅ CORRECTO: Detecta momentum bajista fuerte")
            elif predicted_signal == 'HOLD':
                analysis.append("⚠️ CONSERVADOR: HOLD en momentum bajista fuerte (aceptable)")
            else:
                analysis.append("❌ ERROR: BUY en momentum bajista fuerte")

        # 2. Momentum bajista moderado
        elif change_24h < -2:
            if predicted_signal in ['SELL', 'HOLD']:
                analysis.append("✅ CORRECTO: No compra en momentum bajista")
            else:
                analysis.append("❌ ERROR: BUY en momentum bajista")

        # 3. Régimen bajista
        if bearish_regime:
            if predicted_signal != 'BUY':
                analysis.append("✅ CORRECTO: No compra en régimen bajista")
            else:
                analysis.append("❌ ERROR: BUY en régimen bajista detectado")

        # 4. Confianza
        if confidence > 0.7:
            analysis.append(f"✅ CONFIANZA ALTA: {confidence:.1%}")
        elif confidence > 0.5:
            analysis.append(f"⚠️ CONFIANZA MEDIA: {confidence:.1%}")
        else:
            analysis.append(f"❌ CONFIANZA BAJA: {confidence:.1%}")

        return " | ".join(analysis)

async def main():
    """🚀 Función principal de verificación"""

    print("✅ VERIFICACIÓN FINAL - CORRECCIÓN DE MOMENTUM BAJISTA")
    print("=" * 70)
    print("Verificando que el TCN ahora detecta correctamente el momentum bajista")
    print("=" * 70)

    verifier = MomentumFixVerifier()

    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

    results = {}

    for symbol in symbols:
        try:
            # Hacer predicción con modelo corregido
            result = verifier.predict_with_corrected_model(symbol)
            results[symbol] = result

            if result:
                # Analizar corrección
                analysis = verifier.analyze_momentum_detection(symbol, result)
                print(f"\n📊 ANÁLISIS DE CORRECCIÓN:")
                print(f"  {analysis}")

            print("\n" + "="*70)

        except Exception as e:
            print(f"❌ Error verificando {symbol}: {e}")
            results[symbol] = None

    # Resumen final
    print(f"\n🎯 RESUMEN DE VERIFICACIÓN:")
    print("=" * 50)

    for symbol, result in results.items():
        if result:
            signal = result['predicted_signal']
            change_24h = result['change_24h']
            confidence = result['confidence']

            # Determinar si es correcto
            if change_24h < -2 and signal != 'BUY':
                status = "✅ CORRECTO"
            elif change_24h < -2 and signal == 'BUY':
                status = "❌ ERROR"
            else:
                status = "⚠️ NEUTRAL"

            print(f"{symbol}: {signal} ({confidence:.1%}) | {change_24h:+.1f}% | {status}")
        else:
            print(f"{symbol}: ❌ ERROR EN PREDICCIÓN")

    print(f"\n💡 CONCLUSIÓN:")
    print("Si ves ✅ CORRECTO para símbolos con momentum bajista,")
    print("entonces el problema ha sido RESUELTO exitosamente.")
    print("Si ves ❌ ERROR, necesitamos más ajustes.")

if __name__ == "__main__":
    asyncio.run(main())
