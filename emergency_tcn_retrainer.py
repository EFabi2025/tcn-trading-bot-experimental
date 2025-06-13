#!/usr/bin/env python3
"""
🚨 EMERGENCY TCN RETRAINER
Corregir el problema de momentum bajista del TCN
"""

import asyncio
import numpy as np
import pandas as pd
from binance.client import Client
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
from dotenv import load_dotenv
import pickle
from datetime import datetime, timedelta

# Cargar variables de entorno
load_dotenv()

class EmergencyTCNRetrainer:
    """🚨 Reentrenador de emergencia para TCN"""

    def __init__(self):
        self.client = Client(
            api_key=os.getenv('BINANCE_API_KEY'),
            api_secret=os.getenv('BINANCE_SECRET_KEY')
        )
        self.lookback_window = 48  # 48 horas de historia
        self.prediction_horizon = 6  # Predecir 6 horas adelante

        # UMBRALES CORREGIDOS para detectar momentum bajista
        self.corrected_thresholds = {
            'BTCUSDT': {
                'strong_sell': -0.03,  # -3% (era -0.8%)
                'weak_sell': -0.015,   # -1.5% (era -0.3%)
                'strong_buy': 0.03,    # +3% (era +0.8%)
                'weak_buy': 0.015,     # +1.5% (era +0.3%)
            },
            'ETHUSDT': {
                'strong_sell': -0.04,  # -4% (era -0.8%)
                'weak_sell': -0.02,    # -2% (era -0.3%)
                'strong_buy': 0.04,    # +4% (era +0.8%)
                'weak_buy': 0.02,      # +2% (era +0.3%)
            },
            'BNBUSDT': {
                'strong_sell': -0.035, # -3.5% (era -1.0%)
                'weak_sell': -0.018,   # -1.8% (era -0.4%)
                'strong_buy': 0.035,   # +3.5% (era +1.0%)
                'weak_buy': 0.018,     # +1.8% (era +0.4%)
            }
        }

    def get_real_market_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """📊 Obtener datos reales del mercado"""

        print(f"📊 Obteniendo datos reales de {symbol} para {days} días...")

        # Obtener klines de 1 hora
        klines = self.client.get_historical_klines(
            symbol,
            Client.KLINE_INTERVAL_1HOUR,
            f"{days} days ago UTC"
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

    def create_momentum_aware_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """📈 Crear features que detecten momentum bajista"""

        print("📈 Creando features conscientes del momentum...")

        features = pd.DataFrame(index=df.index)

        # 1. RETORNOS MULTI-TIMEFRAME (crítico para momentum)
        for period in [1, 2, 4, 6, 12, 24]:
            features[f'return_{period}h'] = df['close'].pct_change(period)
            features[f'return_{period}h_abs'] = np.abs(features[f'return_{period}h'])

        # 2. MOMENTUM ACUMULATIVO (detecta tendencias sostenidas)
        for window in [6, 12, 24, 48]:
            features[f'momentum_{window}h'] = (df['close'] / df['close'].shift(window) - 1)
            features[f'momentum_strength_{window}h'] = np.abs(features[f'momentum_{window}h'])

        # 3. MEDIAS MÓVILES Y POSICIÓN RELATIVA
        for period in [6, 12, 24, 48]:
            sma = df['close'].rolling(period).mean()
            features[f'sma_{period}h'] = sma
            features[f'price_vs_sma_{period}h'] = (df['close'] - sma) / sma
            features[f'sma_slope_{period}h'] = sma.pct_change(6)  # Pendiente de la SMA

        # 4. RSI MULTI-TIMEFRAME
        for period in [6, 14, 24]:
            features[f'rsi_{period}'] = self.calculate_rsi(df['close'], period)
            features[f'rsi_{period}_oversold'] = (features[f'rsi_{period}'] < 30).astype(int)
            features[f'rsi_{period}_overbought'] = (features[f'rsi_{period}'] > 70).astype(int)

        # 5. VOLATILIDAD DIRECCIONAL
        for window in [6, 12, 24]:
            returns = df['close'].pct_change()
            features[f'vol_{window}h'] = returns.rolling(window).std()

            # Volatilidad direccional (más volátil en bajadas)
            upside_vol = returns[returns > 0].rolling(window).std()
            downside_vol = returns[returns < 0].rolling(window).std()
            features[f'downside_vol_{window}h'] = downside_vol.fillna(0)
            features[f'vol_skew_{window}h'] = (downside_vol - upside_vol).fillna(0)

        # 6. VOLUMEN Y MOMENTUM
        features['volume_sma_20'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_sma_20']
        features['volume_momentum'] = df['volume'].rolling(6).mean() / df['volume'].rolling(24).mean()

        # 7. BANDAS DE BOLLINGER Y POSICIÓN
        bb_period = 20
        bb_middle = df['close'].rolling(bb_period).mean()
        bb_std = df['close'].rolling(bb_period).std()
        features['bb_upper'] = bb_middle + (bb_std * 2)
        features['bb_lower'] = bb_middle - (bb_std * 2)
        features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        features['bb_squeeze'] = (features['bb_upper'] - features['bb_lower']) / bb_middle

        # 8. MACD MEJORADO
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        features['macd_normalized'] = features['macd'] / df['close']

        # 9. DETECCIÓN DE REGÍMENES DE MERCADO
        # Volatilidad alta + momentum negativo = régimen bajista
        vol_24h = df['close'].pct_change().rolling(24).std()
        momentum_24h = features['momentum_24h']

        features['regime_bearish'] = ((vol_24h > vol_24h.rolling(100).quantile(0.7)) &
                                     (momentum_24h < -0.02)).astype(int)
        features['regime_bullish'] = ((vol_24h > vol_24h.rolling(100).quantile(0.7)) &
                                     (momentum_24h > 0.02)).astype(int)
        features['regime_sideways'] = ((features['regime_bearish'] == 0) &
                                      (features['regime_bullish'] == 0)).astype(int)

        # 10. FEATURES DE CONTEXTO TEMPORAL
        features['hour'] = df['timestamp'].dt.hour / 24.0
        features['day_of_week'] = df['timestamp'].dt.dayofweek / 7.0

        print(f"✅ Features creadas: {len(features.columns)} features")

        return features.fillna(method='ffill').fillna(0)

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """📊 Calcular RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def create_corrected_labels(self, df: pd.DataFrame, features: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """🎯 Crear etiquetas corregidas que detecten momentum bajista"""

        print(f"🎯 Creando etiquetas corregidas para {symbol}...")

        thresholds = self.corrected_thresholds[symbol]

        labels = []

        for i in range(len(df) - self.prediction_horizon):
            current_price = df['close'].iloc[i]
            future_price = df['close'].iloc[i + self.prediction_horizon]
            future_return = (future_price - current_price) / current_price

            # CONTEXTO ADICIONAL para mejorar clasificación
            current_momentum_24h = features[f'momentum_24h'].iloc[i] if f'momentum_24h' in features.columns else 0
            current_rsi = features['rsi_14'].iloc[i] if 'rsi_14' in features.columns else 50
            current_regime = features['regime_bearish'].iloc[i] if 'regime_bearish' in features.columns else 0

            # LÓGICA CORREGIDA - más sensible al momentum bajista
            if future_return <= thresholds['strong_sell']:
                # SELL fuerte
                label = 0
            elif future_return <= thresholds['weak_sell']:
                # SELL débil, pero considerar contexto
                if current_momentum_24h < -0.01 or current_rsi < 35 or current_regime == 1:
                    label = 0  # SELL (momentum bajista confirmado)
                else:
                    label = 1  # HOLD
            elif future_return >= thresholds['strong_buy']:
                # BUY fuerte, pero solo si no estamos en régimen bajista
                if current_regime == 1 and current_momentum_24h < -0.02:
                    label = 1  # HOLD (no comprar en caída fuerte)
                else:
                    label = 2  # BUY
            elif future_return >= thresholds['weak_buy']:
                # BUY débil, muy conservador en régimen bajista
                if current_regime == 1 or current_momentum_24h < -0.005:
                    label = 1  # HOLD (no comprar en momentum bajista)
                else:
                    label = 2  # BUY
            else:
                # HOLD por defecto
                label = 1

            labels.append(label)

        # Agregar labels al DataFrame
        df_labeled = df.iloc[:-self.prediction_horizon].copy()
        df_labeled['label'] = labels

        # Verificar distribución
        label_counts = pd.Series(labels).value_counts().sort_index()
        total = len(labels)

        print("📊 Distribución de etiquetas corregidas:")
        class_names = ['SELL', 'HOLD', 'BUY']
        for i, name in enumerate(class_names):
            count = label_counts.get(i, 0)
            pct = count / total * 100
            print(f"   - {name}: {count} ({pct:.1f}%)")

        return df_labeled

    def prepare_training_data(self, df: pd.DataFrame, features: pd.DataFrame) -> tuple:
        """🔧 Preparar datos para entrenamiento"""

        print("🔧 Preparando datos para entrenamiento...")

        # Alinear features con labels
        features_aligned = features.iloc[:-self.prediction_horizon]

        # Seleccionar features numéricas
        feature_columns = [col for col in features_aligned.columns if features_aligned[col].dtype in ['float64', 'int64']]

        # Normalizar features
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features_aligned[feature_columns])

        # Crear secuencias temporales
        X = []
        y = []

        for i in range(self.lookback_window, len(features_scaled)):
            # Secuencia de features
            sequence = features_scaled[i-self.lookback_window:i]
            X.append(sequence)

            # Label correspondiente
            y.append(df['label'].iloc[i])

        X = np.array(X)
        y = np.array(y)

        print(f"✅ Datos preparados:")
        print(f"   - X shape: {X.shape}")
        print(f"   - y shape: {y.shape}")
        print(f"   - Features utilizadas: {len(feature_columns)}")

        return X, y, scaler, feature_columns

    def create_emergency_tcn_model(self, input_shape: tuple) -> tf.keras.Model:
        """🧠 Crear modelo TCN de emergencia optimizado para momentum"""

        print("🧠 Creando modelo TCN de emergencia...")

        model = tf.keras.Sequential([
            # Input
            tf.keras.layers.Input(shape=input_shape),

            # Normalización de entrada
            tf.keras.layers.LayerNormalization(),

            # TCN Layers - optimizadas para detectar momentum
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Conv1D(filters=128, kernel_size=3, dilation_rate=2, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Conv1D(filters=256, kernel_size=3, dilation_rate=4, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Conv1D(filters=128, kernel_size=3, dilation_rate=8, padding='causal', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),

            # Global pooling
            tf.keras.layers.GlobalMaxPooling1D(),

            # Dense layers
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),

            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),

            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),

            # Output layer
            tf.keras.layers.Dense(3, activation='softmax')
        ])

        # Compilar con configuración optimizada
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"✅ Modelo creado: {model.count_params():,} parámetros")

        return model

    def retrain_symbol(self, symbol: str) -> bool:
        """🔄 Reentrenar modelo para un símbolo específico"""

        print(f"\n🔄 REENTRENANDO MODELO PARA {symbol}")
        print("=" * 60)

        try:
            # 1. Obtener datos reales
            df = self.get_real_market_data(symbol, days=30)

            # 2. Crear features conscientes del momentum
            features = self.create_momentum_aware_features(df)

            # 3. Crear etiquetas corregidas
            df_labeled = self.create_corrected_labels(df, features, symbol)

            # 4. Preparar datos de entrenamiento
            X, y, scaler, feature_columns = self.prepare_training_data(df_labeled, features)

            # 5. Split de datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # 6. Crear y entrenar modelo
            model = self.create_emergency_tcn_model((X.shape[1], X.shape[2]))

            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
            ]

            print("🚀 Entrenando modelo...")
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=50,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )

            # 7. Evaluar modelo
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            print(f"✅ Precisión en test: {test_acc:.3f}")

            # 8. Guardar modelo y scaler
            os.makedirs(f'models/emergency_{symbol.lower()}', exist_ok=True)

            model.save(f'models/emergency_{symbol.lower()}/model.h5')

            with open(f'models/emergency_{symbol.lower()}/scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)

            with open(f'models/emergency_{symbol.lower()}/feature_columns.pkl', 'wb') as f:
                pickle.dump(feature_columns, f)

            print(f"✅ Modelo guardado en models/emergency_{symbol.lower()}/")

            return True

        except Exception as e:
            print(f"❌ Error reentrenando {symbol}: {e}")
            return False

async def main():
    """🚀 Función principal"""

    print("🚨 EMERGENCY TCN RETRAINER")
    print("Corrigiendo problema de momentum bajista")
    print("=" * 60)

    retrainer = EmergencyTCNRetrainer()

    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

    results = {}

    for symbol in symbols:
        print(f"\n🎯 Procesando {symbol}...")
        success = retrainer.retrain_symbol(symbol)
        results[symbol] = success

        if success:
            print(f"✅ {symbol}: Reentrenamiento exitoso")
        else:
            print(f"❌ {symbol}: Reentrenamiento fallido")

    print(f"\n🎯 RESUMEN FINAL:")
    print("=" * 40)

    successful = sum(results.values())
    total = len(results)

    print(f"✅ Exitosos: {successful}/{total}")

    if successful == total:
        print("\n🎉 ¡TODOS LOS MODELOS REENTRENADOS EXITOSAMENTE!")
        print("💡 Los nuevos modelos deberían detectar mejor el momentum bajista")
        print("💡 Umbrales corregidos para mayor sensibilidad")
        print("💡 Features mejoradas para detectar tendencias sostenidas")
    else:
        print(f"\n⚠️ {total - successful} modelos fallaron")
        print("💡 Revisa los errores y reintenta")

if __name__ == "__main__":
    asyncio.run(main())
