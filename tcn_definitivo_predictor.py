#!/usr/bin/env python3
"""
TCN DEFINITIVO PREDICTOR - Integración de Modelos al Sistema Principal
Predictor unificado que utiliza los 3 modelos definitivos entrenados
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TCNDefinitivoPredictor:
    """
    Predictor definitivo que integra los 3 modelos TCN entrenados
    - BTCUSDT: 59.7% accuracy, distribución balanceada
    - ETHUSDT: ~60% accuracy, distribución balanceada
    - BNBUSDT: 60.1% accuracy, distribución balanceada
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = {}
        self.class_weights = {}
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        self.model_stats = {
            'BTCUSDT': {'accuracy': 0.597, 'loss': 0.835},
            'ETHUSDT': {'accuracy': 0.600, 'loss': 0.840},  # Estimado
            'BNBUSDT': {'accuracy': 0.601, 'loss': 0.858}
        }

        # Thresholds específicos utilizados en entrenamiento
        self.thresholds = {
            'BTCUSDT': {'sell': -0.0014, 'buy': 0.0014},  # -0.14%/+0.14%
            'ETHUSDT': {'sell': -0.0026, 'buy': 0.0027},  # -0.26%/+0.27%
            'BNBUSDT': {'sell': -0.0015, 'buy': 0.0015}   # -0.15%/+0.15%
        }

        # ✅ CORREGIDO: Cargar modelos automáticamente al inicializar
        self.load_all_models()

    def load_all_models(self) -> bool:
        """Cargar todos los modelos definitivos"""
        logger.info("🔄 Cargando modelos definitivos...")

        success_count = 0
        for symbol in self.symbols:
            if self._load_model_for_symbol(symbol):
                success_count += 1
                logger.info(f"✅ {symbol}: Modelo cargado exitosamente")
            else:
                logger.error(f"❌ {symbol}: Error cargando modelo")

        if success_count == len(self.symbols):
            logger.info("🎉 Todos los modelos definitivos cargados correctamente")
            return True
        else:
            logger.warning(f"⚠️ Solo {success_count}/{len(self.symbols)} modelos cargados")
            return False

    def _load_model_for_symbol(self, symbol: str) -> bool:
        """Cargar modelo, scaler y features para un símbolo específico"""
        try:
            model_dir = f"models/definitivo_{symbol.lower()}"

            # Verificar que el directorio existe
            if not os.path.exists(model_dir):
                logger.error(f"Directorio no encontrado: {model_dir}")
                return False

            # Cargar modelo
            model_path = os.path.join(model_dir, "best_model.h5")
            if os.path.exists(model_path):
                self.models[symbol] = tf.keras.models.load_model(model_path)
                logger.info(f"  📊 Modelo cargado: {model_path}")
            else:
                logger.error(f"  ❌ Modelo no encontrado: {model_path}")
                return False

            # Cargar scaler
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scalers[symbol] = pickle.load(f)
                logger.info(f"  🔧 Scaler cargado: {scaler_path}")
            else:
                logger.error(f"  ❌ Scaler no encontrado: {scaler_path}")
                return False

            # Cargar feature columns
            features_path = os.path.join(model_dir, "feature_columns.pkl")
            if os.path.exists(features_path):
                with open(features_path, 'rb') as f:
                    self.feature_columns[symbol] = pickle.load(f)
                logger.info(f"  📋 Features cargadas: {len(self.feature_columns[symbol])} columnas")
            else:
                logger.error(f"  ❌ Features no encontradas: {features_path}")
                return False

            # Cargar class weights (opcional)
            weights_path = os.path.join(model_dir, "class_weights.pkl")
            if os.path.exists(weights_path):
                with open(weights_path, 'rb') as f:
                    self.class_weights[symbol] = pickle.load(f)
                logger.info(f"  ⚖️ Class weights cargados")

            return True

        except Exception as e:
            logger.error(f"Error cargando modelo {symbol}: {e}")
            return False

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crear las 66 features técnicos EXACTOS utilizados en entrenamiento
        Replica exactamente la lógica de tcn_definitivo_trainer.py con TA-Lib
        """
        try:
            import talib
        except ImportError:
            logger.error("TA-Lib no está instalado. Instalar con: pip install TA-Lib")
            return pd.DataFrame()

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
                logger.warning(f"Features creados: {len(features.columns)}, esperados: 66")
                # Ajustar si es necesario
                while len(features.columns) < 66:
                    features[f'padding_{len(features.columns)}'] = 0
                features = features.iloc[:, :66]  # Tomar solo las primeras 66

            return features

        except Exception as e:
            logger.error(f"Error creando features: {e}")
            return pd.DataFrame()

    def predict(self, symbol: str, market_data: pd.DataFrame) -> Optional[Dict]:
        """
        Realizar predicción para un símbolo específico

        Args:
            symbol: Símbolo a predecir (BTCUSDT, ETHUSDT, BNBUSDT)
            market_data: DataFrame con datos OHLCV

        Returns:
            Dict con predicción, confianza y detalles
        """
        if symbol not in self.models:
            logger.error(f"Modelo no cargado para {symbol}")
            return None

        try:
            # Crear features
            features = self.create_features(market_data)

            # Verificar que tenemos suficientes datos
            if len(features) < 48:  # Secuencia mínima requerida
                logger.warning(f"Datos insuficientes para {symbol}: {len(features)} < 48")
                return None

            # Seleccionar features utilizadas en entrenamiento
            feature_cols = self.feature_columns[symbol]
            features_selected = features[feature_cols].iloc[-48:]  # Últimas 48 observaciones

            # Normalizar con el scaler entrenado
            features_scaled = self.scalers[symbol].transform(features_selected)

            # Crear secuencia para el modelo
            sequence = features_scaled.reshape(1, 48, len(feature_cols))

            # Realizar predicción
            prediction = self.models[symbol].predict(sequence, verbose=0)
            probabilities = prediction[0]

            # Interpretar resultado
            predicted_class = np.argmax(probabilities)
            confidence = float(np.max(probabilities))

            class_names = ['SELL', 'HOLD', 'BUY']
            signal = class_names[predicted_class]

            # Información adicional
            current_price = float(market_data['close'].iloc[-1])
            model_accuracy = self.model_stats[symbol]['accuracy']

            result = {
                'symbol': symbol,
                'signal': signal,
                'confidence': confidence,
                'probabilities': {
                    'SELL': float(probabilities[0]),
                    'HOLD': float(probabilities[1]),
                    'BUY': float(probabilities[2])
                },
                'current_price': current_price,
                'model_accuracy': model_accuracy,
                'threshold_used': self.thresholds[symbol],
                'timestamp': datetime.now().isoformat(),
                'features_count': len(feature_cols)
            }

            logger.info(f"🎯 {symbol}: {signal} (conf: {confidence:.3f})")
            return result

        except Exception as e:
            logger.error(f"Error en predicción {symbol}: {e}")
            return None

    def predict_all_symbols(self, market_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Realizar predicciones para todos los símbolos

        Args:
            market_data_dict: Dict con datos de mercado por símbolo

        Returns:
            Dict con predicciones por símbolo
        """
        predictions = {}

        for symbol in self.symbols:
            if symbol in market_data_dict:
                prediction = self.predict(symbol, market_data_dict[symbol])
                if prediction:
                    predictions[symbol] = prediction
            else:
                logger.warning(f"Datos no disponibles para {symbol}")

        return predictions

    def predict_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Método de compatibilidad para integración con sistema principal
        Obtiene datos de Binance y realiza predicción
        """
        try:
            # Importar cliente de Binance
            import requests

            # Obtener datos de Binance
            url = f"https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': '5m',
                'limit': 100
            }

            response = requests.get(url, params=params)
            if response.status_code != 200:
                logger.error(f"Error obteniendo datos de Binance para {symbol}")
                return None

            klines = response.json()

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
            df.set_index('timestamp', inplace=True)

            # Realizar predicción
            return self.predict(symbol, df)

        except Exception as e:
            logger.error(f"Error en predict_symbol para {symbol}: {e}")
            return None

    def get_model_info(self) -> Dict:
        """Obtener información de los modelos cargados"""
        info = {
            'models_loaded': len(self.models),
            'symbols': list(self.models.keys()),
            'model_stats': self.model_stats,
            'thresholds': self.thresholds,
            'total_parameters': sum([model.count_params() for model in self.models.values()]),
            'load_timestamp': datetime.now().isoformat()
        }
        return info

# Función de utilidad para testing
def test_definitivo_predictor():
    """Test del predictor definitivo"""
    print("🧪 Testing TCN Definitivo Predictor...")

    predictor = TCNDefinitivoPredictor()

    # Cargar modelos
    if predictor.load_all_models():
        print("✅ Todos los modelos cargados correctamente")

        # Mostrar información
        info = predictor.get_model_info()
        print(f"📊 Modelos cargados: {info['models_loaded']}")
        print(f"🎯 Símbolos: {info['symbols']}")
        print(f"🧠 Parámetros totales: {info['total_parameters']:,}")

        # Generar datos de prueba
        print("\n🔄 Generando datos de prueba...")
        dates = pd.date_range(start='2024-01-01', periods=100, freq='5T')

        test_data = {}
        for symbol in predictor.symbols:
            # Simular datos OHLCV
            base_price = {'BTCUSDT': 45000, 'ETHUSDT': 3000, 'BNBUSDT': 400}[symbol]
            returns = np.random.normal(0, 0.01, 100)
            prices = base_price * np.exp(np.cumsum(returns))

            test_data[symbol] = pd.DataFrame({
                'open': prices * (1 + np.random.normal(0, 0.001, 100)),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.002, 100))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.002, 100))),
                'close': prices,
                'volume': np.random.lognormal(10, 0.5, 100)
            }, index=dates)

        # Realizar predicciones
        predictions = predictor.predict_all_symbols(test_data)

        print(f"\n🎯 Predicciones realizadas: {len(predictions)}")
        for symbol, pred in predictions.items():
            print(f"  {symbol}: {pred['signal']} (conf: {pred['confidence']:.3f})")

        return True
    else:
        print("❌ Error cargando modelos")
        return False

if __name__ == "__main__":
    test_definitivo_predictor()
