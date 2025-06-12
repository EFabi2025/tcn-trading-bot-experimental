"""
🧪 EDUCATIONAL ML Predictor - Trading Bot Experimental

Este módulo implementa un predictor educacional que:
- Integra el modelo TCN existente (1.1MB)
- Demuestra Feature Engineering para ML en trading
- Implementa predicciones con gestión de riesgo
- Incluye validación y logging educacional

⚠️ EXPERIMENTAL: Solo para fines educacionales
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
from decimal import Decimal
from datetime import datetime, timezone
from pathlib import Path

import structlog
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from ..interfaces.trading_interfaces import IMLPredictor
from ..schemas.trading_schemas import MarketDataSchema, TradingSignalSchema
from ..core.config import TradingBotSettings
from ..core.logging_config import TradingLogger

logger = structlog.get_logger(__name__)


class EducationalMLPredictor(IMLPredictor):
    """
    🎓 Predictor educacional de ML para trading

    Características educacionales:
    - Integra modelo TCN pre-entrenado (1.1MB)
    - Feature engineering educacional
    - Predicciones con confidence scoring
    - Manejo de errores didáctico
    """

    def __init__(self, settings: TradingBotSettings, trading_logger: TradingLogger):
        """
        Inicializa el predictor educacional

        Args:
            settings: Configuración del bot
            trading_logger: Logger estructurado para educación
        """
        self.settings = settings
        self.logger = trading_logger
        self.model: Optional[tf.keras.Model] = None
        self.scaler: Optional[MinMaxScaler] = None
        self.is_model_loaded = False

        # Configuración educacional
        self.sequence_length = 60  # 60 períodos para TCN
        self.features_count = 14   # Número de features del modelo
        self.confidence_threshold = 0.6  # Threshold educacional

        # Buffer para datos históricos
        self.price_buffer: List[Dict] = []
        self.max_buffer_size = 200  # Buffer educacional

        self._load_model_and_scaler()

    def _load_model_and_scaler(self) -> None:
        """Carga el modelo TCN y scaler para educación"""
        try:
            # Rutas educacionales
            model_path = Path("models/tcn_anti_bias_fixed.h5")
            scaler_path = Path("models/feature_scalers_fixed.pkl")

            # Verificar existencia de archivos
            if not model_path.exists():
                raise FileNotFoundError(f"🎓 Modelo no encontrado: {model_path}")

            if not scaler_path.exists():
                raise FileNotFoundError(f"🎓 Scaler no encontrado: {scaler_path}")

            # Cargar modelo TCN educacional
            self.model = tf.keras.models.load_model(str(model_path))

            # Cargar scaler educacional
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

            self.is_model_loaded = True

            # Logging educacional
            model_params = self.model.count_params() if self.model else 0
            self.logger.log_system_event(
                "educational_ml_model_loaded",
                model_path=str(model_path),
                model_parameters=model_params,
                sequence_length=self.sequence_length,
                features_count=self.features_count,
                educational_note="Modelo TCN cargado para experimentación"
            )

        except Exception as e:
            self.logger.log_error(
                "educational_ml_model_load_failed",
                error=str(e),
                educational_tip="Verificar que el modelo TCN esté entrenado"
            )
            raise

    async def predict(self, market_data: List[MarketDataSchema]) -> TradingSignalSchema:
        """
        🎓 EDUCATIONAL: Genera predicción usando modelo TCN

        Args:
            market_data: Lista de datos de mercado históricos

        Returns:
            Señal de trading educacional con confidence
        """
        if not self.is_model_loaded:
            raise ValueError("🎓 Modelo no cargado para predicción educacional")

        try:
            # Actualizar buffer educacional
            self._update_price_buffer(market_data)

            # Verificar datos suficientes
            if len(self.price_buffer) < self.sequence_length:
                return self._create_neutral_signal(
                    "Datos insuficientes para predicción educacional"
                )

            # Feature engineering educacional
            features = self._extract_features()

            # Preparar secuencia para TCN
            sequence = self._prepare_sequence(features)

            # Predicción educacional
            prediction = self._make_prediction(sequence)

            # Interpretar predicción
            signal = self._interpret_prediction(prediction, market_data[-1])

            self.logger.log_ml_prediction(
                signal.dict(),
                prediction_raw=float(prediction[0]),
                features_count=len(features),
                educational_note="Predicción TCN para educación"
            )

            return signal

        except Exception as e:
            self.logger.log_error(
                "educational_ml_prediction_failed",
                error=str(e),
                educational_tip="Error en predicción del modelo TCN"
            )
            # Retornar señal neutral en caso de error
            return self._create_neutral_signal("Error en predicción educacional")

    def _update_price_buffer(self, market_data: List[MarketDataSchema]) -> None:
        """Actualiza buffer de precios para educación"""
        for data in market_data:
            price_info = {
                'timestamp': data.timestamp,
                'open': float(data.price),  # Simplificado para educación
                'high': float(data.high_24h),
                'low': float(data.low_24h),
                'close': float(data.price),
                'volume': float(data.volume),
                'price_change': float(data.price_change_24h),
                'price_change_percent': float(data.price_change_percent_24h)
            }

            self.price_buffer.append(price_info)

            # Mantener tamaño del buffer
            if len(self.price_buffer) > self.max_buffer_size:
                self.price_buffer.pop(0)

    def _extract_features(self) -> np.ndarray:
        """
        🎓 Feature Engineering educacional para TCN

        Extrae 14 features técnicas del buffer de precios
        """
        if len(self.price_buffer) < self.sequence_length:
            raise ValueError("Buffer insuficiente para feature extraction")

        # Convertir a DataFrame para educación
        df = pd.DataFrame(self.price_buffer[-self.sequence_length:])

        features = []

        for i in range(len(df)):
            row_features = []

            # 1. Precio normalizado
            row_features.append(df.iloc[i]['close'])

            # 2. Volumen normalizado
            row_features.append(df.iloc[i]['volume'])

            # 3. Price change percent
            row_features.append(df.iloc[i]['price_change_percent'])

            # 4-6. RSI educacional (simplificado)
            if i >= 14:
                close_prices = df.iloc[i-14:i+1]['close'].values
                rsi = self._calculate_rsi(close_prices)
                row_features.extend([rsi, rsi/100, (rsi-50)/50])
            else:
                row_features.extend([50.0, 0.5, 0.0])  # Valores neutros

            # 7-9. MACD educacional (simplificado)
            if i >= 26:
                close_prices = df.iloc[i-26:i+1]['close'].values
                macd, signal_line, histogram = self._calculate_macd(close_prices)
                row_features.extend([macd, signal_line, histogram])
            else:
                row_features.extend([0.0, 0.0, 0.0])

            # 10-11. Bollinger Bands educacional
            if i >= 20:
                close_prices = df.iloc[i-20:i+1]['close'].values
                bb_upper, bb_lower, bb_percent = self._calculate_bollinger(close_prices)
                row_features.extend([bb_percent, bb_upper - bb_lower])
            else:
                row_features.extend([0.5, 0.0])

            # 12-14. Volatilidad y momentum educacional
            if i >= 10:
                close_prices = df.iloc[i-10:i+1]['close'].values
                volatility = np.std(close_prices) / np.mean(close_prices)
                momentum = (close_prices[-1] - close_prices[0]) / close_prices[0]
                volume_ratio = df.iloc[i]['volume'] / df.iloc[max(0, i-10):i+1]['volume'].mean()
                row_features.extend([volatility, momentum, volume_ratio])
            else:
                row_features.extend([0.0, 0.0, 1.0])

            features.append(row_features)

        features_array = np.array(features)

        # Normalizar features con scaler educacional
        if self.scaler:
            features_array = self.scaler.transform(features_array)

        return features_array

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """RSI educacional simplificado"""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi)

    def _calculate_macd(self, prices: np.ndarray) -> Tuple[float, float, float]:
        """MACD educacional simplificado"""
        if len(prices) < 26:
            return 0.0, 0.0, 0.0

        # EMA simplificado para educación
        ema_12 = np.mean(prices[-12:])
        ema_26 = np.mean(prices[-26:])

        macd = ema_12 - ema_26
        signal_line = macd * 0.9  # Simplificado
        histogram = macd - signal_line

        return float(macd), float(signal_line), float(histogram)

    def _calculate_bollinger(self, prices: np.ndarray, period: int = 20) -> Tuple[float, float, float]:
        """Bollinger Bands educacional"""
        if len(prices) < period:
            return 0.0, 0.0, 0.5

        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])

        bb_upper = sma + (2 * std)
        bb_lower = sma - (2 * std)

        current_price = prices[-1]
        bb_percent = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5

        return float(bb_upper), float(bb_lower), float(bb_percent)

    def _prepare_sequence(self, features: np.ndarray) -> np.ndarray:
        """Prepara secuencia para el modelo TCN"""
        if len(features) < self.sequence_length:
            raise ValueError("Features insuficientes para secuencia TCN")

        # Tomar últimos sequence_length elementos
        sequence = features[-self.sequence_length:]

        # Reshape para TCN: (batch_size, sequence_length, features)
        sequence = sequence.reshape(1, self.sequence_length, self.features_count)

        return sequence

    def _make_prediction(self, sequence: np.ndarray) -> np.ndarray:
        """Realiza predicción con modelo TCN"""
        prediction = self.model.predict(sequence, verbose=0)
        return prediction

    def _interpret_prediction(self, prediction: np.ndarray, latest_data: MarketDataSchema) -> TradingSignalSchema:
        """
        🎓 Interpreta predicción del modelo para señal educacional

        Args:
            prediction: Output del modelo TCN
            latest_data: Últimos datos de mercado

        Returns:
            Señal de trading educacional
        """
        # Extraer valor de predicción
        pred_value = float(prediction[0][0])  # Assuming single output

        # Calcular confidence educacional
        confidence = min(abs(pred_value - 0.5) * 2, 1.0)

        # Determinar acción basada en predicción
        if pred_value > 0.6 and confidence > self.confidence_threshold:
            action = "BUY"
            strength = "STRONG" if confidence > 0.8 else "MEDIUM"
        elif pred_value < 0.4 and confidence > self.confidence_threshold:
            action = "SELL"
            strength = "STRONG" if confidence > 0.8 else "MEDIUM"
        else:
            action = "HOLD"
            strength = "WEAK"

        # Crear señal educacional
        signal = TradingSignalSchema(
            symbol=latest_data.symbol,
            action=action,
            confidence=Decimal(str(round(confidence, 3))),
            strength=strength,
            timestamp=datetime.now(timezone.utc),
            price=latest_data.price,
            reasoning=f"TCN prediction: {pred_value:.3f}, confidence: {confidence:.3f}",
            metadata={
                "model_type": "TCN",
                "prediction_value": pred_value,
                "confidence_threshold": self.confidence_threshold,
                "features_used": self.features_count,
                "educational_note": "Señal generada por modelo TCN experimental"
            }
        )

        return signal

    def _create_neutral_signal(self, reason: str) -> TradingSignalSchema:
        """Crea señal neutral educacional"""
        return TradingSignalSchema(
            symbol="BTCUSDT",  # Default educacional
            action="HOLD",
            confidence=Decimal('0.0'),
            strength="WEAK",
            timestamp=datetime.now(timezone.utc),
            price=Decimal('0.0'),
            reasoning=reason,
            metadata={
                "model_type": "TCN",
                "educational_note": "Señal neutral por condiciones insuficientes"
            }
        )

    async def retrain_model(self, historical_data: List[MarketDataSchema]) -> bool:
        """
        🎓 EDUCATIONAL: Simula reentrenamiento del modelo

        En modo educacional, solo registra la operación
        """
        self.logger.log_system_event(
            "educational_model_retrain_requested",
            data_points=len(historical_data),
            educational_note="Reentrenamiento simulado - solo para demostración"
        )

        # En modo educacional, simular reentrenamiento exitoso
        return True

    def get_model_performance(self) -> Dict[str, Any]:
        """🎓 Obtiene métricas de performance educacional"""
        if not self.is_model_loaded:
            return {"error": "Modelo no cargado"}

        return {
            "model_loaded": self.is_model_loaded,
            "model_parameters": self.model.count_params() if self.model else 0,
            "sequence_length": self.sequence_length,
            "features_count": self.features_count,
            "confidence_threshold": self.confidence_threshold,
            "buffer_size": len(self.price_buffer),
            "max_buffer_size": self.max_buffer_size,
            "educational_note": "Métricas del modelo TCN experimental"
        }

    async def close(self) -> None:
        """🎓 Cierra recursos del predictor educacional"""
        self.logger.log_system_event(
            "educational_ml_predictor_closed",
            educational_note="Predictor ML educacional cerrado"
        )

        # Limpiar buffer
        self.price_buffer.clear()

        # En TensorFlow, el modelo se gestiona automáticamente
        self.is_model_loaded = False
