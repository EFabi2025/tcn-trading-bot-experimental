#!/usr/bin/env python3
"""
🚀 TCN ADAPTATIVE TRAINER V3 - OPTIMIZADO PARA TIMEFRAMES 1M
============================================================

Entrenador TCN V3 optimizado específicamente para timeframes de 1 minuto con:
- Arquitectura TCN optimizada para 1M
- Sistema de features centralizado mejorado
- Etiquetado adaptativo para alta frecuencia
- Configuración específica para trading de 1M
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import talib
import warnings
import pickle
import os
from collections import Counter
warnings.filterwarnings('ignore')

# Importar motor de features centralizado
from centralized_features_engine3 import CentralizedFeaturesEngine

# ===============================================================================
# CONFIGURACIÓN OPTIMIZADA PARA 1M
# ===============================================================================

class TCN1MConfig:
    """Configuración optimizada para modelos TCN en timeframes de 1m"""
    
    # PARÁMETROS TEMPORALES ESPECÍFICOS PARA 1M
    SEQUENCE_LENGTH = 120           # 2 horas de datos (120 minutos)
    PREDICTION_HORIZON = 15         # Predicción a 15 minutos
    OVERLAP_RATIO = 0.8            # 80% de solapamiento entre secuencias
    
    # ARQUITECTURA TCN OPTIMIZADA PARA 1M
    TCN_FILTERS = [64, 128, 256, 128, 64]  # Pirámide de filtros
    KERNEL_SIZE = 3                 # Kernel pequeño para capturar patrones rápidos
    DILATIONS = [1, 2, 4, 8, 16, 32]      # Dilataciones exponenciales
    DROPOUT_RATE = 0.3             # Regularización
    
    # CONFIGURACIÓN DE ENTRENAMIENTO PARA 1M
    BATCH_SIZE = 256               # Batch grande para estabilidad
    LEARNING_RATE = 0.001          # Learning rate conservador
    EPOCHS = 100                   # Épocas máximas
    EARLY_STOPPING_PATIENCE = 15  # Paciencia para early stopping
    
    # BALANCEADO DE CLASES PARA 1M
    CLASS_WEIGHTS = {0: 1.2, 1: 1.0, 2: 1.1}  # BUY, HOLD, SELL
    LABEL_SMOOTHING = 0.1          # Suavizado de etiquetas
    
    # FEATURE SETS OPTIMIZADOS PARA 1M
    FEATURE_SETS = {
        'tcn_definitivo_v3_enhanced': '62 features (54 base + 8 bajistas)',
        'fast_crypto': '25 features ultra-livianas para entrenamiento rápido',
        'ultra_momentum': '25 features ultra-momentum',
        'ultra_momentum_combined': '71 features (TCN V3 + Ultra momentum)'
    }

@tf.keras.utils.register_keras_serializable(package="BalancedLoss")
class BalancedTradingLoss(tf.keras.losses.Loss):
    """🎯 TradingRealityLoss BALANCEADA - Optimizada para 1M"""
    
    def __init__(self, config: dict = None, name: str = 'balanced_trading_loss',
                 reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.AUTO):
        super().__init__(name=name, reduction=reduction)
        self.config = config or {}
        
        # 🎯 PARÁMETROS BALANCEADOS PARA TRADING ACTIVO EN 1M
        self.false_positive_penalty = self.config.get('false_positive_penalty', 1.3)
        self.false_negative_penalty = self.config.get('false_negative_penalty', 1.2)
        self.volatility_weight = self.config.get('volatility_weight', True)
        self.transaction_cost_aware = self.config.get('transaction_cost_aware', True)
        self.asymmetric_penalties = self.config.get('asymmetric_penalties', True)
        
        # 🎯 NUEVOS PARÁMETROS PARA BALANCEAR ACTIVIDAD EN 1M
        self.opportunity_loss_penalty = self.config.get('opportunity_loss_penalty', 1.4)
        self.trade_frequency_incentive = self.config.get('trade_frequency_incentive', 0.95)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        
        # 🛡️ CONFIGURACIÓN DE RIESGO MODERADA PARA 1M
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)
        self.max_drawdown_penalty = self.config.get('max_drawdown_penalty', 1.8)
        
        print(f"🎯 BalancedTradingLoss inicializada - OPTIMIZADA PARA 1M")
        print(f"   📊 False Positive Penalty: {self.false_positive_penalty}x")
        print(f"   📊 False Negative Penalty: {self.false_negative_penalty}x")
        print(f"   💎 Opportunity Loss Penalty: {self.opportunity_loss_penalty}x")
        print(f"   ⚡ Trade Frequency Incentive: {self.trade_frequency_incentive}x")
        print(f"   🎯 Umbral de confianza: {self.confidence_threshold}")

    def call(self, y_true, y_pred):
        """🎯 Función de pérdida principal balanceada para 1M"""
        return self.balanced_trading_loss(y_true, y_pred)
    
    def balanced_trading_loss(self, y_true, y_pred):
        """🎯 Loss function balanceada para trading activo en 1M"""
        # Convertir a tensores si es necesario
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # 🎯 CROSS-ENTROPY BASE
        base_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
        
        # 🎯 PENALIZACIONES BALANCEADAS PARA TRADING EN 1M
        trading_penalty = self._calculate_balanced_trading_penalties(y_true, y_pred)
        
        # 🎯 VOLATILIDAD WEIGHTING MODERADO PARA 1M
        volatility_factor = self._calculate_moderate_volatility_weighting(y_true, y_pred)
        
        # 🎯 TRANSACTION COST AWARENESS LIGERO PARA 1M
        cost_factor = self._calculate_light_transaction_cost_factor(y_true, y_pred)
        
        # 🎯 INCENTIVO POR OPORTUNIDADES APROVECHADAS EN 1M
        opportunity_factor = self._calculate_opportunity_factor(y_true, y_pred)
        
        # 🎯 COMBINACIÓN FINAL BALANCEADA PARA 1M
        final_loss = base_loss * trading_penalty * volatility_factor * cost_factor * opportunity_factor
        
        return final_loss
    
    def _calculate_balanced_trading_penalties(self, y_true, y_pred):
        """🎯 Penalizaciones balanceadas - optimizadas para 1M"""
        # Obtener predicciones como clases
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        
        # 🎯 CONFIANZA DE PREDICCIÓN PARA MODULAR PENALIZACIONES
        confidence = tf.reduce_max(y_pred, axis=-1)
        high_confidence_mask = tf.greater(confidence, self.confidence_threshold)
        
        # 🎯 FALSE POSITIVE PENALTY (moderado y dependiente de confianza)
        false_positive_mask = tf.logical_and(
            tf.equal(y_true, 1),  # HOLD real
            tf.logical_or(tf.equal(y_pred_classes, 0), tf.equal(y_pred_classes, 2))  # Predicción BUY/SELL
        )
        
        # 🎯 FALSE NEGATIVE PENALTY (penalizar perder oportunidades claras en 1M)
        false_negative_mask = tf.logical_and(
            tf.logical_or(tf.equal(y_true, 0), tf.equal(y_true, 2)),  # BUY/SELL real
            tf.equal(y_pred_classes, 1)  # Predicción HOLD
        )
        
        # 🎯 APLICAR PENALIZACIONES BALANCEADAS PARA 1M
        penalty = tf.ones_like(y_true, dtype=tf.float32)
        
        # False Positive: Solo penalizar fuertemente si la confianza es alta
        fp_penalty = tf.where(high_confidence_mask, 
                             tf.ones_like(penalty) * self.false_positive_penalty,
                             tf.ones_like(penalty) * 1.1)
        penalty = tf.where(false_positive_mask, fp_penalty, penalty)
        
        # False Negative: Penalizar más las oportunidades perdidas con alta confianza
        fn_penalty = tf.where(high_confidence_mask,
                             tf.ones_like(penalty) * self.opportunity_loss_penalty,
                             tf.ones_like(penalty) * self.false_negative_penalty)
        penalty = tf.where(false_negative_mask, fn_penalty, penalty)
        
        return penalty
    
    def _calculate_moderate_volatility_weighting(self, y_true, y_pred):
        """⚡ Ponderar por volatilidad del mercado - OPTIMIZADO PARA 1M"""
        if not self.volatility_weight:
            return tf.ones_like(y_true, dtype=tf.float32)
        
        # 🎯 SIMULAR VOLATILIDAD BASADA EN CONFIANZA DE PREDICCIÓN
        confidence = tf.reduce_max(y_pred, axis=-1)
        
        # 🎯 VOLATILIDAD MODERADA - optimizada para 1M
        volatility_factor = 1.0 + (1.0 - confidence) * 0.2
        
        return volatility_factor
    
    def _calculate_light_transaction_cost_factor(self, y_true, y_pred):
        """💰 Factor de costos de transacción LIGERO PARA 1M"""
        if not self.transaction_cost_aware:
            return tf.ones_like(y_true, dtype=tf.float32)
        
        # 🎯 OBTENER PREDICCIONES
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        
        # 🎯 CALCULAR ACTIVIDAD DE TRADING
        is_trading = tf.logical_or(tf.equal(y_pred_classes, 0), tf.equal(y_pred_classes, 2))
        
        # 🎯 PENALIZACIÓN LIGERA POR TRADING (incentiva actividad moderada en 1M)
        trading_penalty = tf.where(is_trading, 
                                 tf.ones_like(y_true, dtype=tf.float32) * 1.05,
                                 tf.ones_like(y_true, dtype=tf.float32))
        
        return trading_penalty
    
    def _calculate_opportunity_factor(self, y_true, y_pred):
        """💎 Factor de oportunidades - incentiva trades acertados en 1M"""
        # Obtener predicciones como clases
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        
        # 🎯 IDENTIFICAR TRADES CORRECTOS
        correct_trades_mask = tf.logical_and(
            tf.logical_or(tf.equal(y_true, 0), tf.equal(y_true, 2)),  # Debería hacer trade
            tf.logical_or(tf.equal(y_pred_classes, 0), tf.equal(y_pred_classes, 2))  # Predijo trade
        )
        
        # 🎯 TRADES CORRECTOS CON MISMA DIRECCIÓN
        exact_match_mask = tf.equal(y_true, tf.cast(y_pred_classes, tf.float32))
        correct_direction_mask = tf.logical_and(correct_trades_mask, exact_match_mask)
        
        # 🎯 APLICAR INCENTIVO LIGERO A TRADES CORRECTOS
        opportunity_factor = tf.where(correct_direction_mask,
                                    tf.ones_like(y_true, dtype=tf.float32) * self.trade_frequency_incentive,
                                    tf.ones_like(y_true, dtype=tf.float32))
        
        return opportunity_factor

    def get_config(self):
        """📝 Configuración para serialización"""
        config = super().get_config()
        config.update({
            'config': self.config,
            'false_positive_penalty': self.false_positive_penalty,
            'false_negative_penalty': self.false_negative_penalty,
            'volatility_weight': self.volatility_weight,
            'transaction_cost_aware': self.transaction_cost_aware,
            'asymmetric_penalties': self.asymmetric_penalties,
            'opportunity_loss_penalty': self.opportunity_loss_penalty,
            'trade_frequency_incentive': self.trade_frequency_incentive,
            'confidence_threshold': self.confidence_threshold,
            'risk_free_rate': self.risk_free_rate,
            'max_drawdown_penalty': self.max_drawdown_penalty
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """🔄 Recrear desde configuración"""
        return cls(**config)

class AdaptiveTCNTrainerV3:
    """🎯 Entrenador TCN V3 optimizado para timeframes de 1M"""

    def __init__(self, config: TCN1MConfig = None):
        self.config = config or TCN1MConfig()
        self.pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "DOTUSDT"]
        
        # ✅ CONFIGURACIÓN OPTIMIZADA PARA 1M
        self.lookback_window = self.config.SEQUENCE_LENGTH
        self.prediction_horizon = self.config.PREDICTION_HORIZON
        self.features_engine = CentralizedFeaturesEngine()

        # ✅ CONFIGURACIÓN: Thresholds adaptativos y loss balanceada
        self.use_adaptive_thresholds = True
        self.use_balanced_loss = True
        
        # 🎯 CONFIGURACIÓN DE BALANCED TRADING LOSS PARA 1M
        self.loss_config = {
            'false_positive_penalty': 1.3,
            'false_negative_penalty': 1.2,
            'opportunity_loss_penalty': 1.4,
            'trade_frequency_incentive': 0.95,
            'confidence_threshold': 0.7,
            'volatility_weight': True,
            'transaction_cost_aware': True,
            'asymmetric_penalties': True,
            'max_drawdown_penalty': 1.8
        }
        
        # 🎯 THRESHOLDS ADAPTATIVOS PARA 1M
        self.fixed_thresholds = {
            'BTCUSDT': {
                'strong_sell': -0.004, 'weak_sell': -0.002,
                'weak_buy': 0.002, 'strong_buy': 0.004
            },
            'ETHUSDT': {
                'strong_sell': -0.0026, 'weak_sell': -0.0012,
                'weak_buy': 0.0013, 'strong_buy': 0.0027
            },
            'BNBUSDT': {
                'strong_sell': -0.0015, 'weak_sell': -0.0007,
                'weak_buy': 0.0007, 'strong_buy': 0.0015
            },
            'XRPUSDT': {
                'strong_sell': -0.0018, 'weak_sell': -0.0009,
                'weak_buy': 0.0009, 'strong_buy': 0.0018,
            },
            'DOTUSDT': {
                'strong_sell': -0.0018, 'weak_sell': -0.0009,
                'weak_buy': 0.0009, 'strong_buy': 0.0018,
            },
        }

    def calculate_adaptive_thresholds(self, df: pd.DataFrame, symbol: str) -> dict:
        """🎯 Calcular thresholds adaptativos basados en volatilidad ATR - OPTIMIZADO PARA 1M"""
        if not self.use_adaptive_thresholds:
            return self.fixed_thresholds[symbol]
        
        try:
            # Calcular ATR para volatilidad adaptativa
            high_prices = df['high'].values.astype(float)
            low_prices = df['low'].values.astype(float)
            close_prices = df['close'].values.astype(float)
            
            # ATR de 14 períodos
            atr_14 = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
            
            # Promedio de ATR reciente (últimas 50 velas)
            avg_atr = np.nanmean(atr_14[-50:]) if len(atr_14) > 50 else np.nanmean(atr_14)
            avg_price = np.mean(close_prices[-50:]) if len(close_prices) > 50 else np.mean(close_prices)
            
            # ATR como porcentaje del precio
            atr_percent = (avg_atr / avg_price) if avg_price > 0 else 0.02
            
            # 🎯 THRESHOLDS ADAPTATIVOS OPTIMIZADOS PARA 1M
            base_threshold = atr_percent * 0.6
            
            adaptive_thresholds = {
                'strong_sell': -base_threshold * 1.4,
                'weak_sell': -base_threshold * 0.7,
                'weak_buy': base_threshold * 0.7,
                'strong_buy': base_threshold * 1.4
            }
            
            print(f"🎯 {symbol}: ATR adaptativo {atr_percent:.4f} ({atr_percent*100:.2f}%)")
            print(f"   📊 Thresholds optimizados para 1M: Buy {adaptive_thresholds['strong_buy']:.4f}, Sell {adaptive_thresholds['strong_sell']:.4f}")
            
            return adaptive_thresholds
            
        except Exception as e:
            print(f"⚠️ Error calculando thresholds adaptativos para {symbol}: {e}")
            print(f"   🔄 Usando thresholds fijos como fallback")
            return self.fixed_thresholds[symbol]

    def create_balanced_labels(self, df: pd.DataFrame, features: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """🎯 Crear etiquetas balanceadas optimizadas para 1M"""

        print(f"🎯 Creando etiquetas balanceadas para 1M en {symbol}...")

        close_prices = df['close'].values
        
        # ✅ CAMBIO PRINCIPAL: Usar thresholds adaptativos balanceados
        thresholds = self.calculate_adaptive_thresholds(df, symbol)
        
        labels = []

        # 🔄 LÓGICA BALANCEADA OPTIMIZADA PARA 1M
        for i in range(len(close_prices) - self.prediction_horizon):
            current_price = close_prices[i]
            future_price = close_prices[i + self.prediction_horizon]

            # Calcular retorno futuro
            future_return = (future_price - current_price) / current_price

            # 🎯 LÓGICA BALANCEADA OPTIMIZADA PARA 1M
            if future_return <= thresholds['strong_sell']:
                label = 0  # SELL
            elif future_return <= thresholds['weak_sell']:
                # ✅ MODIFICADO: Zona gris más permisiva a trades en 1M
                try:
                    current_rsi = features['rsi_14'].iloc[i] if i < len(features) else 50
                    current_macd = features['macd_histogram'].iloc[i] if i < len(features) else 0
                except:
                    current_rsi = 50
                    current_macd = 0

                # ✅ UMBRALES MÁS PERMISIVOS PARA TRADES EN 1M
                if current_rsi > 55 or current_macd < -0.001:
                    label = 0  # SELL (confirmación técnica)
                else:
                    label = 1  # HOLD
            elif future_return >= thresholds['strong_buy']:
                label = 2  # BUY
            elif future_return >= thresholds['weak_buy']:
                # ✅ MODIFICADO: Zona gris más permisiva a trades en 1M
                try:
                    current_rsi = features['rsi_14'].iloc[i] if i < len(features) else 50
                    current_macd = features['macd_histogram'].iloc[i] if i < len(features) else 0
                except:
                    current_rsi = 50
                    current_macd = 0

                # ✅ UMBRALES MÁS PERMISIVOS PARA TRADES EN 1M
                if current_rsi < 45 or current_macd > 0.001:
                    label = 2  # BUY (confirmación técnica)
                else:
                    label = 1  # HOLD
            else:
                # ✅ ZONA NEUTRAL: Más agresiva en detectar momentum en 1M
                if i >= 3:
                    recent_momentum = (close_prices[i] - close_prices[i-3]) / close_prices[i-3]
                    # ✅ UMBRALES MÁS BAJOS PARA DETECTAR MOMENTUM EN 1M
                    if recent_momentum > 0.005:
                        label = 2  # BUY (momentum positivo)
                    elif recent_momentum < -0.005:
                        label = 0  # SELL (momentum negativo)
                    else:
                        label = 1  # HOLD
                else:
                    label = 1  # HOLD

            labels.append(label)

        # Agregar labels al DataFrame
        df_labeled = df.iloc[:-self.prediction_horizon].copy()
        df_labeled['label'] = labels

        # Verificar distribución
        label_counts = pd.Series(labels).value_counts().sort_index()
        total = len(labels)

        print("📊 Distribución de etiquetas balanceadas para 1M:")
        class_names = ['SELL', 'HOLD', 'BUY']
        for i, name in enumerate(class_names):
            count = label_counts.get(i, 0) or 0
            pct = (count / total * 100) if total > 0 and count is not None else 0
            print(f"   - {name}: {count} ({pct:.1f}%)")

        return df_labeled

    async def get_real_market_data(self, symbol: str, days: int = 10) -> pd.DataFrame:
        """📊 Obtener datos reales de mercado - OPTIMIZADO PARA 1M"""
        print(f"📊 Obteniendo {days} días de datos reales para {symbol} (1M)...")

        base_url = "https://api.binance.com"
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        async with aiohttp.ClientSession() as session:
            url = f"{base_url}/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': '1m',  # ✅ ESPECÍFICO PARA 1M
                'startTime': start_time,
                'endTime': end_time,
                'limit': 1000
            }

            all_data = []
            current_start = start_time

            while current_start < end_time:
                params['startTime'] = current_start

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if not data:
                            break
                        all_data.extend(data)
                        current_start = data[-1][6] + 1
                    else:
                        print(f"❌ Error API: {response.status}")
                        break

                await asyncio.sleep(0.1)

        # Convertir a DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        # Convertir tipos
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp').sort_index()

        print(f"✅ Obtenidos {len(df)} registros de {symbol} (1M)")
        return df

    def prepare_training_data(self, df: pd.DataFrame, features: pd.DataFrame) -> tuple:
        """🔧 Preparar datos para entrenamiento - OPTIMIZADO PARA 1M"""
        print("🔧 Preparando datos para entrenamiento (1M)...")

        features_aligned = features.iloc[:-self.prediction_horizon]
        feature_columns = [col for col in features_aligned.columns if features_aligned[col].dtype in ['float64', 'int64']]

        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features_aligned[feature_columns])

        X = []
        y = []

        # ✅ SECUENCIAS OPTIMIZADAS PARA 1M
        for i in range(self.lookback_window, len(features_scaled)):
            sequence = features_scaled[i-self.lookback_window:i]
            X.append(sequence)
            y.append(df['label'].iloc[i])

        X = np.array(X)
        y = np.array(y)

        print(f"✅ Datos preparados para 1M: X shape: {X.shape}, y shape: {y.shape}")

        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

        return X, y, scaler, feature_columns, class_weight_dict

    def create_optimized_tcn_model_v3(self, input_shape: tuple):
        """
        🎯 Modelo TCN V3 optimizado para timeframes de 1M
        """

        try:
            from tensorflow.keras.layers import (
                Input, Conv1D, Add, BatchNormalization, Dropout,
                Activation, GlobalAveragePooling1D, Dense, LayerNormalization
            )
            from tensorflow.keras.models import Model
        except ImportError:
            # Fallback para versiones diferentes de TensorFlow
            Input = tf.keras.layers.Input
            Conv1D = tf.keras.layers.Conv1D
            Add = tf.keras.layers.Add
            BatchNormalization = tf.keras.layers.BatchNormalization
            Dropout = tf.keras.layers.Dropout
            Activation = tf.keras.layers.Activation
            GlobalAveragePooling1D = tf.keras.layers.GlobalAveragePooling1D
            Dense = tf.keras.layers.Dense
            LayerNormalization = tf.keras.layers.LayerNormalization
            Model = tf.keras.Model

        def temporal_block(x, filters, dilation_rate, dropout_rate=0.3):
            """Bloque temporal optimizado para 1M"""
            prev_x = x
            
            # Primera convolución dilatada
            x = Conv1D(filters, self.config.KERNEL_SIZE, padding='causal', dilation_rate=dilation_rate)(x)
            x = LayerNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(dropout_rate)(x)

            # Segunda convolución dilatada
            x = Conv1D(filters, self.config.KERNEL_SIZE, padding='causal', dilation_rate=dilation_rate)(x)
            x = LayerNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(dropout_rate)(x)

            # Skip connection con ajuste dimensional
            if prev_x.shape[-1] != filters:
                prev_x = Conv1D(filters, 1, padding='same')(prev_x)

            return Add()([prev_x, x])

        inputs = Input(shape=input_shape)
        x = inputs

        # ✅ ARQUITECTURA TCN OPTIMIZADA PARA 1M
        for i, (filters, dilation) in enumerate(zip(self.config.TCN_FILTERS, self.config.DILATIONS)):
            x = temporal_block(x, filters=filters, dilation_rate=dilation, dropout_rate=self.config.DROPOUT_RATE)
            
            # Atención temporal cada 2 bloques
            if i % 2 == 1:
                # Mecanismo de atención temporal simplificado
                attention = Dense(x.shape[-1], activation='tanh')(x)
                attention = Dense(x.shape[-1], activation='softmax')(attention)
                x = tf.keras.layers.Multiply()([x, attention])

        # Global pooling para reducir dimensionalidad
        x = GlobalAveragePooling1D()(x)

        # Capas Dense optimizadas para 1M
        x = Dense(128, activation='relu')(x)
        x = Dropout(self.config.DROPOUT_RATE)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(self.config.DROPOUT_RATE)(x)
        outputs = Dense(3, activation='softmax')(x)  # 3 clases

        model = Model(inputs, outputs)
        
        # ✅ COMPILAR CON BALANCED TRADING LOSS OPTIMIZADA PARA 1M
        if self.use_balanced_loss:
            balanced_loss = BalancedTradingLoss(self.loss_config)
            model.compile(
                optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.config.LEARNING_RATE),
                loss=balanced_loss,
                metrics=['accuracy']
            )
            print(f"✅ Modelo TCN V3 optimizado para 1M con BalancedTradingLoss creado: {model.count_params():,} parámetros")
        else:
            model.compile(
                optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.config.LEARNING_RATE),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            print(f"✅ Modelo TCN V3 optimizado para 1M con loss estándar creado: {model.count_params():,} parámetros")

        return model

    async def train_optimized_model(self, symbol: str, feature_set: str = 'tcn_definitivo_v3_enhanced') -> bool:
        """🎯 Entrenar modelo V3 optimizado para 1M"""

        print(f"\n🎯 ENTRENANDO MODELO V3 OPTIMIZADO PARA 1M - {symbol}")
        print("=" * 70)
        print(f"🎯 Feature set: {feature_set}")
        print(f"🎯 BalancedTradingLoss: {'HABILITADA' if self.use_balanced_loss else 'DESHABILITADA'}")
        print(f"🎯 Secuencia: {self.config.SEQUENCE_LENGTH} minutos")
        print(f"🎯 Horizonte: {self.config.PREDICTION_HORIZON} minutos")
        print("=" * 70)

        try:
            # 1. Obtener datos
            df = await self.get_real_market_data(symbol, days=10)

            # 2. Calcular features optimizadas para 1M
            print(f"🔄 Calculando features optimizadas para 1M...")
            features = self.features_engine.calculate_features(df, feature_set=feature_set)

            if features.empty:
                print(f"❌ Error calculando features")
                return False

            # 3. Crear etiquetas balanceadas optimizadas para 1M
            df_labeled = self.create_balanced_labels(df, features, symbol)

            # 4. Preparar datos optimizados para 1M
            X, y, scaler, feature_columns, class_weights = self.prepare_training_data(df_labeled, features)

            # 5. Split temporal optimizado para 1M
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # 6. Crear y entrenar modelo V3 optimizado para 1M
            model = self.create_optimized_tcn_model_v3((X.shape[1], X.shape[2]))

            # ✅ CALLBACKS OPTIMIZADOS PARA 1M
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    patience=self.config.EARLY_STOPPING_PATIENCE, 
                    restore_best_weights=True,
                    monitor='val_loss'
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    patience=8, 
                    factor=0.5,
                    min_lr=1e-6,
                    monitor='val_loss'
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    f'models/optimized_1m_{symbol.lower()}/best_model.h5',
                    save_best_only=True,
                    monitor='val_accuracy'
                ),
                tf.keras.callbacks.CSVLogger(f'logs/optimized_1m_{symbol.lower()}_training.csv')
            ]

            print("🚀 Entrenando modelo V3 optimizado para 1M...")
            os.makedirs(f'models/optimized_1m_{symbol.lower()}', exist_ok=True)
            os.makedirs(f'logs', exist_ok=True)

            # ✅ ENTRENAMIENTO OPTIMIZADO PARA 1M
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=self.config.EPOCHS,
                batch_size=self.config.BATCH_SIZE,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )

            # 7. Evaluar modelo optimizado para 1M
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            print(f"✅ Resultados optimizados para 1M:")
            print(f"   Accuracy: {test_acc:.3f}")
            print(f"   Loss: {test_loss:.3f}")

            # 8. Calcular métricas adicionales manualmente
            y_pred = model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            
            # Métricas detalladas
            print(f"\n📊 MÉTRICAS DETALLADAS PARA 1M:")
            print(classification_report(y_test, y_pred_classes, target_names=['SELL', 'HOLD', 'BUY']))
            
            # Matriz de confusión
            cm = confusion_matrix(y_test, y_pred_classes)
            print(f"📊 MATRIZ DE CONFUSIÓN:")
            print(cm)

            # 9. Análisis de actividad de trading predicha para 1M
            # Calcular distribución de predicciones
            pred_counts = pd.Series(y_pred_classes).value_counts().sort_index()
            total_pred = len(y_pred_classes)
            
            print(f"\n📊 ANÁLISIS DE ACTIVIDAD DE TRADING PREDICHA (1M):")
            class_names = ['SELL', 'HOLD', 'BUY']
            total_trades_predicted = pred_counts.get(0, 0) + pred_counts.get(2, 0)
            trade_activity_pct = (total_trades_predicted / total_pred * 100) if total_pred > 0 else 0
            
            for i, name in enumerate(class_names):
                count = pred_counts.get(i, 0)
                pct = (count / total_pred * 100) if total_pred > 0 else 0
                print(f"   - {name}: {count} ({pct:.1f}%)")
            
            print(f"🎯 ACTIVIDAD DE TRADING TOTAL (1M): {trade_activity_pct:.1f}% (objetivo: 30-80%)")
            
            if trade_activity_pct < 20:
                print(f"⚠️  ADVERTENCIA: Actividad de trading muy baja ({trade_activity_pct:.1f}%)")
                print(f"   💡 Considera ajustar loss_config para más actividad")
            elif trade_activity_pct > 80:
                print(f"⚠️  ADVERTENCIA: Actividad de trading muy alta ({trade_activity_pct:.1f}%)")
                print(f"   💡 Considera ajustar loss_config para menos actividad")
            else:
                print(f"✅ ACTIVIDAD DE TRADING BALANCEADA (1M): {trade_activity_pct:.1f}%")

            # 9. Guardar componentes optimizados para 1M
            model.save(f'models/optimized_1m_{symbol.lower()}/model.h5')
            
            with open(f'models/optimized_1m_{symbol.lower()}/scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
                
            with open(f'models/optimized_1m_{symbol.lower()}/feature_columns.pkl', 'wb') as f:
                pickle.dump(feature_columns, f)
                
            # ✅ NUEVO: Guardar configuración completa para 1M
            config_data = {
                'loss_config': self.loss_config,
                'model_config': {
                    'sequence_length': self.config.SEQUENCE_LENGTH,
                    'prediction_horizon': self.config.PREDICTION_HORIZON,
                    'tcn_filters': self.config.TCN_FILTERS,
                    'dilations': self.config.DILATIONS,
                    'feature_set': feature_set
                }
            }
            
            with open(f'models/optimized_1m_{symbol.lower()}/config.pkl', 'wb') as f:
                pickle.dump(config_data, f)

            print(f"✅ Modelo V3 optimizado para 1M guardado en models/optimized_1m_{symbol.lower()}/")
            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            return False

async def main():
    """🎯 Entrenar modelos V3 optimizados para 1M"""
    
    print("🎯 ENTRENADOR TCN V3 - OPTIMIZADO PARA TIMEFRAMES 1M")
    print("=" * 70)
    print("🎯 Objetivo: Modelos TCN optimizados específicamente para trading en 1M")
    print("=" * 70)
    
    # ✅ CONFIGURACIÓN OPTIMIZADA PARA 1M
    config = TCN1MConfig()
    trainer = AdaptiveTCNTrainerV3(config)
    
    print(f"\n💡 CONFIGURACIÓN OPTIMIZADA PARA 1M:")
    print(f"   Secuencia: {config.SEQUENCE_LENGTH} minutos")
    print(f"   Horizonte: {config.PREDICTION_HORIZON} minutos")
    print(f"   Batch Size: {config.BATCH_SIZE}")
    print(f"   Learning Rate: {config.LEARNING_RATE}")
    print(f"   TCN Filters: {config.TCN_FILTERS}")
    print(f"   Dilations: {config.DILATIONS}")
    
    print(f"\n💡 CONFIGURACIÓN DE BALANCED TRADING LOSS:")
    print(f"   False Positive Penalty: {trainer.loss_config['false_positive_penalty']}x")
    print(f"   False Negative Penalty: {trainer.loss_config['false_negative_penalty']}x")
    print(f"   Opportunity Loss Penalty: {trainer.loss_config['opportunity_loss_penalty']}x")
    print(f"   Trade Frequency Incentive: {trainer.loss_config['trade_frequency_incentive']}x")
    
    # Entrenar modelos V3 optimizados para 1M
    print(f"\n🚀 ENTRENANDO MODELOS V3 OPTIMIZADOS PARA 1M...")
    
    results = {}
    feature_sets = ['tcn_definitivo_v3_enhanced', 'fast_crypto']
    
    for symbol in trainer.pairs:
        print(f"\n{'='*70}")
        print(f"🎯 ENTRENANDO {symbol} - OPTIMIZADO PARA 1M")
        print(f"{'='*70}")
        
        # Probar con feature set optimizado para 1M
        success = await trainer.train_optimized_model(symbol, feature_set='tcn_definitivo_v3_enhanced')
        results[symbol] = success
        
        if success:
            print(f"✅ {symbol}: MODELO V3 OPTIMIZADO PARA 1M ENTRENADO EXITOSAMENTE")
        else:
            print(f"❌ {symbol}: ERROR EN ENTRENAMIENTO")
    
    print(f"\n{'='*70}")
    print(f"🎯 RESUMEN FINAL DE ENTRENAMIENTO V3 OPTIMIZADO PARA 1M")
    print(f"{'='*70}")
    for symbol, success in results.items():
        status = "✅ ÉXITO" if success else "❌ FALLO"
        print(f"   {symbol}: {status}")
    
    successful = sum(results.values())
    print(f"\n🎯 Modelos V3 optimizados para 1M entrenados: {successful}/{len(results)}")
    print(f"📁 Modelos guardados en: models/optimized_1m_[symbol]/")
    
    if successful > 0:
        print(f"\n🚀 MODELOS V3 OPTIMIZADOS PARA 1M LISTOS PARA USAR:")
        print(f"   - Arquitectura TCN optimizada para 1M")
        print(f"   - Secuencias de {config.SEQUENCE_LENGTH} minutos")
        print(f"   - Predicción a {config.PREDICTION_HORIZON} minutos")
        print(f"   - TradingRealityLoss optimizada para 1M")
        print(f"   - Thresholds adaptativos para 1M")
        print(f"   - Configuración completa guardada")
    else:
        print(f"\n⚠️  NO SE ENTRENARON MODELOS EXITOSAMENTE")
        print(f"   Revisa los logs para identificar errores")

if __name__ == "__main__":
    asyncio.run(main())
