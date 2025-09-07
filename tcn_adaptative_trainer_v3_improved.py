#!/usr/bin/env python3
"""
🎯 TCN ADAPTATIVE TRAINER V3 IMPROVED - BALANCED TRADING LOSS OPTIMIZADA
Versión mejorada que corrige el sobretrading del V3 original
🔧 FIXED: Balance perfecto entre calidad y cantidad de trades

✅ NUEVA FUNCIONALIDAD: SELECCIÓN DE REGÍMENES DE MERCADO EQUILIBRADOS
- Detecta automáticamente mercados alcistas, bajistas y laterales
- Selecciona datos equilibrados de cada régimen para entrenamiento robusto
- Métodos: automático, manual y estratificado
- Visualización de distribución de regímenes

🎯 EJEMPLOS DE USO:

1. CONFIGURACIÓN INTERACTIVA CON REGÍMENES EQUILIBRADOS:
   python tcn_adaptative_trainer_v3_improved.py
   # Selecciona 's' en el paso 9 para habilitar regímenes equilibrados

2. CONFIGURACIÓN MANUAL:
   config = TrainingConfig()
   config.use_balanced_regimes = True
   config.regime_balance_method = 'manual'
   config.target_samples_per_regime = 500

⚖️ BENEFICIOS DE REGÍMENES EQUILIBRADOS:
- Modelos más robustos en cualquier condición de mercado
- Mejor generalización a diferentes entornos de trading
- Reducción de overfitting a un régimen específico
- Entrenamiento más equilibrado y representativo
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
import talib
import warnings
import pickle
import os
from collections import Counter
warnings.filterwarnings('ignore')

# Importar motor de features actual (sin cambios)
from centralized_features_engine3 import CentralizedFeaturesEngine

# ✅ NUEVA IMPORTACIÓN PARA DETECCIÓN DE REGÍMENES DE MERCADO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class TrainingConfig:
    """🔧 Configuración completa de entrenamiento V3 MEJORADO - TOTALMENTE CONFIGURABLE"""

    def __init__(self):
        # 📊 TIMEFRAMES DISPONIBLES
        self.available_timeframes = {
            '1m': '1m',
            '3m': '3m',
            '5m': '5m'
        }

        # 💎 PARES DISPONIBLES
        self.available_pairs = [
            'BTCUSDT', 'ETHUSDT', 'DOTUSDT', 'XRPUSDT',
            'BNBUSDT', 'ADAUSDT'
        ]

        # ⚙️ CONFIGURACIÓN POR DEFECTO
        self.timeframe = '1m'
        self.pairs = ['BTCUSDT']
        self.prediction_horizon = 6
        self.lookback_window = 24
        self.training_days = 30
        self.start_date = None  # Fecha específica opcional
        self.end_date = None    # Fecha específica opcional

        # 🎯 PARÁMETROS DE MODELO
        self.epochs = 50
        self.batch_size = 64
        self.use_adaptive_thresholds = True
        
        # 🎯 FEATURE SET
        self.feature_set = 'tcn_definitivo_v3'  # Por defecto

        # ✅ NUEVO: SELECCIÓN DE REGÍMENES DE MERCADO EQUILIBRADOS
        self.use_balanced_regimes = False  # Por defecto deshabilitado
        self.regime_balance_method = 'auto'  # 'auto', 'manual', 'stratified'
        self.target_samples_per_regime = None  # Para método manual

    def from_args(self, args):
        """Configurar desde argumentos de línea de comandos"""
        if args.timeframe:
            if args.timeframe in self.available_timeframes:
                self.timeframe = args.timeframe
            else:
                print(f"⚠️ Timeframe {args.timeframe} no válido. Usando {self.timeframe}")

        if args.pairs:
            valid_pairs = [p.upper() for p in args.pairs if p.upper() in self.available_pairs]
            if valid_pairs:
                self.pairs = valid_pairs
            else:
                print(f"⚠️ Ningún par válido encontrado. Usando {self.pairs}")

        if args.prediction_horizon:
            self.prediction_horizon = args.prediction_horizon

        if args.lookback_window:
            self.lookback_window = args.lookback_window

        if args.training_days:
            self.training_days = args.training_days

        if args.start_date:
            try:
                self.start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
            except ValueError:
                print(f"⚠️ Fecha de inicio inválida: {args.start_date}. Formato: YYYY-MM-DD")

        if args.end_date:
            try:
                self.end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
            except ValueError:
                print(f"⚠️ Fecha de fin inválida: {args.end_date}. Formato: YYYY-MM-DD")

        if hasattr(args, 'epochs') and args.epochs:
            self.epochs = args.epochs

        if hasattr(args, 'batch_size') and args.batch_size:
            self.batch_size = args.batch_size

        if hasattr(args, 'feature_set') and args.feature_set:
            self.feature_set = args.feature_set

    def print_config(self):
        """Mostrar configuración actual"""
        print("\n🔧 CONFIGURACIÓN DE ENTRENAMIENTO V3 MEJORADO:")
        print("=" * 60)
        print(f"⏰ Timeframe: {self.timeframe}")
        print(f"💎 Pares: {', '.join(self.pairs)}")
        print(f"🔮 Horizonte predicción: {self.prediction_horizon}")
        print(f"📊 Ventana lookback: {self.lookback_window}")
        if self.start_date and self.end_date:
            print(f"📅 Período: {self.start_date.strftime('%Y-%m-%d')} a {self.end_date.strftime('%Y-%m-%d')}")
        else:
            print(f"📅 Días entrenamiento: {self.training_days}")
        print(f"🎯 Épocas: {self.epochs}")
        print(f"📦 Batch size: {self.batch_size}")
        print(f"🎯 Feature Set: {self.feature_set}")
        print(f"🔧 Thresholds adaptativos: {'✅' if self.use_adaptive_thresholds else '❌'}")
        print(f"⚖️ Regímenes equilibrados: {'✅' if self.use_balanced_regimes else '❌'}")
        if self.use_balanced_regimes:
            print(f"   📊 Método: {self.regime_balance_method}")
            if self.target_samples_per_regime:
                print(f"   📊 Muestras objetivo por régimen: {self.target_samples_per_regime}")
        print(f"🚀 Loss Function: ImprovedTradingLoss (Anti-Sobretrading)")
        print("=" * 60)


# 🚀 CAPAS ESPECIALIZADAS PARA CRYPTO
class AttentionLayer(tf.keras.layers.Layer):
    """🎯 Capa de atención para features críticas de crypto (volumen, volatilidad)"""
    
    def __init__(self, attention_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.attention_dim = attention_dim
        
    def build(self, input_shape):
        self.query = tf.keras.layers.Dense(self.attention_dim)
        self.key = tf.keras.layers.Dense(self.attention_dim)
        self.value = tf.keras.layers.Dense(self.attention_dim)
        self.attention_weights = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs):
        # Calcular atención
        query = self.query(inputs)
        key = self.key(inputs)
        value = self.value(inputs)
        
        # Attention weights
        attention_scores = tf.matmul(query, key, transpose_b=True)
        attention_weights = tf.nn.softmax(attention_scores / tf.math.sqrt(tf.cast(self.attention_dim, tf.float32)), axis=-1)
        
        # Aplicar atención
        attended = tf.matmul(attention_weights, value)
        
        # Feature importance weighting
        feature_importance = self.attention_weights(inputs)
        weighted_output = attended * feature_importance
        
        # ✅ CORREGIDO: Solo hacer residual connection si las dimensiones coinciden
        if weighted_output.shape[-1] == inputs.shape[-1]:
            return weighted_output + inputs  # Residual connection
        else:
            return weighted_output  # Sin residual connection si dimensiones no coinciden


class GatingLayer(tf.keras.layers.Layer):
    """⚡ Capa de gating para regímenes de volatilidad"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        self.gate = tf.keras.layers.Dense(input_shape[-1], activation='sigmoid')
        
    def call(self, inputs):
        gate_values = self.gate(inputs)
        return inputs * gate_values


class MultiScaleTCNBlock(tf.keras.layers.Layer):
    """🔄 Bloque TCN multi-escala OPTIMIZADO para timeframes cortos (1m, 3m)"""
    
    def __init__(self, filters=32, kernel_size=3, dropout_rate=0.2, efficient_mode=False, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.efficient_mode = efficient_mode  # Para 1m y 3m
        
    def build(self, input_shape):
        # ✅ DILATIONS OPTIMIZADAS PARA EFICIENCIA
        if self.efficient_mode:
            # Para 1m y 3m: Máxima eficiencia
            self.short_term = [1, 2]         
            self.medium_term = [4]           # ✅ ULTRA REDUCIDO para 1m/3m
            self.long_term = [8]             # ✅ MÍNIMO para mantener contexto
        else:
            # Para 5m+: Configuración estándar optimizada
            self.short_term = [1, 2]         # ✅ REDUCIDO: [1,2,4]→[1,2]
            self.medium_term = [4, 8]        # ✅ REDUCIDO: [8,16,32]→[4,8]  
            self.long_term = [16, 32]        # ✅ REDUCIDO: [64,128,256]→[16,32]
        
        # Bloques TCN para cada escala (menos bloques)
        self.short_blocks = [self._create_tcn_block(d) for d in self.short_term]
        self.medium_blocks = [self._create_tcn_block(d) for d in self.medium_term]
        self.long_blocks = [self._create_tcn_block(d) for d in self.long_term]
        
        # Fusion de escalas optimizada
        fusion_units = self.filters // 2 if self.efficient_mode else self.filters
        self.scale_fusion = tf.keras.layers.Dense(fusion_units)
        
    def _create_tcn_block(self, dilation_rate):
        return tf.keras.Sequential([
            tf.keras.layers.Conv1D(self.filters, self.kernel_size, 
                                  padding='causal', dilation_rate=dilation_rate),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(self.dropout_rate)
        ])
        
    def call(self, inputs):
        # Procesar cada escala temporal
        short_features = tf.concat([block(inputs) for block in self.short_blocks], axis=-1)
        medium_features = tf.concat([block(inputs) for block in self.medium_blocks], axis=-1)
        long_features = tf.concat([block(inputs) for block in self.long_blocks], axis=-1)
        
        # Fusionar escalas
        multi_scale_features = tf.concat([short_features, medium_features, long_features], axis=-1)
        fused = self.scale_fusion(multi_scale_features)
        
        # ✅ CORREGIDO: Solo hacer residual connection si las dimensiones coinciden
        if fused.shape[-1] == inputs.shape[-1]:
            return fused + inputs  # Residual connection
        else:
            return fused  # Sin residual connection si dimensiones no coinciden


@tf.keras.utils.register_keras_serializable(package="ImprovedLoss")
class ImprovedTradingLoss(tf.keras.losses.Loss):
    """🎯 TradingRealityLoss MEJORADA - Balance perfecto calidad vs cantidad"""
    
    def __init__(self, config: dict = None, name: str = 'improved_trading_loss',
                 reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.AUTO):
        super().__init__(name=name, reduction=reduction)
        self.config = config or {}
        
        # 🎯 PARÁMETROS EQUILIBRADOS - MENOS CONSERVADORES
        self.false_positive_penalty = self.config.get('false_positive_penalty', 1.3)  # ✅ REDUCIDO: 1.7 → 1.3 (más equilibrado)
        self.false_negative_penalty = self.config.get('false_negative_penalty', 1.1)  # ✅ REDUCIDO: 1.3 → 1.1 (menos penalización)
        self.volatility_weight = self.config.get('volatility_weight', True)
        self.transaction_cost_aware = self.config.get('transaction_cost_aware', True)
        self.asymmetric_penalties = self.config.get('asymmetric_penalties', True)
        
        # 🎯 PARÁMETROS MÁS EQUILIBRADOS - NO TAN CONSERVADORES
        self.opportunity_loss_penalty = self.config.get('opportunity_loss_penalty', 1.2)  # ✅ REDUCIDO: 1.5 → 1.2 (más oportunidades)
        self.trade_frequency_incentive = self.config.get('trade_frequency_incentive', 0.95)  # ✅ MÁS INCENTIVO: 0.98 → 0.95 (más trades)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.65)  # ✅ MENOS ESTRICTO: 0.8 → 0.65 (más flexibilidad)
        self.quality_threshold = self.config.get('quality_threshold', 0.6)  # ✅ MENOS ESTRICTO: 0.75 → 0.6 (más trades)
        
        # 🛡️ CONFIGURACIÓN DE RIESGO MÁS EQUILIBRADA
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)
        self.max_drawdown_penalty = self.config.get('max_drawdown_penalty', 1.5)  # ✅ REDUCIDO: 2.0 → 1.5 (menos conservador)
        
        print(f"🎯 ImprovedTradingLoss inicializada - EQUILIBRIO CALIDAD/CANTIDAD")
        print(f"   📊 False Positive Penalty: {self.false_positive_penalty}x (equilibrado)")
        print(f"   📊 False Negative Penalty: {self.false_negative_penalty}x (más oportunidades)")
        print(f"   💎 Opportunity Loss Penalty: {self.opportunity_loss_penalty}x (equilibrado)")
        print(f"   ⚡ Trade Frequency Incentive: {self.trade_frequency_incentive}x (más incentivo)")
        print(f"   🎯 Confidence Threshold: {self.confidence_threshold} (más flexible)")
        print(f"   💫 Quality Threshold: {self.quality_threshold} (más flexible)")

    def call(self, y_true, y_pred):
        """🎯 Función de pérdida principal mejorada"""
        return self.improved_trading_loss(y_true, y_pred)
    
    def improved_trading_loss(self, y_true, y_pred):
        """🎯 Loss function mejorada para trading de calidad"""
        # Convertir a tensores si es necesario
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # 🎯 CROSS-ENTROPY BASE
        base_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
        
        # 🎯 PENALIZACIONES MEJORADAS ANTI-SOBRETRADING
        trading_penalty = self._calculate_improved_trading_penalties(y_true, y_pred)
        
        # 🎯 FACTOR DE CALIDAD NUEVO
        quality_factor = self._calculate_quality_factor(y_true, y_pred)
        
        # 🎯 VOLATILIDAD WEIGHTING MODERADO
        volatility_factor = self._calculate_moderate_volatility_weighting(y_true, y_pred)
        
        # 🎯 TRANSACTION COST AWARENESS MÁS ESTRICTO
        cost_factor = self._calculate_strict_transaction_cost_factor(y_true, y_pred)
        
        # 🎯 INCENTIVO POR OPORTUNIDADES APROVECHADAS (MEJORADO)
        opportunity_factor = self._calculate_improved_opportunity_factor(y_true, y_pred)
        
        # 🎯 COMBINACIÓN FINAL MEJORADA
        final_loss = base_loss * trading_penalty * quality_factor * volatility_factor * cost_factor * opportunity_factor
        
        return final_loss
    
    def _calculate_improved_trading_penalties(self, y_true, y_pred):
        """🎯 Penalizaciones mejoradas anti-sobretrading"""
        # Obtener predicciones como clases
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        
        # 🎯 CONFIANZA Y CALIDAD PARA MODULAR PENALIZACIONES
        confidence = tf.reduce_max(y_pred, axis=-1)
        high_confidence_mask = tf.greater(confidence, self.confidence_threshold)
        quality_mask = tf.greater(confidence, self.quality_threshold)
        
        # 🎯 FALSE POSITIVE PENALTY (MÁS ESTRICTO CONTRA SOBRETRADING)
        false_positive_mask = tf.logical_and(
            tf.equal(y_true, 1),  # HOLD real
            tf.logical_or(tf.equal(y_pred_classes, 0), tf.equal(y_pred_classes, 2))  # Predicción BUY/SELL
        )
        
        # 🎯 FALSE NEGATIVE PENALTY (oportunidades perdidas)
        false_negative_mask = tf.logical_and(
            tf.logical_or(tf.equal(y_true, 0), tf.equal(y_true, 2)),  # BUY/SELL real
            tf.equal(y_pred_classes, 1)  # Predicción HOLD
        )
        
        # 🎯 APLICAR PENALIZACIONES MEJORADAS
        penalty = tf.ones_like(y_true, dtype=tf.float32)
        
        # False Positive: Penalización moderada (menos agresiva)
        fp_penalty = tf.where(quality_mask, 
                             tf.ones_like(penalty) * (self.false_positive_penalty * 1.1),  # ✅ REDUCIDO: 1.2 → 1.1
                             tf.where(high_confidence_mask,
                                     tf.ones_like(penalty) * self.false_positive_penalty,
                                     tf.ones_like(penalty) * 1.05))  # ✅ REDUCIDO: 1.2 → 1.05 (más suave)
        penalty = tf.where(false_positive_mask, fp_penalty, penalty)
        
        # False Negative: Penalización más suave para no perder oportunidades
        fn_penalty = tf.where(quality_mask,
                             tf.ones_like(penalty) * (self.opportunity_loss_penalty * 1.05),  # ✅ REDUCIDO: 1.1 → 1.05
                             tf.where(high_confidence_mask,
                                     tf.ones_like(penalty) * self.opportunity_loss_penalty,
                                     tf.ones_like(penalty) * self.false_negative_penalty))
        penalty = tf.where(false_negative_mask, fn_penalty, penalty)
        
        return penalty
    
    def _calculate_quality_factor(self, y_true, y_pred):
        """💫 Factor de calidad EQUILIBRADO - menos restrictivo"""
        confidence = tf.reduce_max(y_pred, axis=-1)
        
        # ✅ FACTOR MÁS EQUILIBRADO - menos penalización por baja confianza
        quality_factor = tf.where(tf.greater(confidence, self.quality_threshold),
                                tf.ones_like(confidence) * 0.97,  # ✅ REDUCIDO: 0.95 → 0.97 (menos descuento)
                                tf.where(tf.greater(confidence, self.confidence_threshold),
                                        tf.ones_like(confidence) * 0.99,  # ✅ REDUCIDO: 0.98 → 0.99 (más suave)
                                        tf.ones_like(confidence) * 1.05))  # ✅ REDUCIDO: 1.1 → 1.05 (menos penalización)
        
        return quality_factor
    
    def _calculate_moderate_volatility_weighting(self, y_true, y_pred):
        """⚡ Ponderar por volatilidad del mercado - MÁS SUAVE"""
        if not self.volatility_weight:
            return tf.ones_like(y_true, dtype=tf.float32)
        
        confidence = tf.reduce_max(y_pred, axis=-1)
        volatility_factor = 1.0 + (1.0 - confidence) * 0.08  # ✅ REDUCIDO: 0.15 → 0.08 (mucho más suave)
        
        return volatility_factor
    
    def _calculate_strict_transaction_cost_factor(self, y_true, y_pred):
        """💰 Factor de costos de transacción EQUILIBRADO"""
        if not self.transaction_cost_aware:
            return tf.ones_like(y_true, dtype=tf.float32)
        
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        is_trading = tf.logical_or(tf.equal(y_pred_classes, 0), tf.equal(y_pred_classes, 2))
        
        # ✅ PENALIZACIÓN MÁS SUAVE POR TRADING (más equilibrado)
        trading_penalty = tf.where(is_trading, 
                                 tf.ones_like(y_true, dtype=tf.float32) * 1.03,  # ✅ REDUCIDO: 1.08 → 1.03 (mucho más suave)
                                 tf.ones_like(y_true, dtype=tf.float32))
        
        return trading_penalty
    
    def _calculate_improved_opportunity_factor(self, y_true, y_pred):
        """💎 MEJORADO: Factor de oportunidades con calidad"""
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        confidence = tf.reduce_max(y_pred, axis=-1)
        
        # 🎯 TRADES CORRECTOS DE ALTA CALIDAD
        correct_trades_mask = tf.logical_and(
            tf.logical_or(tf.equal(y_true, 0), tf.equal(y_true, 2)),
            tf.logical_or(tf.equal(y_pred_classes, 0), tf.equal(y_pred_classes, 2))
        )
        
        exact_match_mask = tf.equal(y_true, tf.cast(y_pred_classes, tf.float32))
        correct_direction_mask = tf.logical_and(correct_trades_mask, exact_match_mask)
        
        # ✅ INCENTIVO BASADO EN CALIDAD
        quality_mask = tf.greater(confidence, self.quality_threshold)
        high_quality_correct = tf.logical_and(correct_direction_mask, quality_mask)
        
        opportunity_factor = tf.where(high_quality_correct,
                                    tf.ones_like(y_true, dtype=tf.float32) * 0.96,  # ✅ REDUCIDO: 0.93 → 0.96 (menos incentivo extremo)
                                    tf.where(correct_direction_mask,
                                            tf.ones_like(y_true, dtype=tf.float32) * self.trade_frequency_incentive,
                                            tf.ones_like(y_true, dtype=tf.float32)))
        
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
            'quality_threshold': self.quality_threshold,
            'risk_free_rate': self.risk_free_rate,
            'max_drawdown_penalty': self.max_drawdown_penalty
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """🔄 Recrear desde configuración mejorada"""
        # Extraer configuración específica
        improved_config = config.pop('config', {})
        
        # ✅ PROCESAR PARÁMETROS INDIVIDUALES GUARDADOS POR get_config()
        improved_config.update({
            'false_positive_penalty': config.pop('false_positive_penalty', 1.7),
            'false_negative_penalty': config.pop('false_negative_penalty', 1.3),
            'volatility_weight': config.pop('volatility_weight', True),
            'transaction_cost_aware': config.pop('transaction_cost_aware', True),
            'asymmetric_penalties': config.pop('asymmetric_penalties', True),
            'opportunity_loss_penalty': config.pop('opportunity_loss_penalty', 1.5),
            'trade_frequency_incentive': config.pop('trade_frequency_incentive', 0.98),
            'confidence_threshold': config.pop('confidence_threshold', 0.8),
            'quality_threshold': config.pop('quality_threshold', 0.75),
            'risk_free_rate': config.pop('risk_free_rate', 0.02),
            'max_drawdown_penalty': config.pop('max_drawdown_penalty', 2.0)
        })
        
        # Pasar configuración procesada al constructor
        config['config'] = improved_config
        return cls(**config)


class MarketRegimeSelector:
    """📊 Selector inteligente de regímenes de mercado para entrenamiento equilibrado"""
    
    def __init__(self):
        self.regime_names = {
            0: 'BAJISTA',
            1: 'LATERAL', 
            2: 'ALCISTA'
        }
        self.regime_colors = {
            0: '#ff6b6b',  # Rojo para bajista
            1: '#4ecdc4',  # Verde para lateral
            2: '#45b7d1'   # Azul para alcista
        }
        
    def detect_market_regimes(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        🔍 Detectar regímenes de mercado usando clustering inteligente
        
        ✅ MÉTODOS DE DETECCIÓN:
        - Volatilidad del precio
        - Tendencia direccional
        - Momentum del volumen
        - RSI y MACD para confirmación
        """
        
        print(f"🔍 Detectando regímenes de mercado para {symbol}...")
        
        try:
            # ✅ CALCULAR INDICADORES TÉCNICOS PARA DETECCIÓN
            close_prices = df['close'].values.astype(float)
            volumes = df['volume'].values.astype(float)
            
            # 1. Volatilidad (ATR normalizado)
            high_prices = df['high'].values.astype(float)
            low_prices = df['low'].values.astype(float)
            
            atr_14 = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
            atr_normalized = atr_14 / close_prices  # Volatilidad relativa
            
            # 2. Tendencia direccional (SMA slope)
            sma_20 = talib.SMA(close_prices, timeperiod=20)
            sma_50 = talib.SMA(close_prices, timeperiod=50)
            
            # Calcular pendiente de SMA
            sma_slope = np.zeros_like(close_prices)
            for i in range(20, len(close_prices)):
                if i >= 20 and not np.isnan(sma_20[i]):
                    # Pendiente de los últimos 10 períodos
                    start_idx = max(0, i-10)
                    if not np.isnan(sma_20[start_idx]):
                        sma_slope[i] = (sma_20[i] - sma_20[start_idx]) / sma_20[start_idx]
            
            # 3. Momentum del volumen
            volume_sma = talib.SMA(volumes, timeperiod=20)
            volume_momentum = (volumes - volume_sma) / volume_sma
            
            # 4. RSI para confirmación
            rsi_14 = talib.RSI(close_prices, timeperiod=14)
            
            # 5. MACD para confirmación
            macd, macd_signal, macd_hist = talib.MACD(close_prices)
            
            # ✅ PREPARAR FEATURES PARA CLUSTERING
            features_for_clustering = []
            valid_indices = []
            
            for i in range(len(close_prices)):
                # Solo usar índices donde todos los indicadores son válidos
                if (not np.isnan(atr_normalized[i]) and 
                    not np.isnan(sma_slope[i]) and 
                    not np.isnan(volume_momentum[i]) and
                    not np.isnan(rsi_14[i]) and
                    not np.isnan(macd_hist[i])):
                    
                    features_for_clustering.append([
                        atr_normalized[i],
                        sma_slope[i],
                        volume_momentum[i],
                        rsi_14[i] / 100.0,  # Normalizar RSI a [0,1]
                        macd_hist[i] / close_prices[i]  # MACD normalizado
                    ])
                    valid_indices.append(i)
            
            if len(features_for_clustering) < 100:
                print(f"⚠️  Datos insuficientes para clustering: {len(features_for_clustering)} muestras")
                return self._create_fallback_regimes(df, symbol)
            
            # ✅ APLICAR CLUSTERING K-MEANS
            features_array = np.array(features_for_clustering)
            
            # Escalar features para clustering
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_array)
            
            # Clustering con 3 regímenes
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            regime_labels = kmeans.fit_predict(features_scaled)
            
            # ✅ INTERPRETAR REGÍMENES BASADO EN CARACTERÍSTICAS
            regime_interpretations = self._interpret_regimes(features_array, regime_labels)
            
            # ✅ CREAR DATAFRAME CON REGÍMENES
            df_with_regimes = df.iloc[valid_indices].copy()
            df_with_regimes['market_regime'] = regime_labels
            df_with_regimes['regime_name'] = [self.regime_names[label] for label in regime_labels]
            
            # ✅ ANALIZAR DISTRIBUCIÓN DE REGÍMENES
            self._analyze_regime_distribution(df_with_regimes, symbol)
            
            print(f"✅ Regímenes detectados para {symbol}: {len(df_with_regimes)} muestras válidas")
            return df_with_regimes
            
        except Exception as e:
            print(f"⚠️  Error detectando regímenes: {e}")
            print(f"   🔄 Usando detección de regímenes por defecto...")
            return self._create_fallback_regimes(df, symbol)
    
    def _interpret_regimes(self, features: np.ndarray, labels: np.ndarray) -> dict:
        """🎯 Interpretar qué cluster representa cada régimen"""
        
        regime_centers = {}
        for regime_id in range(3):
            regime_mask = labels == regime_id
            if np.any(regime_mask):  # ✅ CORREGIDO: np.any() en lugar de np.any
                regime_features = features[regime_mask]
                regime_centers[regime_id] = np.mean(regime_features, axis=0)
        
        # Interpretar basado en características:
        # [atr_norm, sma_slope, volume_momentum, rsi_norm, macd_norm]
        
        # Encontrar el régimen más bajista (pendiente negativa, RSI bajo)
        regime_scores = {}
        for regime_id, center in regime_centers.items():
            # Score negativo para características bajistas
            score = (center[1] * -2.0 +  # Pendiente negativa = bajista
                    (0.5 - center[3]) * 2.0 +  # RSI bajo = bajista
                    center[4] * -1.0)  # MACD negativo = bajista
            regime_scores[regime_id] = score
        
        # Ordenar regímenes por score (más negativo = más bajista)
        sorted_regimes = sorted(regime_scores.items(), key=lambda x: x[1])
        
        # Mapear: 0=bajista, 1=lateral, 2=alcista
        regime_mapping = {}
        for i, (regime_id, score) in enumerate(sorted_regimes):
            regime_mapping[regime_id] = i
        
        return regime_mapping
    
    def _analyze_regime_distribution(self, df: pd.DataFrame, symbol: str):
        """📊 Analizar distribución de regímenes detectados"""
        
        regime_counts = df['market_regime'].value_counts().sort_index()
        total_samples = len(df)
        
        print(f"📊 DISTRIBUCIÓN DE REGÍMENES - {symbol}:")
        print("=" * 50)
        
        for regime_id in range(3):
            count = regime_counts.get(regime_id, 0)
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            regime_name = self.regime_names[regime_id]
            
            print(f"   {regime_name:>8}: {count:>4} muestras ({percentage:>5.1f}%)")
        
        # ✅ VERIFICAR BALANCE
        min_samples_per_regime = total_samples * 0.15  # Mínimo 15% por régimen
        balanced = all(count >= min_samples_per_regime for count in regime_counts.values())
        
        if balanced:
            print(f"✅ Distribución equilibrada (mínimo {min_samples_per_regime:.0f} por régimen)")
        else:
            print(f"⚠️  Distribución desequilibrada (mínimo {min_samples_per_regime:.0f} por régimen)")
            print(f"   💡 Considera ajustar parámetros de clustering")
    
    def _create_fallback_regimes(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """🔄 Crear regímenes por defecto usando método simple"""
        
        print(f"🔄 Creando regímenes por defecto para {symbol}...")
        
        df_fallback = df.copy()
        close_prices = df_fallback['close'].values.astype(float)
        
        # Método simple basado en SMA
        sma_20 = talib.SMA(close_prices, timeperiod=20)
        sma_50 = talib.SMA(close_prices, timeperiod=50)
        
        regimes = []
        for i in range(len(close_prices)):
            if i < 50 or np.isnan(sma_20[i]) or np.isnan(sma_50[i]):
                regimes.append(1)  # Lateral por defecto
            else:
                # Calcular tendencia
                trend = (sma_20[i] - sma_50[i]) / sma_50[i]
                
                if trend > 0.02:  # 2% por encima
                    regimes.append(2)  # Alcista
                elif trend < -0.02:  # 2% por debajo
                    regimes.append(0)  # Bajista
                else:
                    regimes.append(1)  # Lateral
        
        df_fallback['market_regime'] = regimes
        df_fallback['regime_name'] = [self.regime_names[r] for r in regimes]
        
        print(f"✅ Regímenes por defecto creados para {symbol}")
        return df_fallback
    
    def select_balanced_regime_data(self, df: pd.DataFrame, symbol: str, 
                                  target_samples_per_regime: int = None) -> pd.DataFrame:
        """
        ⚖️ Seleccionar datos equilibrados de cada régimen de mercado
        
        ✅ OBJETIVOS:
        - Mismo número de muestras de cada régimen
        - Distribución temporal equilibrada
        - Calidad de datos consistente
        """
        
        print(f"⚖️ Seleccionando datos equilibrados de regímenes para {symbol}...")
        
        # Detectar regímenes si no están presentes
        if 'market_regime' not in df.columns:
            df = self.detect_market_regimes(df, symbol)
        
        if df.empty:
            print(f"❌ ERROR: No se pudieron detectar regímenes para {symbol}")
            return df
        
        # ✅ CALCULAR MUESTRAS OBJETIVO POR REGÍMEN
        if target_samples_per_regime is None:
            # Usar el régimen con menos muestras como referencia
            regime_counts = df['market_regime'].value_counts()
            target_samples_per_regime = min(regime_counts.values)
            print(f"📊 Objetivo automático: {target_samples_per_regime} muestras por régimen")
        else:
            print(f"📊 Objetivo manual: {target_samples_per_regime} muestras por régimen")
        
        # ✅ SELECCIONAR MUESTRAS EQUILIBRADAS
        balanced_data = []
        
        for regime_id in range(3):
            regime_mask = df['market_regime'] == regime_id
            regime_data = df[regime_mask]
            
            if len(regime_data) == 0:
                print(f"⚠️  Régimen {self.regime_names[regime_id]}: Sin datos")
                continue
            
            # ✅ ESTRATEGIA DE SELECCIÓN INTELIGENTE
            if len(regime_data) <= target_samples_per_regime:
                # Usar todos los datos disponibles
                selected_data = regime_data
                print(f"   {self.regime_names[regime_id]:>8}: {len(selected_data)} muestras (todas disponibles)")
            else:
                # ✅ SELECCIÓN ESTRATIFICADA POR TIEMPO
                selected_data = self._select_stratified_samples(regime_data, target_samples_per_regime)
                print(f"   {self.regime_names[regime_id]:>8}: {len(selected_data)} muestras (seleccionadas de {len(regime_data)})")
            
            balanced_data.append(selected_data)
        
        # ✅ COMBINAR DATOS EQUILIBRADOS
        if balanced_data:
            final_df = pd.concat(balanced_data, ignore_index=True)
            final_df = final_df.sort_values('timestamp').reset_index(drop=True)
            
            # ✅ VERIFICAR BALANCE FINAL
            final_regime_counts = final_df['market_regime'].value_counts().sort_index()
            print(f"\n📊 BALANCE FINAL - {symbol}:")
            print("=" * 40)
            
            for regime_id in range(3):
                count = final_regime_counts.get(regime_id, 0)
                regime_name = self.regime_names[regime_id]
                print(f"   {regime_name:>8}: {count:>4} muestras")
            
            total_final = len(final_df)
            print(f"   {'TOTAL':>8}: {total_final:>4} muestras")
            
            print(f"✅ Datos equilibrados seleccionados para {symbol}")
            return final_df
        else:
            print(f"❌ ERROR: No se pudieron seleccionar datos equilibrados para {symbol}")
            return df
    
    def _select_stratified_samples(self, regime_data: pd.DataFrame, target_samples: int) -> pd.DataFrame:
        """📊 Selección estratificada por tiempo para mantener distribución temporal"""
        
        # Ordenar por timestamp
        regime_data = regime_data.sort_values('timestamp').reset_index(drop=True)
        
        if len(regime_data) <= target_samples:
            return regime_data
        
        # ✅ ESTRATEGIA: Seleccionar muestras distribuidas uniformemente en el tiempo
        step = len(regime_data) / target_samples
        selected_indices = []
        
        for i in range(target_samples):
            index = int(i * step)
            if index < len(regime_data):
                selected_indices.append(index)
        
        # ✅ AGREGAR MUESTRAS ADICIONALES SI ES NECESARIO
        while len(selected_indices) < target_samples:
            # Agregar muestras aleatorias no seleccionadas
            remaining_indices = [i for i in range(len(regime_data)) if i not in selected_indices]
            if remaining_indices:
                selected_indices.append(np.random.choice(remaining_indices))
            else:
                break
        
        selected_indices = sorted(list(set(selected_indices)))  # Eliminar duplicados y ordenar
        
        return regime_data.iloc[selected_indices]


class AdaptiveTCNTrainerV3Improved:
    """🎯 Entrenador TCN V3 MEJORADO con TradingRealityLoss optimizada - TOTALMENTE CONFIGURABLE"""

    def __init__(self, config: TrainingConfig = None):
        # ✅ CONFIGURACIÓN PERSONALIZABLE
        self.config = config if config else TrainingConfig()

        # Usar configuración para parámetros
        self.pairs = self.config.pairs
        self.timeframe = self.config.timeframe
        
        # ✅ OPTIMIZACIÓN AUTOMÁTICA PARA TIMEFRAMES CORTOS
        if self.timeframe in ['1m', '3m']:
            # Para 1m y 3m: Reducir lookback y training days para eficiencia
            self.lookback_window = min(self.config.lookback_window, 16)  # Máximo 16 para 1m/3m
            self.training_days = min(self.config.training_days, 45)      # ✅ AUMENTADO: 15 → 45 días para 1m/3m
            print(f"⚡ OPTIMIZACIÓN 1m/3m: lookback={self.lookback_window}, days={self.training_days}")
        else:
            self.lookback_window = self.config.lookback_window
            # ✅ VALIDACIÓN ADICIONAL: Limitar días para evitar descargas excesivas
            if self.config.training_days > 120:  # ✅ AUMENTADO: 90 → 120 días
                print(f"⚠️  Días de entrenamiento reducidos de {self.config.training_days} a 120 para evitar descargas excesivas")
                self.training_days = 120
            else:
                self.training_days = self.config.training_days
        
        self.prediction_horizon = self.config.prediction_horizon
        self.start_date = self.config.start_date
        self.end_date = self.config.end_date
        self.use_adaptive_thresholds = self.config.use_adaptive_thresholds
        self.feature_set = self.config.feature_set
        
        # ✅ OPTIMIZACIÓN ADICIONAL: Batch size y epochs para timeframes cortos
        if self.timeframe in ['1m', '3m']:
            self.epochs = min(self.config.epochs, 50)        # Máximo 50 epochs para 1m/3m
            self.batch_size = max(self.config.batch_size, 64) # Mínimo 64 batch para eficiencia
        else:
            self.epochs = self.config.epochs
            self.batch_size = self.config.batch_size
        
        # Motor de features centralizado
        self.features_engine = CentralizedFeaturesEngine()

        # ✅ NUEVO: SELECTOR DE REGÍMENES DE MERCADO
        self.regime_selector = MarketRegimeSelector()
        self.use_balanced_regimes = self.config.use_balanced_regimes
        self.regime_balance_method = self.config.regime_balance_method
        self.target_samples_per_regime = self.config.target_samples_per_regime

        # ✅ CONFIGURACIÓN: Thresholds adaptativos y loss mejorada
        self.use_adaptive_thresholds = True
        self.use_improved_loss = True  # ✅ MEJORADO: ImprovedTradingLoss
        
        # 🎯 CONFIGURACIÓN DE IMPROVED TRADING LOSS (EQUILIBRADA)
        self.loss_config = {
            'false_positive_penalty': 1.3,      # ✅ REDUCIDO para más equilibrio
            'false_negative_penalty': 1.1,      # ✅ REDUCIDO para más oportunidades
            'opportunity_loss_penalty': 1.2,    # ✅ REDUCIDO para más equilibrio
            'trade_frequency_incentive': 0.95,  # ✅ MÁS INCENTIVO para más trades
            'confidence_threshold': 0.65,       # ✅ MENOS ESTRICTO para más flexibilidad
            'quality_threshold': 0.6,           # ✅ MENOS ESTRICTO para más trades
            'volatility_weight': True,
            'transaction_cost_aware': True,
            'asymmetric_penalties': True,
            'max_drawdown_penalty': 1.5         # ✅ REDUCIDO para menos conservadurismo
        }
        
        # 🎯 THRESHOLDS ADAPTATIVOS MÁS EQUILIBRADOS
        self.fixed_thresholds = {
            'BTCUSDT': {
                'strong_sell': -0.003, 'weak_sell': -0.0015,  # ✅ MENOS CONSERVADOR
                'weak_buy': 0.0015, 'strong_buy': 0.003
            },
            'ETHUSDT': {
                'strong_sell': -0.002, 'weak_sell': -0.001,
                'weak_buy': 0.001, 'strong_buy': 0.002
            },
            'BNBUSDT': {
                'strong_sell': -0.0015, 'weak_sell': -0.0008,
                'weak_buy': 0.0008, 'strong_buy': 0.0015
            },
            'XRPUSDT': {
                'strong_sell': -0.0015, 'weak_sell': -0.0008,
                'weak_buy': 0.0008, 'strong_buy': 0.0015,
            },
            'DOTUSDT': {
                'strong_sell': -0.0015, 'weak_sell': -0.0008,
                'weak_buy': 0.0008, 'strong_buy': 0.0015,
            },
        }

    def calculate_adaptive_thresholds(self, df: pd.DataFrame, symbol: str) -> dict:
        """🎯 Calcular thresholds adaptativos MÁS EQUILIBRADOS"""
        if not self.use_adaptive_thresholds:
            return self.fixed_thresholds[symbol]
        
        try:
            high_prices = df['high'].values.astype(float)
            low_prices = df['low'].values.astype(float)
            close_prices = df['close'].values.astype(float)
            
            atr_14 = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
            avg_atr = np.nanmean(atr_14[-50:]) if len(atr_14) > 50 else np.nanmean(atr_14)
            avg_price = np.mean(close_prices[-50:]) if len(close_prices) > 50 else np.mean(close_prices)
            
            atr_percent = (avg_atr / avg_price) if avg_price > 0 else 0.02
            
            # 🎯 THRESHOLDS MÁS EQUILIBRADOS (no tan conservadores)
            base_threshold = atr_percent * 0.5  # ✅ REDUCIDO: 0.7 → 0.5 para más trades
            
            adaptive_thresholds = {
                'strong_sell': -base_threshold * 1.3,   # ✅ REDUCIDO: 1.6 → 1.3 (más oportunidades)
                'weak_sell': -base_threshold * 0.6,     # ✅ REDUCIDO: 0.8 → 0.6 (más flexibilidad)
                'weak_buy': base_threshold * 0.6,       # ✅ REDUCIDO: 0.8 → 0.6 (más flexibilidad)
                'strong_buy': base_threshold * 1.3      # ✅ REDUCIDO: 1.6 → 1.3 (más oportunidades)
            }
            
            print(f"🎯 {symbol}: ATR adaptativo {atr_percent:.4f} ({atr_percent*100:.2f}%)")
            print(f"   📊 Thresholds EQUILIBRADOS: Buy {adaptive_thresholds['strong_buy']:.4f}, Sell {adaptive_thresholds['strong_sell']:.4f}")
            
            return adaptive_thresholds
            
        except Exception as e:
            print(f"⚠️ Error calculando thresholds adaptativos para {symbol}: {e}")
            return self.fixed_thresholds[symbol]

    def create_quality_labels(self, df: pd.DataFrame, features: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """🎯 Crear etiquetas de CALIDAD (menos cantidad, más precisión)"""

        print(f"🎯 Creando etiquetas de CALIDAD para {symbol} (anti-sobretrading)...")

        close_prices = df['close'].values
        thresholds = self.calculate_adaptive_thresholds(df, symbol)
        
        labels = []

        for i in range(len(close_prices) - self.prediction_horizon):
            current_price = close_prices[i]
            future_price = close_prices[i + self.prediction_horizon]

            future_return = (future_price - current_price) / current_price

            # 🎯 LÓGICA MÁS CONSERVADORA (calidad > cantidad)
            if future_return <= thresholds['strong_sell']:
                label = 0  # SELL solo para señales muy fuertes
            elif future_return <= thresholds['weak_sell']:
                # ✅ MÁS CONSERVADOR: Zona gris muy selectiva
                try:
                    current_rsi = features['rsi_14'].iloc[i] if i < len(features) else 50
                    current_macd = features['macd_histogram'].iloc[i] if i < len(features) else 0
                except:
                    current_rsi = 50
                    current_macd = 0

                # ✅ UMBRALES MÁS EQUILIBRADOS
                if current_rsi > 60 and current_macd < -0.001:  # ✅ MENOS restrictivo
                    label = 0  # SELL con confirmación moderada
                else:
                    label = 1  # HOLD (más equilibrado)
            elif future_return >= thresholds['strong_buy']:
                label = 2  # BUY solo para señales muy fuertes
            elif future_return >= thresholds['weak_buy']:
                # ✅ MÁS CONSERVADOR: Zona gris muy selectiva
                try:
                    current_rsi = features['rsi_14'].iloc[i] if i < len(features) else 50
                    current_macd = features['macd_histogram'].iloc[i] if i < len(features) else 0
                except:
                    current_rsi = 50
                    current_macd = 0

                # ✅ UMBRALES MÁS EQUILIBRADOS
                if current_rsi < 40 and current_macd > 0.001:  # ✅ MENOS restrictivo
                    label = 2  # BUY con confirmación moderada
                else:
                    label = 1  # HOLD (más equilibrado)
            else:
                # ✅ ZONA NEUTRAL: MÁS EQUILIBRADA
                if i >= 5:
                    recent_momentum = (close_prices[i] - close_prices[i-5]) / close_prices[i-5]
                    # ✅ UMBRALES MÁS BAJOS PARA DETECTAR MOMENTUM MODERADO
                    if recent_momentum > 0.008:  # ✅ REDUCIDO: 0.015 → 0.008 (más oportunidades)
                        label = 2  # BUY con momentum moderado
                    elif recent_momentum < -0.008:  # ✅ REDUCIDO: -0.015 → -0.008 (más oportunidades)
                        label = 0  # SELL con momentum moderado
                    else:
                        label = 1  # HOLD (por defecto)
                else:
                    label = 1  # HOLD

            labels.append(label)

        df_labeled = df.iloc[:-self.prediction_horizon].copy()
        df_labeled['label'] = labels

        # Verificar distribución
        label_counts = pd.Series(labels).value_counts().sort_index()
        total = len(labels)

        print("📊 Distribución de etiquetas de CALIDAD:")
        class_names = ['SELL', 'HOLD', 'BUY']
        for i, name in enumerate(class_names):
            count = label_counts.get(i, 0) or 0
            pct = (count / total * 100) if total > 0 and count is not None else 0
            print(f"   - {name}: {count} ({pct:.1f}%)")

        return df_labeled

    # ✅ MÉTODOS HEREDADOS (con mejoras menores)
    async def get_real_market_data(self, symbol: str, days: int = None) -> pd.DataFrame:
        """📊 Obtener datos reales de mercado - CONFIGURABLE Y PRECISO"""
        # Usar configuración del trainer si no se especifica
        days = days or self.training_days
        print(f"📊 Obteniendo EXACTAMENTE {days} días de datos reales para {symbol}...")

        base_url = "https://api.binance.com"
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        # ✅ CALCULAR LÍMITE EXACTO DE DATOS
        # Para 1m: 1 día = 1440 registros, 90 días = 129,600 registros
        expected_records = days * 1440  # 1440 minutos por día
        print(f"   📊 Registros esperados: {expected_records:,} ({days} días × 1440 minutos)")

        async with aiohttp.ClientSession() as session:
            url = f"{base_url}/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': '1m',
                'startTime': start_time,
                'endTime': end_time,
                'limit': 1000
            }

            all_data = []
            current_start = start_time
            total_records = 0

            while current_start < end_time and total_records < expected_records:
                params['startTime'] = current_start

                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if not data:
                            break
                        
                        # ✅ LIMITAR DATOS AL NÚMERO EXACTO SOLICITADO
                        remaining_needed = expected_records - total_records
                        if len(data) > remaining_needed:
                            data = data[:remaining_needed]
                            print(f"   📊 Limitando a {remaining_needed} registros restantes")
                        
                        all_data.extend(data)
                        total_records += len(data)
                        current_start = data[-1][6] + 1
                        
                        # ✅ VERIFICAR SI YA TENEMOS SUFICIENTES DATOS
                        if total_records >= expected_records:
                            print(f"   ✅ Datos suficientes obtenidos: {total_records:,} registros")
                            break
                    else:
                        print(f"❌ Error API: {response.status}")
                        break

                await asyncio.sleep(0.1)

        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp').sort_index()

        # ✅ VERIFICACIÓN FINAL DE DATOS
        actual_days = len(df) / 1440  # Convertir registros a días
        print(f"✅ Obtenidos {len(df):,} registros de {symbol}")
        print(f"   📊 Días reales obtenidos: {actual_days:.1f} (solicitados: {days})")
        
        if abs(actual_days - days) > 1:
            print(f"   ⚠️  Diferencia significativa: {abs(actual_days - days):.1f} días")
        
        return df

    def prepare_training_data(self, df: pd.DataFrame, features: pd.DataFrame) -> tuple:
        """🔧 Preparar datos para entrenamiento - SIN CAMBIOS"""
        print("🔧 Preparando datos para entrenamiento...")

        features_aligned = features.iloc[:-self.prediction_horizon]
        feature_columns = [col for col in features_aligned.columns if features_aligned[col].dtype in ['float64', 'int64']]

        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features_aligned[feature_columns])

        X = []
        y = []

        for i in range(self.lookback_window, len(features_scaled)):
            sequence = features_scaled[i-self.lookback_window:i]
            X.append(sequence)
            y.append(df['label'].iloc[i])

        X = np.array(X)
        y = np.array(y)

        print(f"✅ Datos preparados: X shape: {X.shape}, y shape: {y.shape}")

        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

        return X, y, scaler, feature_columns, class_weight_dict

    def create_crypto_specialized_tcn_model(self, input_shape: tuple):
        """🚀 Modelo TCN CRYPTO-ESPECIALIZADO con arquitectura multi-escala optimizada para 1m/3m"""

        try:
            from tensorflow.keras.layers import (
                Input, Conv1D, Add, BatchNormalization, Dropout,
                Activation, GlobalAveragePooling1D, Dense, Concatenate
            )
            from tensorflow.keras.models import Model
        except ImportError:
            Input = tf.keras.layers.Input
            Conv1D = tf.keras.layers.Conv1D
            Add = tf.keras.layers.Add
            BatchNormalization = tf.keras.layers.BatchNormalization
            Dropout = tf.keras.layers.Dropout
            Activation = tf.keras.layers.Activation
            GlobalAveragePooling1D = tf.keras.layers.GlobalAveragePooling1D
            Dense = tf.keras.layers.Dense
            Concatenate = tf.keras.layers.Concatenate
            Model = tf.keras.Model

        # ✅ DETECCIÓN AUTOMÁTICA DE TIMEFRAME CORTO
        is_short_timeframe = self.timeframe in ['1m', '3m']
        efficient_mode = is_short_timeframe
        
        print(f"🚀 Creando modelo TCN CRYPTO-ESPECIALIZADO {'ULTRA-EFICIENTE' if efficient_mode else 'OPTIMIZADO'}...")
        if efficient_mode:
            print("   ⚡ MODO EFICIENTE: Activado para timeframes 1m/3m")
            print("   📊 Multi-escala: [1,2] + [4] + [8] (ultra-compacto)")
            print("   🎯 Filtros: Reducidos 50% para máxima velocidad")
        else:
            print("   📊 Multi-escala: [1,2] + [4,8] + [16,32] (optimizado)")
            print("   🎯 Attention: Compacta (16 dims)")
        print("   🔧 Arquitectura: Auto-adaptativa según timeframe")

        # 🎯 INPUTS ESPECIALIZADOS PARA CRYPTO
        inputs = Input(shape=input_shape)
        
        # 🔄 ENCODERS SEPARADOS POR TIPO DE DATO
        # ✅ ARQUITECTURA ADAPTATIVA SEGÚN TIMEFRAME
        feature_dim = input_shape[-1]
        
        print(f"   🔧 Arquitectura {'ULTRA-COMPACTA' if efficient_mode else 'COMPACTA'} para {feature_dim} features")
        
        # 🎯 ENCODER ÚNICO MULTI-ESCALA ADAPTATIVO
        base_filters = 16 if efficient_mode else 32
        x = MultiScaleTCNBlock(filters=base_filters, dropout_rate=0.3, efficient_mode=efficient_mode)(inputs)
        
        # 🎯 ATTENTION ADAPTATIVA
        attention_dim = 8 if efficient_mode else 16
        x = AttentionLayer(attention_dim=attention_dim)(x)
        
        # ⚡ GATING COMPACTO
        x = GatingLayer()(x)
        
        # 🔄 BLOQUE TCN ADICIONAL ADAPTATIVO
        if not efficient_mode:
            # Solo para timeframes >= 5m
            second_filters = base_filters * 2
            x = MultiScaleTCNBlock(filters=second_filters, dropout_rate=0.3, efficient_mode=efficient_mode)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
        
        # 🎯 POOLING GLOBAL
        x = GlobalAveragePooling1D()(x)
        
        # 🧠 HEAD ADAPTATIVO
        if efficient_mode:
            # Para 1m/3m: Head ultra-compacto
            x = Dense(32, activation='relu')(x)
            x = Dropout(0.2)(x)
            x = Dense(16, activation='relu')(x)
            x = Dropout(0.1)(x)
        else:
            # Para 5m+: Head optimizado
            x = Dense(64, activation='relu')(x)   # ✅ REDUCIDO: 256→64
            x = Dropout(0.3)(x)
            x = Dense(32, activation='relu')(x)   # ✅ REDUCIDO: 128→32
            x = Dropout(0.2)(x)
        
        # 🎯 OUTPUT CON CALIBRACIÓN DE CONFIANZA
        outputs = Dense(3, activation='softmax')(x)

        model = Model(inputs, outputs)
        
        # ✅ COMPILAR CON IMPROVED TRADING LOSS
        if self.use_improved_loss:
            improved_loss = ImprovedTradingLoss(self.loss_config)
            model.compile(
                optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-4),  # ✅ Learning rate más bajo para modelo complejo
                loss=improved_loss,
                metrics=['accuracy']
            )
            print(f"🚀 Modelo TCN CRYPTO-ESPECIALIZADO COMPACTO: {model.count_params():,} parámetros")
            print(f"   🎯 ImprovedTradingLoss: HABILITADA")
            print(f"   📊 Arquitectura: Multi-escala + Attention (COMPACTA)")
        else:
            model.compile(
                optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-4),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            print(f"🚀 Modelo TCN CRYPTO-ESPECIALIZADO creado: {model.count_params():,} parámetros")
            print(f"   🎯 Loss estándar: HABILITADA")

        return model

    def create_definitive_tcn_model_v3_improved(self, input_shape: tuple):
        """🎯 Modelo TCN V3 MEJORADO (LEGACY - mantenido para compatibilidad)"""
        return self.create_crypto_specialized_tcn_model(input_shape)

    async def train_improved_model(self, symbol: str) -> bool:
        """🎯 Entrenar modelo V3 MEJORADO anti-sobretrading"""

        print(f"\n🚀 ENTRENANDO MODELO TCN CRYPTO-ESPECIALIZADO PARA {symbol}")
        print("=" * 70)
        print(f"🎯 ImprovedTradingLoss: {'HABILITADA' if self.use_improved_loss else 'DESHABILITADA'}")
        print(f"🎯 Objetivo: EQUILIBRIO CALIDAD/CANTIDAD (más oportunidades)")
        print(f"🚀 Arquitectura: Multi-escala + Attention + Encoders separados")
        print("=" * 70)
        
        # ✅ SOLUCIÓN AGRESIVA: Reset completo de TensorFlow
        try:
            tf.keras.backend.clear_session()
            print("🧹 Sesión de TensorFlow limpiada")
            
            # ✅ CONFIGURACIÓN ADICIONAL: Evitar conflictos de nombres
            import os
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
            
            # ✅ RESET COMPLETO DE GPU
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("🔧 Configuración de GPU reseteada")
            
            # ✅ SOLUCIÓN FINAL: Forzar limpieza de memoria
            import gc
            gc.collect()
            print("🗑️ Memoria limpiada")
            
        except Exception as e:
            print(f"⚠️ Advertencia en limpieza: {e}")

        try:
            # ✅ VALIDACIÓN FINAL DE DÍAS DE ENTRENAMIENTO
            actual_training_days = self.training_days
            if self.timeframe in ['1m', '3m'] and actual_training_days > 45:
                print(f"⚠️  ADVERTENCIA: Reduciendo días de {actual_training_days} a 45 para timeframe {self.timeframe}")
                actual_training_days = 45
            elif self.timeframe == '5m' and actual_training_days > 120:
                print(f"⚠️  ADVERTENCIA: Reduciendo días de {actual_training_days} a 120 para timeframe {self.timeframe}")
                actual_training_days = 120
            
            df = await self.get_real_market_data(symbol, days=actual_training_days)

            # ✅ NUEVO: SELECCIÓN DE REGÍMENES EQUILIBRADOS
            if self.use_balanced_regimes:
                print(f"⚖️ Aplicando selección de regímenes equilibrados...")
                
                try:
                    # Detectar regímenes de mercado
                    df_with_regimes = self.regime_selector.detect_market_regimes(df, symbol)
                    
                    if not df_with_regimes.empty:
                        # Seleccionar datos equilibrados según el método configurado
                        if self.regime_balance_method == 'manual' and self.target_samples_per_regime:
                            df_balanced = self.regime_selector.select_balanced_regime_data(
                                df_with_regimes, symbol, self.target_samples_per_regime
                            )
                        else:
                            df_balanced = self.regime_selector.select_balanced_regime_data(
                                df_with_regimes, symbol
                            )
                        
                        if not df_balanced.empty:
                            df = df_balanced
                            print(f"✅ Datos equilibrados seleccionados: {len(df)} muestras")
                        else:
                            print(f"⚠️  No se pudieron seleccionar datos equilibrados, usando datos originales")
                    else:
                        print(f"⚠️  No se pudieron detectar regímenes, usando datos originales")
                        
                except Exception as e:
                    print(f"⚠️  Error en selección de regímenes: {e}")
                    print(f"   🔄 Continuando con datos originales...")

            print(f"🔄 Calculando features...")
            features = self.features_engine.calculate_features(df, feature_set='tcn_definitivo')

            if features.empty:
                print(f"❌ Error calculando features")
                return False

            # ✅ USAR ETIQUETAS DE CALIDAD
            df_labeled = self.create_quality_labels(df, features, symbol)

            X, y, scaler, feature_columns, class_weights = self.prepare_training_data(df_labeled, features)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # ✅ SOLUCIÓN ADICIONAL: Limpiar memoria antes de crear modelo
            import gc
            gc.collect()
            
            model = self.create_definitive_tcn_model_v3_improved((X.shape[1], X.shape[2]))

            # ✅ CALLBACKS SIMPLIFICADOS PARA EVITAR CONFLICTOS
            model_dir = f'models/improved_v3_{symbol.lower()}'
            os.makedirs(model_dir, exist_ok=True)
            
            # ✅ CALLBACKS OPTIMIZADOS PARA TIMEFRAME
            if self.timeframe in ['1m', '3m']:
                # Para timeframes cortos: Menor paciencia para entrenar más rápido
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, verbose=0),
                    tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, verbose=0)
                ]
            else:
                # Para timeframes normales: Paciencia estándar
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, verbose=0),
                    tf.keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5, verbose=0)
                ]

            print("🚀 Entrenando modelo V3 MEJORADO...")
            
            # ✅ LIMPIAR ARCHIVOS EXISTENTES PARA EVITAR CONFLICTOS
            import shutil
            model_path = f'{model_dir}/best_model.h5'
            if os.path.exists(model_path):
                os.remove(model_path)
                print(f"🧹 Archivo existente eliminado: {model_path}")

            # ✅ SOLUCIÓN DEFINITIVA: Crear datasets manualmente para evitar conflictos
            try:
                # Crear datasets de TensorFlow explícitamente
                train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
                train_dataset = train_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
                
                val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
                val_dataset = val_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
                
                # Usar fit sin validation_data directo
                history = model.fit(
                    train_dataset,
                    validation_data=val_dataset,
                    epochs=self.epochs,
                    callbacks=callbacks,
                    class_weight=class_weights,
                    verbose=1
                )
                
            except Exception as fit_error:
                print(f"⚠️ Error con datasets, intentando método alternativo: {fit_error}")
                
                # ✅ MÉTODO ALTERNATIVO: Sin datasets explícitos
                history = model.fit(
                    X_train, y_train,
                    validation_split=0.2,  # En lugar de validation_data
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    callbacks=[],  # Sin callbacks que puedan causar conflictos
                    class_weight=class_weights,
                    verbose=1
                )

            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            print(f"✅ Accuracy: {test_acc:.3f}")

            # ✅ ANÁLISIS DE ACTIVIDAD ESPERADA
            y_pred = model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            
            pred_counts = pd.Series(y_pred_classes).value_counts().sort_index()
            total_pred = len(y_pred_classes)
            
            print(f"\n📊 ANÁLISIS DE ACTIVIDAD PREDICHA (MEJORADO):")
            class_names = ['SELL', 'HOLD', 'BUY']
            total_trades_predicted = pred_counts.get(0, 0) + pred_counts.get(2, 0)
            trade_activity_pct = (total_trades_predicted / total_pred * 100) if total_pred > 0 else 0
            
            for i, name in enumerate(class_names):
                count = pred_counts.get(i, 0)
                pct = (count / total_pred * 100) if total_pred > 0 else 0
                print(f"   - {name}: {count} ({pct:.1f}%)")
            
            print(f"🎯 ACTIVIDAD DE TRADING PREDICHA: {trade_activity_pct:.1f}% (objetivo: 20-40%)")
            
            if trade_activity_pct < 15:
                print(f"⚠️  ADVERTENCIA: Muy conservador ({trade_activity_pct:.1f}%) - Podría perder oportunidades")
            elif trade_activity_pct > 50:
                print(f"⚠️  ADVERTENCIA: Aún agresivo ({trade_activity_pct:.1f}%) - Podría hacer sobretrading")
            else:
                print(f"✅ ACTIVIDAD ÓPTIMA: {trade_activity_pct:.1f}% - Balance perfecto calidad/cantidad")

            # ✅ GUARDAR TODOS LOS COMPONENTES NECESARIOS - MÉTODO ROBUSTO
            try:
                print("💾 Guardando modelo y todos los componentes...")
                
                # ✅ SOLUCIÓN AGRESIVA: Borrar directorio completo y recrear
                import shutil
                if os.path.exists(model_dir):
                    shutil.rmtree(model_dir)
                    print(f"🗑️ Directorio completo eliminado: {model_dir}")
                
                os.makedirs(model_dir, exist_ok=True)
                print(f"📁 Directorio recreado limpio: {model_dir}")
                
                # ✅ LIMPIAR CUALQUIER REFERENCIA DE TENSORFLOW
                tf.keras.backend.clear_session()
                import gc
                gc.collect()
                
                # 1️⃣ GUARDAR MODELO CON FORMATO ALTERNATIVO (SAVEDMODEL)
                try:
                    # Intentar guardar en formato SavedModel primero (más robusto)
                    model.save(f'{model_dir}/saved_model', save_format='tf')
                    print(f"✅ Modelo guardado: saved_model/ (formato TensorFlow)")
                    
                    # También intentar H5 si es posible
                    try:
                        model.save(f'{model_dir}/model.h5', save_format='h5')
                        print(f"✅ Modelo guardado: model.h5 (formato H5)")
                    except Exception as h5_error:
                        print(f"⚠️ No se pudo guardar H5 (no crítico): {h5_error}")
                        
                except Exception as save_error:
                    print(f"⚠️ Error con SavedModel, intentando solo pesos: {save_error}")
                    
                    # FALLBACK: Guardar solo pesos
                    model.save_weights(f'{model_dir}/model_weights.h5')
                    print(f"✅ Pesos guardados: model_weights.h5 (fallback)")
                    
                    # Guardar arquitectura como JSON
                    with open(f'{model_dir}/model_architecture.json', 'w') as f:
                        f.write(model.to_json())
                    print(f"✅ Arquitectura guardada: model_architecture.json")
                
                # 2️⃣ GUARDAR SCALER (CRÍTICO)
                with open(f'{model_dir}/scaler.pkl', 'wb') as f:
                    pickle.dump(scaler, f)
                print(f"✅ Scaler guardado: scaler.pkl")
                    
                # 3️⃣ GUARDAR COLUMNAS DE FEATURES (CRÍTICO)
                with open(f'{model_dir}/feature_columns.pkl', 'wb') as f:
                    pickle.dump(feature_columns, f)
                print(f"✅ Feature columns guardadas: feature_columns.pkl")
                    
                # 4️⃣ GUARDAR CONFIGURACIÓN DE LOSS (CRÍTICO)
                with open(f'{model_dir}/loss_config.pkl', 'wb') as f:
                    pickle.dump(self.loss_config, f)
                print(f"✅ Loss config guardada: loss_config.pkl")
                
                # 5️⃣ GUARDAR THRESHOLDS ADAPTATIVOS (CRÍTICO)
                current_thresholds = self.calculate_adaptive_thresholds(df, symbol)
                with open(f'{model_dir}/thresholds.pkl', 'wb') as f:
                    pickle.dump(current_thresholds, f)
                print(f"✅ Thresholds guardados: thresholds.pkl")
                
                # 6️⃣ GUARDAR CONFIGURACIÓN DEL TRAINER (IMPORTANTE)
                trainer_config = {
                    'timeframe': self.timeframe,
                    'lookback_window': self.lookback_window,
                    'prediction_horizon': self.prediction_horizon,
                    'feature_set': self.feature_set,
                    'use_adaptive_thresholds': self.use_adaptive_thresholds,
                    'use_improved_loss': self.use_improved_loss,
                    'training_days': self.training_days,
                    'symbol': symbol
                }
                with open(f'{model_dir}/trainer_config.pkl', 'wb') as f:
                    pickle.dump(trainer_config, f)
                print(f"✅ Trainer config guardada: trainer_config.pkl")
                
                # 7️⃣ GUARDAR HISTORIAL DE ENTRENAMIENTO (ÚTIL)
                with open(f'{model_dir}/training_history.pkl', 'wb') as f:
                    pickle.dump(history.history, f)
                print(f"✅ Training history guardado: training_history.pkl")
                
                # 8️⃣ GUARDAR INFO DEL MODELO (JSON LEGIBLE)
                import json
                model_info = {
                    'symbol': symbol,
                    'total_params': int(model.count_params()),
                    'test_accuracy': float(test_acc),
                    'test_loss': float(test_loss),
                    'trade_activity_pct': float(trade_activity_pct),
                    'architecture': 'TCN_CRYPTO_ESPECIALIZADO_COMPACTO',
                    'loss_function': 'ImprovedTradingLoss',
                    'timeframe': self.timeframe,
                    'lookback_window': self.lookback_window,
                    'prediction_horizon': self.prediction_horizon,
                    'feature_set': self.feature_set,
                    'training_date': datetime.now().isoformat(),
                    'epochs_trained': len(history.history['loss']),
                    'final_train_loss': float(history.history['loss'][-1]),
                    'final_val_loss': float(history.history['val_loss'][-1]),
                    'final_val_accuracy': float(history.history['val_accuracy'][-1])
                }
                
                with open(f'{model_dir}/model_info.json', 'w') as f:
                    json.dump(model_info, f, indent=2)
                print(f"✅ Model info guardada: model_info.json")
                
                # 9️⃣ VALIDAR QUE ARCHIVOS ESENCIALES SE GUARDARON
                essential_files = ['scaler.pkl', 'feature_columns.pkl', 'loss_config.pkl']
                model_saved = (
                    os.path.exists(f'{model_dir}/saved_model') or 
                    os.path.exists(f'{model_dir}/model.h5') or 
                    os.path.exists(f'{model_dir}/model_weights.h5')
                )
                
                missing_essential = []
                for req_file in essential_files:
                    if not os.path.exists(f'{model_dir}/{req_file}'):
                        missing_essential.append(req_file)
                
                if missing_essential or not model_saved:
                    print(f"❌ ARCHIVOS ESENCIALES FALTANTES: {missing_essential}")
                    print(f"❌ Modelo guardado: {model_saved}")
                    return False
                else:
                    print(f"✅ TODOS LOS ARCHIVOS ESENCIALES GUARDADOS CORRECTAMENTE")
                    if os.path.exists(f'{model_dir}/saved_model'):
                        print(f"   📁 Modelo: saved_model/ (TensorFlow format)")
                    elif os.path.exists(f'{model_dir}/model.h5'):
                        print(f"   📁 Modelo: model.h5 (H5 format)")
                    else:
                        print(f"   📁 Modelo: model_weights.h5 + model_architecture.json (fallback)")
                    
            except Exception as save_error:
                print(f"❌ ERROR CRÍTICO guardando archivos: {save_error}")
                import traceback
                traceback.print_exc()
                return False

            print(f"✅ Modelo V3 MEJORADO guardado en models/improved_v3_{symbol.lower()}/")
            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            return False


def configurar_interactivamente() -> TrainingConfig:
    """🎯 Configuración INTERACTIVA - El usuario elige todo paso a paso"""

    print("🎯 CONFIGURACIÓN INTERACTIVA DE ENTRENAMIENTO V3 MEJORADO")
    print("=" * 70)
    print("Te voy a preguntar paso a paso qué quieres entrenar...")
    print("=" * 70)

    config = TrainingConfig()

    # 1️⃣ TIMEFRAME
    print(f"\n⏰ PASO 1: TIMEFRAME")
    print(f"Opciones disponibles:")
    timeframes = ['1m', '3m', '5m']
    for i, tf in enumerate(timeframes, 1):
        print(f"  {i}. {tf}")

    while True:
        respuesta = input(f"👉 Elige timeframe [1-3] (default: 1): ").strip()
        if respuesta == '' or respuesta == '1':
            config.timeframe = '1m'
            break
        elif respuesta == '2':
            config.timeframe = '3m'
            break
        elif respuesta == '3':
            config.timeframe = '5m'
            break
        else:
            print("❌ Opción inválida. Elige 1, 2 o 3")

    # 2️⃣ FEATURE SET
    print(f"\n🎯 PASO 2: CONJUNTO DE FEATURES")
    print(f"¿Qué conjunto de features usar?")
    feature_sets = [
        ('tcn_definitivo', '88 features (completo)'),
        ('optimized_crypto', '25 features (optimizado)'),
        ('ultra_optimized', '15 features (ultra optimizado)')
    ]
    for i, (fs, desc) in enumerate(feature_sets, 1):
        print(f"  {i}. {fs} - {desc}")

    while True:
        respuesta = input(f"👉 Elige feature set [1-3] (default: 1): ").strip()
        if respuesta == '' or respuesta == '1':
            config.feature_set = 'tcn_definitivo'
            break
        elif respuesta == '2':
            config.feature_set = 'optimized_crypto'
            break
        elif respuesta == '3':
            config.feature_set = 'ultra_optimized'
            break
        else:
            print("❌ Opción inválida. Elige 1, 2 o 3")

    # 3️⃣ PAR DE TRADING (UN SOLO PAR COMO trainbnb.py)
    print(f"\n💎 PASO 3: PAR DE TRADING")
    print(f"Pares disponibles:")
    pares_disponibles = ['BTCUSDT', 'ETHUSDT', 'DOTUSDT', 'XRPUSDT', 'BNBUSDT', 'ADAUSDT']
    for i, par in enumerate(pares_disponibles, 1):
        print(f"  {i}. {par}")

    while True:
        respuesta = input(f"👉 Elige el par a entrenar [1-6] (default: 1): ").strip()
        if respuesta == '' or respuesta == '1':
            config.pairs = ['BTCUSDT']
            break
        elif respuesta == '2':
            config.pairs = ['ETHUSDT']
            break
        elif respuesta == '3':
            config.pairs = ['DOTUSDT']
            break
        elif respuesta == '4':
            config.pairs = ['XRPUSDT']
            break
        elif respuesta == '5':
            config.pairs = ['BNBUSDT']
            break
        elif respuesta == '6':
            config.pairs = ['ADAUSDT']
            break
        else:
            print("❌ Opción inválida. Elige 1, 2, 3, 4, 5 o 6")

    # 4️⃣ HORIZONTE DE PREDICCIÓN
    print(f"\n🔮 PASO 4: HORIZONTE DE PREDICCIÓN")
    print(f"¿Cuántos minutos en el futuro predecir?")
    horizontes = [3, 6, 12]
    for i, h in enumerate(horizontes, 1):
        print(f"  {i}. {h} minutos")

    while True:
        respuesta = input(f"👉 Elige horizonte [1-3] (default: 2): ").strip()
        if respuesta == '' or respuesta == '2':
            config.prediction_horizon = 6
            break
        elif respuesta == '1':
            config.prediction_horizon = 3
            break
        elif respuesta == '3':
            config.prediction_horizon = 12
            break
        else:
            print("❌ Opción inválida. Elige 1, 2 o 3")

    # 5️⃣ VENTANA DE LOOKBACK
    print(f"\n📊 PASO 5: VENTANA DE ANÁLISIS")
    print(f"¿Cuántos puntos de datos históricos usar?")
    ventanas = [24, 32, 48]
    for i, v in enumerate(ventanas, 1):
        print(f"  {i}. {v} períodos")

    while True:
        respuesta = input(f"👉 Elige ventana [1-3] (default: 1): ").strip()
        if respuesta == '' or respuesta == '1':
            config.lookback_window = 24
            break
        elif respuesta == '2':
            config.lookback_window = 32
            break
        elif respuesta == '3':
            config.lookback_window = 48
            break
        else:
            print("❌ Opción inválida. Elige 1, 2 o 3")

    # 6️⃣ DÍAS DE DATOS
    print(f"\n📅 PASO 6: DATOS DE ENTRENAMIENTO")
    
    # ✅ ADVERTENCIA SOBRE LÍMITES SEGÚN TIMEFRAME
    if config.timeframe in ['1m', '3m']:
        print(f"⚠️  TIMEFRAME CORTO ({config.timeframe}): Máximo 45 días recomendado para eficiencia")
        print(f"   💡 Con 45 días tendrás ~64,800 muestras (suficiente para entrenamiento robusto)")
        max_days = 45
    else:
        print(f"📊 TIMEFRAME NORMAL ({config.timeframe}): Máximo 120 días recomendado")
        print(f"   💡 Con 120 días tendrás ~172,800 muestras (excelente para entrenamiento)")
        max_days = 120
    
    while True:
        respuesta = input(f"👉 ¿Cuántos días de datos usar? (default: {max_days//2}, max: {max_days}): ").strip()
        if respuesta == '':
            config.training_days = max_days // 2
            break
        try:
            dias = int(respuesta)
            if 1 <= dias <= max_days:
                config.training_days = dias
                break
            else:
                print(f"❌ Días debe estar entre 1 y {max_days} para timeframe {config.timeframe}")
        except:
            print("❌ Ingresa un número válido")

    # 7️⃣ ÉPOCAS DE ENTRENAMIENTO
    print(f"\n🎯 PASO 7: ÉPOCAS DE ENTRENAMIENTO")
    while True:
        respuesta = input(f"👉 ¿Cuántas épocas entrenar? (default: 50): ").strip()
        if respuesta == '':
            config.epochs = 50
            break
        try:
            epochs = int(respuesta)
            if 1 <= epochs <= 200:
                config.epochs = epochs
                break
            else:
                print("❌ Épocas debe estar entre 1 y 200")
        except:
            print("❌ Ingresa un número válido")



    # 8️⃣ BATCH SIZE
    print(f"\n📦 PASO 8: TAMAÑO DE BATCH")
    print(f"Opciones recomendadas:")
    batches = [32, 64, 128]
    for i, b in enumerate(batches, 1):
        print(f"  {i}. {b}")

    while True:
        respuesta = input(f"👉 Elige batch size [1-3] (default: 2): ").strip()
        if respuesta == '' or respuesta == '2':
            config.batch_size = 64
            break
        elif respuesta == '1':
            config.batch_size = 32
            break
        elif respuesta == '3':
            config.batch_size = 128
            break
        else:
            print("❌ Opción inválida. Elige 1, 2 o 3")

    # ✅ NUEVO: PASO 9: REGÍMENES DE MERCADO EQUILIBRADOS
    print(f"\n⚖️ PASO 9: REGÍMENES DE MERCADO EQUILIBRADOS")
    print(f"¿Quieres entrenar con datos equilibrados de diferentes regímenes de mercado?")
    print(f"Esto asegura que el modelo funcione bien en mercados alcistas, bajistas y laterales.")
    
    while True:
        respuesta = input(f"👉 ¿Usar regímenes equilibrados? [s/N] (default: N): ").strip().lower()
        if respuesta == '' or respuesta in ['n', 'no']:
            config.use_balanced_regimes = False
            break
        elif respuesta in ['s', 'si', 'sí', 'y', 'yes']:
            config.use_balanced_regimes = True
            break
        else:
            print("❌ Responde 's' para sí o 'N' para no")

    # ✅ CONFIGURACIÓN ADICIONAL SI SE SELECCIONAN REGÍMENES EQUILIBRADOS
    if config.use_balanced_regimes:
        print(f"\n📊 CONFIGURACIÓN DE REGÍMENES EQUILIBRADOS")
        print(f"Métodos disponibles:")
        methods = [
            ('auto', 'Automático - Balance automático'),
            ('manual', 'Manual - Especificar muestras por régimen'),
            ('stratified', 'Estratificado - Mantener distribución temporal')
        ]
        
        for i, (method, desc) in enumerate(methods, 1):
            print(f"  {i}. {method} - {desc}")

        while True:
            respuesta = input(f"👉 Elige método [1-3] (default: 1): ").strip()
            if respuesta == '' or respuesta == '1':
                config.regime_balance_method = 'auto'
                break
            elif respuesta == '2':
                config.regime_balance_method = 'manual'
                break
            elif respuesta == '3':
                config.regime_balance_method = 'stratified'
                break
            else:
                print("❌ Opción inválida. Elige 1, 2 o 3")

        # ✅ CONFIGURACIÓN MANUAL SI SE SELECCIONA
        if config.regime_balance_method == 'manual':
            while True:
                respuesta = input(f"👉 ¿Cuántas muestras por régimen? (default: 500): ").strip()
                if respuesta == '':
                    config.target_samples_per_regime = 500
                    break
                try:
                    samples = int(respuesta)
                    if 100 <= samples <= 2000:
                        config.target_samples_per_regime = samples
                        break
                    else:
                        print("❌ Usa entre 100 y 2000 muestras por régimen")
                except:
                    print("❌ Ingresa un número válido")

    return config


def validate_training_requirements(symbol: str, config: TrainingConfig) -> bool:
    """🔍 Validar requerimientos antes de entrenar"""
    
    print(f"🔍 Validando requerimientos para {symbol}...")
    
    # ✅ VALIDACIÓN 1: Verificar que el símbolo es válido
    valid_symbols = ['BTCUSDT', 'ETHUSDT', 'DOTUSDT', 'XRPUSDT', 'BNBUSDT', 'ADAUSDT']
    if symbol not in valid_symbols:
        print(f"❌ ERROR: Símbolo {symbol} no está en la lista de válidos: {valid_symbols}")
        return False

    # ✅ VALIDACIÓN 2: Verificar que el timeframe es válido
    valid_timeframes = ['1m', '3m', '5m']
    if config.timeframe not in valid_timeframes:
        print(f"❌ ERROR: Timeframe {config.timeframe} no válido. Opciones: {valid_timeframes}")
        return False

    print(f"✅ Validaciones pasadas para {symbol}")
    return True


async def main():
    """🎯 Entrenar modelos V3 MEJORADOS con configuración INTERACTIVA"""
    
    print("🚀 ENTRENADOR TCN CRYPTO-ESPECIALIZADO - CONFIGURACIÓN INTERACTIVA")
    print("=" * 80)
    print("🎯 Te voy a preguntar paso a paso qué quieres entrenar")
    print("🚀 EQUILIBRIO: Calidad = Cantidad (más oportunidades)")
    print("🚀 ARQUITECTURA: Multi-escala + Attention + Encoders separados")
    print("=" * 80)
    
    # ✅ CONFIGURACIÓN INTERACTIVA
    config = configurar_interactivamente()

    # ✅ CONFIRMACIÓN FINAL
    print(f"\n" + "="*70)
    print(f"📋 RESUMEN DE TU CONFIGURACIÓN:")
    config.print_config()
    print(f"="*70)

    respuesta = input(f"\n👉 ¿Todo correcto? ¿Empezar entrenamiento? [s/N]: ").strip().lower()
    if respuesta not in ['s', 'y', 'yes', 'si', 'sí']:
        print("❌ Entrenamiento cancelado. ¡Hasta luego!")
        return

    # ✅ CREAR TRAINER CON CONFIGURACIÓN
    trainer = AdaptiveTCNTrainerV3Improved(config)
    
    print(f"\n💡 CONFIGURACIÓN EQUILIBRADA:")
    print(f"   False Positive Penalty: {trainer.loss_config['false_positive_penalty']}x (equilibrado)")
    print(f"   Trade Frequency Incentive: {trainer.loss_config['trade_frequency_incentive']}x (más incentivo)")
    print(f"   Confidence Threshold: {trainer.loss_config['confidence_threshold']} (más flexible)")
    print(f"   Quality Threshold: {trainer.loss_config['quality_threshold']} (más flexible)")

    # ✅ ENTRENAR EL PAR SELECCIONADO (un solo par como trainbnb.py)
    symbol = config.pairs[0]  # Solo un par fue seleccionado
    print(f"\n🚀 ENTRENANDO V3 MEJORADO PARA {symbol}...")
    print("=" * 60)
    
    # Validar antes de entrenar
    if not validate_training_requirements(symbol, config):
        print(f"❌ VALIDACIÓN FALLIDA para {symbol}. Entrenamiento cancelado.")
        return
    
    # Entrenar modelo
    success = await trainer.train_improved_model(symbol)
    
    # ✅ RESULTADO FINAL
    if success:
        print(f"\n🚀 MODELO TCN CRYPTO-ESPECIALIZADO ENTRENADO EXITOSAMENTE:")
        print(f"   - Par: {symbol}")
        print(f"   - Timeframe: {config.timeframe}")
        print(f"   - Archivo: models/improved_v3_{symbol.lower()}/")
        print(f"   - Configuración equilibrada implementada")
        print(f"   - Balance calidad/cantidad optimizado")
        print(f"   - 🚀 Arquitectura: Multi-escala + Attention + Encoders separados")
        print(f"   - 🎯 Especializado para características únicas del crypto")
        print(f"   - Listo para backtest y ensemble")
    else:
        print(f"\n❌ ERROR EN ENTRENAMIENTO DEL MODELO V3 MEJORADO")


if __name__ == "__main__":
    # 🎯 MENÚ INTERACTIVO DIRECTO (como trainbnb.py)
    asyncio.run(main())
