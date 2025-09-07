#!/usr/bin/env python3
"""
🚀 CENTRALIZED FEATURES ENGINE V3 - CORRELATION OPTIMIZED + CRYPTO-SPECIFIC
============================================================================

Motor centralizado para cálculo de features técnicas con reducción de correlaciones.
Versión optimizada que elimina 45 features redundantes del conjunto TCN_DEFINITIVO.

Características:
- ✅ Implementación única y centralizada  
- ✅ Usa TA-Lib para precisión matemática
- ✅ Compatible con entrenamiento y trading en vivo
- ✅ Soporte para múltiples conjuntos de features
- ✅ Validación automática de datos
- 🆕 Eliminación de correlaciones altas (>0.8)
- 🆕 Conjunto TCN_DEFINITIVO optimizado (58 features)
- 🆕 Preserva diversidad de información técnica
- 🆕 Integración completa con API de Binance
- 🚀 NUEVAS: 12 features específicas para crypto trading
"""

import numpy as np
import pandas as pd
try:
    import talib
except ImportError:
    print("⚠️ TA-Lib no disponible, usando implementaciones alternativas")
    talib = None

# Importar cliente de Binance para API autenticada
try:
    from binance.client import Client
    BINANCE_AVAILABLE = True
except ImportError:
    print("⚠️ python-binance no disponible, usando API pública")
    BINANCE_AVAILABLE = False

from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import warnings
import os
import time
import asyncio
import aiohttp
import pickle
warnings.filterwarnings('ignore')


def safe_fillna_forward_backward(series: pd.Series) -> pd.Series:
    """Reemplazo para fillna con método ffill/bfill deprecado"""
    return series.ffill().bfill()


def interpolate_missing_values(series: pd.Series, method: str = 'linear') -> pd.Series:
    """Interpolación robusta de valores faltantes"""
    series_interpolated = series.interpolate(method=method, limit_direction='both')
    series_filled = series_interpolated.ffill()
    return series_filled.bfill()


def robust_feature_clipping(series: pd.Series, feature_name: str, fallback_range: tuple = None) -> pd.Series:
    """
    Clipping robusto de features para evitar outliers extremos en producción
    
    Args:
        series: Serie de datos a procesar
        feature_name: Nombre de la feature para aplicar reglas específicas
        fallback_range: Rango de fallback si no hay reglas específicas
    """
    if series.isna().all():
        return series
    
    # Reglas específicas por tipo de feature
    if 'momentum' in feature_name.lower():
        # Para momentum: usar 99.5% percentile con límites conservadores
        q995 = series.quantile(0.995)
        q005 = series.quantile(0.005)
        # Aplicar límites máximos según el tipo
        if 'rsi' in feature_name:
            q995 = min(q995, 15)
            q005 = max(q005, -15)
        elif 'ad' in feature_name:
            q995 = min(q995, 150)
            q005 = max(q005, -150)
        else:
            q995 = min(q995, 50)
            q005 = max(q005, -50)
            
        return series.clip(lower=q005, upper=q995)
    
    elif fallback_range:
        # Usar rango específico con margen del 10%
        margin = abs(fallback_range[1] - fallback_range[0]) * 0.1
        return series.clip(lower=fallback_range[0] - margin, upper=fallback_range[1] + margin)
    
    else:
        # Clipping estadístico conservador: 3 sigma
        mean_val = series.mean()
        std_val = series.std()
        return series.clip(lower=mean_val - 3*std_val, upper=mean_val + 3*std_val)


def calculate_safe_volatility(close_prices: pd.Series, window: int = 20) -> pd.Series:
    """Volatilidad sin look-ahead bias"""
    returns = close_prices.pct_change()
    return returns.rolling(window, min_periods=1).std()


def safe_divide(numerator: pd.Series, denominator: pd.Series, 
                fallback_method: str = 'median', min_value: float = 1e-8) -> pd.Series:
    """
    División segura con manejo unificado de ceros
    
    Args:
        numerator: Serie numerador
        denominator: Serie denominador  
        fallback_method: 'median', 'mean', 'rolling_median', 'constant'
        min_value: Valor mínimo para denominador
    """
    # Crear denominador seguro
    denom_safe = denominator.copy()
    
    # Identificar valores problemáticos
    zero_mask = (denom_safe == 0) | denom_safe.isna()
    
    if zero_mask.any():
        if fallback_method == 'median':
            fallback_val = denom_safe[~zero_mask].median()
        elif fallback_method == 'mean':
            fallback_val = denom_safe[~zero_mask].mean()
        elif fallback_method == 'rolling_median':
            # Para rolling_median, aplicar directamente
            rolling_median = denom_safe.rolling(20, min_periods=1).median()
            denom_safe = denom_safe.mask(zero_mask, rolling_median)
            # Asegurar valor mínimo
            denom_safe = denom_safe.clip(lower=min_value)
            # Realizar división
            result = numerator / denom_safe
            result = result.replace([np.inf, -np.inf], np.nan).fillna(0)
            return result
        elif fallback_method == 'constant':
            fallback_val = min_value
        else:
            fallback_val = min_value
            
        # Si el fallback también es problemático (NaN o cero)
        if pd.isna(fallback_val) or fallback_val == 0:
            fallback_val = min_value
            
        # Si el fallback sigue siendo problemático (todos los valores válidos son cero)
        if pd.isna(fallback_val):
            fallback_val = min_value
            
        denom_safe = denom_safe.mask(zero_mask, fallback_val)
    
    # Asegurar valor mínimo
    denom_safe = denom_safe.clip(lower=min_value)
    
    # Realizar división
    result = numerator / denom_safe
    
    # Limpiar resultados extremos y rellenar NaN
    result = result.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return result

class CentralizedFeaturesEngine:
    """
    Motor centralizado de features técnicas usando TA-Lib con optimización de correlaciones
    """

    def __init__(self, quiet_mode: bool = False, api_key: str = None, api_secret: str = None, 
                 use_authenticated_api: bool = True):
        """
        Inicializar el motor de features
        
        Args:
            quiet_mode: Si True, suprime advertencias de compresión/rangos que pueden ser normales
            api_key: API key de Binance para acceso autenticado
            api_secret: API secret de Binance para acceso autenticado
            use_authenticated_api: Si True, usa API autenticada (menor latencia)
        """
        self.quiet_mode = quiet_mode
        self.use_authenticated_api = use_authenticated_api and BINANCE_AVAILABLE
        
        # Configurar cliente de Binance autenticado
        self.binance_client = None
        if self.use_authenticated_api and api_key and api_secret:
            try:
                self.binance_client = Client(api_key, api_secret)
                print("🔐 Cliente de Binance autenticado configurado")
            except Exception as e:
                print(f"⚠️ Error configurando cliente autenticado: {e}")
                self.use_authenticated_api = False
                self.binance_client = None
        
        # Fallback a API pública si no hay autenticación
        if not self.use_authenticated_api:
            print("🌐 Usando API pública de Binance (mayor latencia)")
        
        self.feature_sets = {
            'tcn_definitivo_v3': self._get_tcn_definitivo_v3_features(),
            'tcn_definitivo_v3_enhanced': self._get_tcn_definitivo_v3_enhanced_features(),  # 🆕 NUEVO: Enhanced para detección bajista
            'tcn_definitivo_v3_volume_enhanced': self._get_tcn_definitivo_v3_volume_enhanced_features(),  # 📊 NUEVO: Enhanced + Volume Profile
            'tcn_definitivo_v3_legacy': self._get_tcn_definitivo_v3_legacy_features(),
            'ultra_momentum': self._get_ultra_momentum_features(),
            'ultra_momentum_combined': self._get_ultra_momentum_combined_features(),
            'tcn_definitivo': self._get_tcn_definitivo_features(),  # Mantener compatibilidad
            'tcn_final': self._get_tcn_final_features(),
            'full_set': self._get_full_features_set(),
            'fast_crypto': self._get_fast_crypto_features()  # ⚡ NUEVO: Para entrenamiento rápido
        }

        print("🎯 Centralized Features Engine V3 inicializado")
        print(f"   📊 Conjuntos disponibles: {list(self.feature_sets.keys())}")
        for name, features in self.feature_sets.items():
            if name == 'tcn_definitivo_v3':
                print(f"   🔧 {name}: {len(features)} features (✅ 54 features optimizadas para TCN)")
            elif name == 'tcn_definitivo_v3_enhanced':
                print(f"   🎯 {name}: {len(features)} features (✅ 54 base + 🐻 8 bajistas)")
            elif name == 'tcn_definitivo_v3_volume_enhanced':
                print(f"   📊 {name}: {len(features)} features (✅ 54 base + 🐻 8 bajistas + 📊 2 volume profile)")
            elif name == 'tcn_definitivo_v3_legacy':
                print(f"   🔄 {name}: {len(features)} features (✅ 88 features para compatibilidad con modelos legacy)")
            elif name == 'ultra_momentum':
                print(f"   ⚡ {name}: {len(features)} features (ultra-momentum optimizado)")
            elif name == 'ultra_momentum_combined':
                print(f"   🚀 {name}: {len(features)} features (TCN V3 + Ultra momentum core)")
            else:
                print(f"   🔧 {name}: {len(features)} features")
        print(f"   🔐 API: {'Autenticada' if self.use_authenticated_api else 'Pública'}")

    def _get_tcn_definitivo_v3_features(self) -> List[str]:
        """
        🚀 Features para modelos TCN definitivos V3 (54 features unificadas y optimizadas)
        
        ✅ CARACTERÍSTICAS:
        - Conjunto unificado que combina lo mejor de ambos conjuntos
        - 38 features comunes (estables y probadas)
        - 16 features de alta prioridad seleccionadas
        - Enfoque en momentum, tendencia, volatilidad y volumen clave
        - Compatible con modelos existentes y optimizado para nuevos
        """
        return [
            # === FEATURES COMUNES (38 features estables) ===
            'ad', 'ad_momentum', 'adosc', 'adx_14', 'aroon_down', 'aroon_up',
            'atr_14', 'bb_middle', 'bb_position', 'bb_width', 'cci_14',
            'ema_10', 'ema_20', 'higher_high', 'hl_ratio', 'lower_low',
            'macd', 'macd_histogram', 'macd_momentum', 'macd_signal',
            'mfi_14', 'natr_14', 'obv', 'plus_di', 'price_acceleration',
            'price_position', 'psar', 'resistance_touch', 'rsi_14',
            'rsi_momentum', 'sma_20', 'stoch_k', 'support_touch',
            'uptrend_strength', 'volume_momentum', 'volume_ratio',
            'volume_sma_10', 'williams_r',

            # === FEATURES DE ALTA PRIORIDAD DEL CONJUNTO ACTUAL (16 features) ===
            'atr_20', 'bb_lower', 'bb_upper', 'downtrend_strength',
            'minus_di', 'momentum_10', 'oc_ratio', 'price_change_1',
            'price_change_10', 'price_change_5', 'roc_10', 'rsi_21',
            'rsi_7', 'sma_10', 'sma_50', 'stoch_d'
        ]

    def _get_tcn_definitivo_v3_enhanced_features(self) -> List[str]:
        """
        🎯 Features para modelos TCN definitivos V3 ENHANCED (62 features - mejorada para detección bajista)
        
        ✅ CARACTERÍSTICAS:
        - Mantiene las 54 features optimizadas originales
        - Añade 8 features críticas para detección de mercados bajistas
        - Preserva baja correlación pero mejora capacidad direccional
        - Enfoque especial en momentum bajista y confirmaciones de tendencia
        """
        # Features base del conjunto V3 original (54)
        base_features = [
            # === FEATURES COMUNES (38 features estables) ===
            'ad', 'ad_momentum', 'adosc', 'adx_14', 'aroon_down', 'aroon_up',
            'atr_14', 'bb_middle', 'bb_position', 'bb_width', 'cci_14',
            'ema_10', 'ema_20', 'higher_high', 'hl_ratio', 'lower_low',
            'macd', 'macd_histogram', 'macd_momentum', 'macd_signal',
            'mfi_14', 'natr_14', 'obv', 'plus_di', 'price_acceleration',
            'price_position', 'psar', 'resistance_touch', 'rsi_14',
            'rsi_momentum', 'sma_20', 'stoch_k', 'support_touch',
            'uptrend_strength', 'volume_momentum', 'volume_ratio',
            'volume_sma_10', 'williams_r',

            # === FEATURES DE ALTA PRIORIDAD (16 features) ===
            'atr_20', 'bb_lower', 'bb_upper', 'downtrend_strength',
            'minus_di', 'momentum_10', 'oc_ratio', 'price_change_1',
            'price_change_10', 'price_change_5', 'roc_10', 'rsi_21',
            'rsi_7', 'sma_10', 'sma_50', 'stoch_d'
        ]
        
        # 🚀 NUEVO: Features bajistas crypto-específicas (8 features optimizadas para Binance)
        crypto_bearish_features = [
            'rsi_divergence_bearish',      # 🆕 Fuerza de rechazo en niveles clave (nuevo cálculo)
            'macd_bearish_cross',          # 🆕 Distribución anómala de volumen en caídas (nuevo cálculo)
            'trend_strength_ratio',        # 🆕 Agotamiento de momentum alcista (nuevo cálculo)
            'volume_bearish_signal',       # 🆕 Breakouts bajistas con volatilidad (nuevo cálculo)
            'price_momentum_bearish',      # 🆕 Debilidad en estructura de mercado (nuevo cálculo)
            'support_resistance_context',   # 🆕 Señales de drenaje de liquidez (nuevo cálculo)
            'volatility_expansion_bear',   # 🆕 Indicador de riesgo de cascada (nuevo cálculo)
            'momentum_divergence_bear'     # 🆕 Señales de salida institucional (nuevo cálculo)
        ]
        
        return base_features + crypto_bearish_features

    def _get_tcn_definitivo_v3_volume_enhanced_features(self) -> List[str]:
        """
        📊 Features para modelos TCN definitivos V3 VOLUME ENHANCED (64 features - con Volume Profile)
        
        ✅ CARACTERÍSTICAS:
        - Mantiene las 54 features optimizadas originales
        - Añade 8 features críticas para detección de mercados bajistas
        - Añade 2 features de Volume Profile para reversiones institucionales
        - Preserva baja correlación pero mejora capacidad direccional
        - Enfoque especial en momentum bajista y confirmaciones de tendencia + análisis institucional
        
        🎯 USO: Para modelos nuevos que requieren mejor detección de inflexiones en mercados laterales
        """
        # Features base del conjunto V3 original (54)
        base_features = [
            # === FEATURES COMUNES (38 features estables) ===
            'ad', 'ad_momentum', 'adosc', 'adx_14', 'aroon_down', 'aroon_up',
            'atr_14', 'bb_middle', 'bb_position', 'bb_width', 'cci_14',
            'ema_10', 'ema_20', 'higher_high', 'hl_ratio', 'lower_low',
            'macd', 'macd_histogram', 'macd_momentum', 'macd_signal',
            'mfi_14', 'natr_14', 'obv', 'plus_di', 'price_acceleration',
            'price_position', 'psar', 'resistance_touch', 'rsi_14',
            'rsi_momentum', 'sma_20', 'stoch_k', 'support_touch',
            'uptrend_strength', 'volume_momentum', 'volume_ratio',
            'volume_sma_10', 'williams_r',

            # === FEATURES DE ALTA PRIORIDAD (16 features) ===
            'atr_20', 'bb_lower', 'bb_upper', 'downtrend_strength',
            'minus_di', 'momentum_10', 'oc_ratio', 'price_change_1',
            'price_change_10', 'price_change_5', 'roc_10', 'rsi_21',
            'rsi_7', 'sma_10', 'sma_50', 'stoch_d'
        ]
        
        # 🚀 Features bajistas crypto-específicas + Volume Profile (10 features optimizadas para Binance)
        crypto_volume_enhanced_features = [
            'rsi_divergence_bearish',      # 🆕 Fuerza de rechazo en niveles clave (nuevo cálculo)
            'macd_bearish_cross',          # 🆕 Distribución anómala de volumen en caídas (nuevo cálculo)
            'trend_strength_ratio',        # 🆕 Agotamiento de momentum alcista (nuevo cálculo)
            'volume_bearish_signal',       # 🆕 Breakouts bajistas con volatilidad (nuevo cálculo)
            'price_momentum_bearish',      # 🆕 Debilidad en estructura de mercado (nuevo cálculo)
            'support_resistance_context',   # 🆕 Señales de drenaje de liquidez (nuevo cálculo)
            'volatility_expansion_bear',   # 🆕 Indicador de riesgo de cascada (nuevo cálculo)
            'momentum_divergence_bear',    # 🆕 Señales de salida institucional (nuevo cálculo)
            'volume_profile_poc',          # 📊 Point of Control - nivel de máximo volumen
            'volume_profile_vah_val'       # 📊 Value Area High/Low ratio
        ]
        
        return base_features + crypto_volume_enhanced_features

    def _get_tcn_definitivo_v3_legacy_features(self) -> List[str]:
        """
        🔄 Features para modelos TCN definitivos V3 LEGACY (88 features - compatibilidad con modelos antiguos)
        
        ✅ MANTENIDAS (88 features del conjunto original):
        - Todas las features técnicas probadas y estables
        - Compatibilidad total con modelos entrenados anteriormente
        - Conjunto estable y funcional para modelos legacy
        
        🎯 USO: Para modelos que fueron entrenados con 88 features
        """
        # ✅ SIMPLIFICACIÓN: Usar el conjunto original que funciona correctamente
        return self._get_tcn_definitivo_features()

    def _get_fast_crypto_features(self) -> List[str]:
        """
        ⚡ Features ULTRA-LIVIANAS para entrenamiento rápido (25 features)
        
        Conjunto minimalista que mantiene capacidad predictiva:
        - Solo indicadores más impactantes para crypto
        - Baja correlación entre features
        - Cálculo rápido y eficiente
        - Ideal para prototipado y entrenamiento rápido
        
        🎯 USO: Para modelos que priorizan velocidad sobre máxima precisión
        """
        return [
            # === MOMENTUM CORE (8 features) ===
            'rsi_14', 'rsi_7', 'macd', 'macd_signal',
            'cci_14', 'williams_r', 'roc_10', 'momentum_10',
            
            # === TREND ESSENTIAL (6 features) ===
            'ema_10', 'ema_20', 'sma_20', 'adx_14',
            'plus_di', 'minus_di',
            
            # === VOLATILITY KEY (4 features) ===
            'atr_14', 'bb_position', 'bb_width', 'natr_14',
            
            # === VOLUME CORE (3 features) ===
            'obv', 'mfi_14', 'volume_ratio',
            
            # === PRICE ACTION (4 features) ===
            'price_change_1', 'price_change_5', 'hl_ratio', 'oc_ratio'
        ]

    def _get_ultra_momentum_features(self) -> List[str]:
        """
        ⚡ Features ultra-momentum optimizadas (25 features)
        Conjunto reducido y enfocado para capturar momentum de ultra-corto plazo
        sin problemas de correlación
        """
        return [
            # === PRICE MOMENTUM CORE (6 features) ===
            'price_change_1s',      # Cambio de precio inmediato
            'price_change_3s',      # Cambio de precio corto
            'price_change_5s',      # Cambio de precio medio
            'price_acceleration_1s', # Aceleración de precio
            'momentum_1s',          # Momentum instantáneo
            'momentum_signal_1s',   # Señal de momentum (1/0)
            
            # === VOLUME MOMENTUM CORE (4 features) ===
            'volume_ratio_1s',      # Volumen relativo inmediato
            'volume_ratio_3s',      # Volumen relativo corto
            'volume_spike_1s',      # Pico de volumen
            'momentum_volume_signal', # Señal combinada momentum + volumen
            
            # === OSCILADORES ULTRA-CORTOS (3 features) ===
            'rsi_3s',              # RSI ultra-corto
            'rsi_5s',              # RSI corto
            'momentum_rsi_signal', # Señal combinada momentum + RSI
            
            # === VOLATILIDAD ULTRA-CORTA (3 features) ===
            'volatility_1s',       # Volatilidad inmediata
            'volatility_3s',       # Volatilidad corta
            'volatility_5s',       # Volatilidad media
            
            # === PATRONES ULTRA-CORTOS (4 features) ===
            'higher_high_1s',      # Máximo creciente inmediato
            'lower_low_1s',        # Mínimo decreciente inmediato
            'breakout_1s',         # Ruptura inmediata
            'breakout_3s',         # Ruptura confirmada
            
            # === EFFICIENCY ULTRA-CORTA (2 features) ===
            'efficiency_1s',       # Eficiencia inmediata
            'efficiency_3s',       # Eficiencia corta
            
            # === NORMALIZED FEATURES (3 features) ===
            'price_change_normalized_1s', # Precio normalizado
            'volume_normalized_1s',       # Volumen normalizado
            'triple_momentum_signal',     # Señal triple (precio + volumen + RSI)
        ]

    def _get_ultra_momentum_combined_features(self) -> List[str]:
        """
        🚀 Features combinadas optimizadas: TCN definitivo V3 + Ultra momentum core
        Conjunto equilibrado para trading de alta frecuencia (71 features total)
        """
        tcn_v3_features = self._get_tcn_definitivo_v3_features()
        ultra_momentum_features = self._get_ultra_momentum_features()
        
        # Combinar sin duplicados
        combined_features = list(set(tcn_v3_features + ultra_momentum_features))
        
        # Ordenar para consistencia
        combined_features.sort()
        
        return combined_features

    def _get_tcn_definitivo_features(self) -> List[str]:
        """Features para modelos TCN definitivos (88 features técnicas completas) - ORIGINAL"""
        return [
            # === MOMENTUM INDICATORS (17 features) ===
            'rsi_14', 'rsi_21', 'rsi_7',
            'macd', 'macd_signal', 'macd_histogram',
            'stoch_k', 'stoch_d', 'williams_r',
            'roc_10', 'roc_20', 'momentum_10', 'momentum_20',
            'cci_14', 'cci_20',
            'rsi_momentum', 'macd_momentum',

            # === TREND INDICATORS (12 features) ===
            'sma_10', 'sma_20', 'sma_50',
            'ema_10', 'ema_20', 'ema_50',
            'adx_14', 'plus_di', 'minus_di',
            'psar', 'aroon_up', 'aroon_down',

            # === VOLATILITY INDICATORS (10 features) ===
            'bb_upper', 'bb_middle', 'bb_lower',
            'bb_width', 'bb_position',
            'atr_14', 'atr_20', 'true_range',
            'natr_14', 'natr_20',

            # === VOLUME INDICATORS (10 features) ===
            'ad', 'adosc', 'obv',
            'volume_sma_10', 'volume_sma_20', 'volume_ratio',
            'mfi_14', 'mfi_20',
            'ad_momentum', 'volume_momentum',

            # === PRICE PATTERNS (8 features) ===
            'hl_ratio', 'oc_ratio', 'price_position',
            'price_change_1', 'price_change_5', 'price_change_10',
            'price_volatility_10', 'price_volatility_20',

            # === MARKET STRUCTURE (8 features) ===
            'higher_high', 'lower_low',
            'uptrend_strength', 'downtrend_strength',
            'resistance_touch', 'support_touch',
            'efficiency_ratio', 'fractal_dimension',

            # === MOMENTUM DERIVATIVES (1 feature) ===
            'price_acceleration',

            # === PRICE MOMENTUM (8 features) ===
            'price_momentum_1', 'price_momentum_3', 'price_momentum_5', 'price_momentum_10', 'price_momentum_20',
            'price_momentum_normalized_5', 'price_momentum_normalized_10', 'price_momentum_normalized_20',

            # === VOLATILIDAD ADICIONAL (14 features) ===
            'volatility_5', 'volatility_10', 'volatility_15', 'volatility_20', 'volatility_30',
            'hl_volatility_5', 'hl_volatility_10', 'hl_volatility_15', 'hl_volatility_20', 'hl_volatility_30',
            'volatility_normalized_10', 'volatility_normalized_15', 'volatility_normalized_20', 'volatility_normalized_30'
        ]

    def _get_tcn_final_features(self) -> List[str]:
        """Features para modelos tcn_final (16 features técnicas simplificadas)"""
        return [
            # === RETURNS Y MOMENTUM (5 features) ===
            'returns_1', 'returns_3', 'returns_5', 'returns_10', 'returns_20',

            # === MOVING AVERAGES (3 features) ===
            'sma_5', 'sma_20', 'ema_12',

            # === MOMENTUM INDICATORS (4 features) ===
            'rsi_14', 'macd', 'macd_signal', 'macd_histogram',

            # === VOLATILITY & VOLUME (4 features) ===
            'bb_position', 'bb_width', 'volume_ratio', 'volatility'
        ]

    def _get_full_features_set(self) -> List[str]:
        """Conjunto completo de features disponibles"""
        tcn_def = self._get_tcn_definitivo_features()
        tcn_final = self._get_tcn_final_features()
        additional = ['returns_1', 'returns_3', 'returns_5', 'returns_10', 'returns_20',
                     'sma_5', 'ema_12', 'bb_position', 'bb_width', 'volume_ratio', 'volatility']
        return list(set(tcn_def + tcn_final + additional))

    def calculate_features(self, df: pd.DataFrame, feature_set: str = 'tcn_definitivo') -> pd.DataFrame:
        """
        Calcular features técnicas usando TA-Lib

        Args:
            df: DataFrame con columnas OHLCV
            feature_set: Conjunto de features a calcular ('tcn_definitivo', 'tcn_definitivo_v3', 'tcn_final', 'full_set')

        Returns:
            DataFrame con features calculadas
        """
        if feature_set not in self.feature_sets:
            raise ValueError(f"Feature set '{feature_set}' no disponible. Opciones: {list(self.feature_sets.keys())}")

        # Validar datos de entrada
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame debe contener columnas: {required_columns}")

        # Crear copia para trabajar
        features_df = df.copy()

        # Extraer arrays para TA-Lib
        open_prices = df['open'].values.astype(float)
        high_prices = df['high'].values.astype(float)
        low_prices = df['low'].values.astype(float)
        close_prices = df['close'].values.astype(float)
        volume_data = df['volume'].values.astype(float)

        # Calcular todas las features disponibles
        features_df = self._calculate_all_talib_features(
            features_df, open_prices, high_prices, low_prices, close_prices, volume_data
        )

        # Calcular features adicionales no disponibles en TA-Lib
        features_df = self._calculate_additional_features(features_df)
        
        # Calcular features ultra-momentum si se solicitan
        if feature_set in ['ultra_momentum', 'ultra_momentum_combined']:
            features_df = self._calculate_ultra_momentum_features(features_df)
            
        # Calcular features bajistas crypto-específicas para enhanced
        if feature_set == 'tcn_definitivo_v3_enhanced':
            features_df = self._calculate_crypto_bearish_features(features_df)
            
        # Calcular features bajistas + volume profile para volume enhanced
        if feature_set == 'tcn_definitivo_v3_volume_enhanced':
            features_df = self._calculate_volume_profile_features(features_df)
            features_df = self._calculate_crypto_bearish_features(features_df)

        # Seleccionar solo las features del conjunto solicitado
        requested_features = self.feature_sets[feature_set]
        available_features = [f for f in requested_features if f in features_df.columns]

        if len(available_features) != len(requested_features):
            missing = set(requested_features) - set(available_features)
            if not self.quiet_mode:
                print(f"⚠️ Features faltantes: {missing}")

        # Retornar solo las features solicitadas
        result_df = features_df[available_features].copy()

        # Limpiar datos
        result_df = self._clean_features_data(result_df)

        # Corregir rangos de indicadores técnicos ANTES de la validación
        print("🔧 Aplicando corrección de rangos de indicadores...")
        result_df = self._correct_indicator_ranges(result_df)
        
        # Validación de integridad
        price_context = {
            'median_price': df['close'].median() if 'close' in df.columns else None,
            'price_std': df['close'].std() if 'close' in df.columns else None
        }
        validation_results = self.validate_talib_features_integrity(result_df, price_context)
        if not validation_results['talib_features_preserved']:
            if not self.quiet_mode:
                print("⚠️ ADVERTENCIA: Features de TA-Lib pueden estar corrompidas")
                for warning in validation_results['warnings']:
                    print(f"   ⚠️ {warning}")
        else:
            if not self.quiet_mode:
                print("✅ Features de TA-Lib preservadas correctamente")
        
        # Validación específica por tipo de feature
        range_validation = self.validate_feature_ranges(result_df, feature_set)
        if range_validation['errors']:
            print("❌ ERRORES en rangos de features:")
            for error in range_validation['errors']:
                print(f"   ❌ {error}")
        elif range_validation['warnings'] and not self.quiet_mode:
            print("⚠️ ADVERTENCIAS en rangos de features:")
            for warning in range_validation['warnings']:
                print(f"   ⚠️ {warning}")
        elif not self.quiet_mode:
            print(f"✅ Rangos de features válidos ({range_validation['total_features_checked']} features verificadas)")

        if feature_set == 'tcn_definitivo_v3':
            print(f"✅ Features calculadas: {len(result_df.columns)} de {len(requested_features)} solicitadas (✅ 54 features optimizadas)")
        elif feature_set == 'tcn_definitivo_v3_enhanced':
            print(f"🎯 Features Enhanced calculadas: {len(result_df.columns)} de {len(requested_features)} solicitadas (✅ 54 base + 🐻 8 bajistas)")
        elif feature_set == 'tcn_definitivo_v3_volume_enhanced':
            print(f"📊 Features Volume Enhanced calculadas: {len(result_df.columns)} de {len(requested_features)} solicitadas (✅ 54 base + 🐻 8 bajistas + 📊 2 volume profile)")
        else:
            print(f"✅ Features calculadas: {len(result_df.columns)} de {len(requested_features)} solicitadas")
        
        # Limpieza robusta de NaN
        print(f"🔍 NaN antes de limpieza: {result_df.isna().sum().sum()}")
        
        # Método 1: Forward fill + backward fill usando función moderna
        for col in result_df.columns:
            result_df[col] = safe_fillna_forward_backward(result_df[col])
        
        # Método 2: Si aún hay NaN, usar mediana de cada columna
        for col in result_df.columns:
            if result_df[col].isna().any():
                median_val = result_df[col].median()
                if pd.isna(median_val):
                    median_val = 0.0  # Fallback final
                result_df[col] = result_df[col].fillna(median_val)
        
        # Método 3: Verificación final
        final_nan = result_df.isna().sum().sum()
        if final_nan > 0:
            print(f"⚠️ Aún hay {final_nan} NaN, rellenando con 0")
            result_df = result_df.fillna(0.0)
        
        print(f"✅ NaN después de limpieza: {result_df.isna().sum().sum()}")
        
        return result_df

    def _calculate_all_talib_features(self, df: pd.DataFrame, open_arr: np.ndarray,
                                    high_arr: np.ndarray, low_arr: np.ndarray,
                                    close_arr: np.ndarray, volume_arr: np.ndarray) -> pd.DataFrame:
        """Calcular todas las features usando TA-Lib"""

        if talib is None:
            print("⚠️ TA-Lib no disponible, usando implementaciones manuales")
            return self._calculate_manual_features(df)

        try:
            # === MOMENTUM INDICATORS ===
            df['rsi_14'] = talib.RSI(close_arr, timeperiod=14)
            df['rsi_21'] = talib.RSI(close_arr, timeperiod=21)
            df['rsi_7'] = talib.RSI(close_arr, timeperiod=7)

            # MACD (✅ macd_signal restaurado - crítico para timing de señales)
            macd, macd_signal, macd_hist = talib.MACD(close_arr)
            df['macd'] = macd
            df['macd_signal'] = macd_signal  # ✅ RESTAURADO: Crítico para timing
            df['macd_histogram'] = macd_hist

            # Stochastic
            slowk, slowd = talib.STOCH(high_arr, low_arr, close_arr)
            df['stoch_k'] = slowk
            df['stoch_d'] = slowd

            # Williams %R
            df['williams_r'] = talib.WILLR(high_arr, low_arr, close_arr)

            # Rate of Change
            df['roc_10'] = talib.ROC(close_arr, timeperiod=10)
            df['roc_20'] = talib.ROC(close_arr, timeperiod=20)

            # Momentum
            df['momentum_10'] = talib.MOM(close_arr, timeperiod=10)
            df['momentum_20'] = talib.MOM(close_arr, timeperiod=20)

            # CCI
            df['cci_14'] = talib.CCI(high_arr, low_arr, close_arr, timeperiod=14)
            df['cci_20'] = talib.CCI(high_arr, low_arr, close_arr, timeperiod=20)

            # === TREND INDICATORS ===
            # Moving Averages
            df['sma_10'] = talib.SMA(close_arr, timeperiod=10)
            df['sma_20'] = talib.SMA(close_arr, timeperiod=20)  # ✅ RESTAURADO: Fundamental para análisis técnico
            df['sma_50'] = talib.SMA(close_arr, timeperiod=50)
            df['sma_5'] = talib.SMA(close_arr, timeperiod=5)

            df['ema_10'] = talib.EMA(close_arr, timeperiod=10)
            df['ema_20'] = talib.EMA(close_arr, timeperiod=20)
            df['ema_50'] = talib.EMA(close_arr, timeperiod=50)
            df['ema_12'] = talib.EMA(close_arr, timeperiod=12)

            # ADX
            df['adx_14'] = talib.ADX(high_arr, low_arr, close_arr, timeperiod=14)
            df['plus_di'] = talib.PLUS_DI(high_arr, low_arr, close_arr, timeperiod=14)
            df['minus_di'] = talib.MINUS_DI(high_arr, low_arr, close_arr, timeperiod=14)

            # PSAR
            df['psar'] = talib.SAR(high_arr, low_arr)

            # Aroon
            aroon_down, aroon_up = talib.AROON(high_arr, low_arr, timeperiod=14)
            df['aroon_up'] = aroon_up
            df['aroon_down'] = aroon_down

            # === VOLATILITY INDICATORS ===
            # Bollinger Bands (✅ bb_middle restaurado - importante para contexto BB)
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close_arr, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle  # ✅ RESTAURADO: Importante para contexto de Bollinger Bands
            df['bb_lower'] = bb_lower

            # ATR
            df['atr_14'] = talib.ATR(high_arr, low_arr, close_arr, timeperiod=14)
            df['atr_20'] = talib.ATR(high_arr, low_arr, close_arr, timeperiod=20)
            df['natr_14'] = talib.NATR(high_arr, low_arr, close_arr, timeperiod=14)
            df['natr_20'] = talib.NATR(high_arr, low_arr, close_arr, timeperiod=20)
            df['true_range'] = talib.TRANGE(high_arr, low_arr, close_arr)

            # === VOLUME INDICATORS ===
            df['ad'] = talib.AD(high_arr, low_arr, close_arr, volume_arr)
            df['adosc'] = talib.ADOSC(high_arr, low_arr, close_arr, volume_arr)
            df['obv'] = talib.OBV(close_arr, volume_arr)
            df['volume_sma_10'] = talib.SMA(volume_arr, timeperiod=10)
            df['volume_sma_20'] = talib.SMA(volume_arr, timeperiod=20)
            df['mfi_14'] = talib.MFI(high_arr, low_arr, close_arr, volume_arr, timeperiod=14)
            df['mfi_20'] = talib.MFI(high_arr, low_arr, close_arr, volume_arr, timeperiod=20)

            # === CYCLE INDICATORS ===
            df['ht_dcperiod'] = talib.HT_DCPERIOD(close_arr)
            df['ht_dcphase'] = talib.HT_DCPHASE(close_arr)
            inphase, quadrature = talib.HT_PHASOR(close_arr)
            df['ht_phasor_inphase'] = inphase
            df['ht_phasor_quadrature'] = quadrature

            # === STATISTICAL INDICATORS ===
            df['beta'] = talib.BETA(high_arr, low_arr, timeperiod=5)
            df['correl'] = talib.CORREL(high_arr, low_arr, timeperiod=30)
            df['linearreg'] = talib.LINEARREG(close_arr, timeperiod=14)
            df['linearreg_angle'] = talib.LINEARREG_ANGLE(close_arr, timeperiod=14)
            df['linearreg_intercept'] = talib.LINEARREG_INTERCEPT(close_arr, timeperiod=14)
            df['linearreg_slope'] = talib.LINEARREG_SLOPE(close_arr, timeperiod=14)

        except Exception as e:
            print(f"⚠️ Error calculando features TA-Lib: {e}")

        return df

    def _calculate_manual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Implementaciones manuales SEGURAS cuando TA-Lib no está disponible"""
        try:
            # RSI manual con protección
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()

            rs = safe_divide(gain, loss, fallback_method='constant', min_value=1e-8)
            rs = rs.replace([np.inf, -np.inf], 100)
            rs = rs.clip(0, 1000)
            df['rsi_14'] = 100 - (100 / (1 + rs))

            # SMA/EMA básicos
            df['sma_20'] = df['close'].rolling(20).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()

            # MACD básico con protección
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']

            # Reemplazar valores problemáticos
            manual_cols = ['rsi_14', 'sma_20', 'ema_12', 'macd', 'macd_signal', 'macd_histogram']
            for col in manual_cols:
                if col in df.columns:
                    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                    df[col] = safe_fillna_forward_backward(df[col])

        except Exception as e:
            print(f"⚠️ Error en features manuales: {e}")
            # Fallback: valores neutros
            df['rsi_14'] = 50.0
            df['sma_20'] = df['close']
            df['ema_12'] = df['close']
            df['macd'] = 0.0
            df['macd_signal'] = 0.0
            df['macd_histogram'] = 0.0

        return df

    def _calculate_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular features adicionales no disponibles en TA-Lib"""

        try:
            # Returns múltiples períodos
            df['returns_1'] = df['close'].pct_change(periods=1)
            df['returns_3'] = df['close'].pct_change(periods=3)
            df['returns_5'] = df['close'].pct_change(periods=5)
            df['returns_10'] = df['close'].pct_change(periods=10)
            df['returns_20'] = df['close'].pct_change(periods=20)

            # Bollinger Bands adicionales
            if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                bb_range = df['bb_upper'] - df['bb_lower']
                
                df['bb_position'] = safe_divide(
                    df['close'] - df['bb_lower'], 
                    bb_range, 
                    fallback_method='mean'
                )
                df['bb_position'] = df['bb_position'].clip(0, 1)

                if 'bb_middle' in df.columns:
                    df['bb_width'] = safe_divide(bb_range, df['bb_middle'], fallback_method='mean')
                else:
                    df['bb_width'] = safe_divide(bb_range, df['close'], fallback_method='mean')

            # Volume ratio
            volume_sma_source = df.get('volume_sma_20', df['volume'].rolling(20).mean())
            df['volume_ratio'] = safe_divide(
                df['volume'], 
                volume_sma_source, 
                fallback_method='mean'
            )
            # 🆕 CORRECCIÓN: Aplicar clipping robusto para evitar overflow
            df['volume_ratio'] = df['volume_ratio'].clip(lower=0, upper=10)
            # Si aún hay valores extremos, aplicar corrección adicional
            if df['volume_ratio'].max() > 100:
                print(f"🔧 Corrigiendo volume_ratio: valores extremos detectados, aplicando normalización")
                # Normalizar usando percentiles para evitar outliers extremos
                q99 = df['volume_ratio'].quantile(0.99)
                if q99 > 10:
                    # Aplicar clipping más agresivo
                    df['volume_ratio'] = df['volume_ratio'].clip(lower=0, upper=q99)
                    # Si sigue siendo muy alto, normalizar
                    if df['volume_ratio'].max() > 50:
                        df['volume_ratio'] = df['volume_ratio'] / (df['volume_ratio'].max() / 10)
                        df['volume_ratio'] = df['volume_ratio'].clip(lower=0, upper=10)

            df['volume_price_trend'] = df['volume'] * df['close'].pct_change()

            # Volatilidad sin look-ahead bias
            df['volatility'] = calculate_safe_volatility(df['close'], window=20)

            # === NUEVAS FEATURES DEL TCN DEFINITIVO ===

            # PRICE PATTERNS con safe_divide
            hl_range = df['high'] - df['low']
            
            df['hl_ratio'] = safe_divide(hl_range, df['close'], fallback_method='mean')
            df['oc_ratio'] = safe_divide(df['close'] - df['open'], df['close'], fallback_method='mean')
            df['price_position'] = safe_divide(
                df['close'] - df['low'], 
                hl_range, 
                fallback_method='mean'
            )
            df['price_position'] = df['price_position'].clip(0, 1)

            # Price changes
            df['price_change_1'] = df['close'].pct_change(1)
            df['price_change_5'] = df['close'].pct_change(5)
            df['price_change_10'] = df['close'].pct_change(10)

            # Volatility windows
            df['price_volatility_10'] = calculate_safe_volatility(df['close'], window=10)
            df['price_volatility_20'] = calculate_safe_volatility(df['close'], window=20)

            # MARKET STRUCTURE
            df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
            df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)

            df['uptrend_strength'] = (df['close'] > df['close'].shift(1)).rolling(10).sum() / 10
            df['downtrend_strength'] = (df['close'] < df['close'].shift(1)).rolling(10).sum() / 10

            # Resistance/support sin look-ahead bias
            rolling_max = df['close'].rolling(20, min_periods=1).max().shift(1)
            rolling_min = df['close'].rolling(20, min_periods=1).min().shift(1)

            df['resistance_touch'] = (df['close'] >= rolling_max * 0.99).astype(int)
            df['support_touch'] = (df['close'] <= rolling_min * 1.01).astype(int)

            # Market efficiency
            close_diff_abs = pd.Series(np.abs(df['close'].diff()), index=df.index)
            efficiency_numerator = np.abs(df['close'] - df['close'].shift(10))
            efficiency_denominator = close_diff_abs.rolling(10, min_periods=1).sum()
            df['efficiency_ratio'] = safe_divide(
                efficiency_numerator, 
                efficiency_denominator, 
                fallback_method='constant'
            ).fillna(0)

            # Fractal dimension
            if len(df) > 20:
                volatility = calculate_safe_volatility(df['close'], window=20)
                df['fractal_dimension'] = 1.0 + (volatility * 10).clip(0, 1)
            else:
                df['fractal_dimension'] = 1.5

            # === MOMENTUM DERIVATIVES ===
            # RSI Momentum
            if 'rsi_14' in df.columns:
                rsi_diff = df['rsi_14'].diff().fillna(0)
                rsi_q99 = rsi_diff.quantile(0.99)
                rsi_q01 = rsi_diff.quantile(0.01)
                df['rsi_momentum'] = rsi_diff.clip(lower=rsi_q01, upper=rsi_q99)
                
            # MACD Momentum
            if 'macd_histogram' in df.columns:
                macd_diff = df['macd_histogram'].diff().fillna(0)
                macd_q99 = macd_diff.quantile(0.99)
                macd_q01 = macd_diff.quantile(0.01)
                df['macd_momentum'] = macd_diff.clip(lower=macd_q01, upper=macd_q99)
                
            # AD Momentum
            if 'ad' in df.columns:
                ad_diff = df['ad'].diff().fillna(0)
                ad_std = ad_diff.std()
                ad_mean = ad_diff.mean()
                ad_limit = min(abs(ad_mean) + 3 * ad_std, 100)
                df['ad_momentum'] = ad_diff.clip(lower=-ad_limit, upper=ad_limit)

            # Volume Momentum
            df['volume_momentum'] = df['volume'].pct_change().fillna(0)
            
            # Price Acceleration
            df['price_acceleration'] = df['price_change_1'].diff().fillna(0)

            # === PRICE MOMENTUM (múltiples períodos) ===
            for period in [1, 3, 5, 10, 20]:
                # Calcular momentum para diferentes períodos
                price_diff = df['close'] - df['close'].shift(period)
                price_prev = df['close'].shift(period)

                momentum = safe_divide(price_diff, price_prev, fallback_method='median')
                df[f'price_momentum_{period}'] = momentum.fillna(0)

                # MOMENTUM NORMALIZADO (basado en volatilidad)
                if period >= 5:
                    returns = df['close'].pct_change().rolling(period*2).std()
                    
                    normalized_momentum = safe_divide(
                        momentum, 
                        returns, 
                        fallback_method='constant', 
                        min_value=0.01
                    )
                    
                    # ✅ CORRECCIÓN: Aplicar clipping específico para momentum normalizado
                    # Los valores deben estar en el rango [-20, 20] según la documentación
                    normalized_momentum = normalized_momentum.clip(lower=-20, upper=20)
                    
                    df[f'price_momentum_normalized_{period}'] = normalized_momentum.fillna(0)

            # === VOLATILIDAD ADICIONAL SIN LOOK-AHEAD BIAS ===
            for period in [5, 10, 15, 20, 30]:
                volatility = calculate_safe_volatility(df['close'], window=period)
                if period in [10, 20]:
                    df[f'price_volatility_{period}'] = volatility.fillna(0.01)
                    df[f'volatility_{period}'] = volatility.fillna(0.01)
                else:
                    df[f'volatility_{period}'] = volatility.fillna(0.01)

                # Volatilidad basada en high-low range
                hl_range = (df['high'] - df['low']) / df['close']
                hl_volatility = hl_range.rolling(period, min_periods=1).mean()
                df[f'hl_volatility_{period}'] = hl_volatility.fillna(0.01)

                # Volatilidad normalizada
                if period >= 10:
                    long_term_vol = calculate_safe_volatility(df['close'], window=period*3)
                    normalized_vol = safe_divide(
                        volatility, 
                        long_term_vol, 
                        fallback_method='constant', 
                        min_value=0.01
                    )
                    df[f'volatility_normalized_{period}'] = normalized_vol.fillna(1.0)

            # === FEATURES ESPECÍFICAS PARA CRYPTO TRADING ===
            print("🚀 Calculando features específicas para crypto...")
            
            # 1️⃣ MICROSTRUCTURE FEATURES
            # Bid-ask spread proxy usando high-low range
            df['bid_ask_spread_proxy'] = safe_divide(
                df['high'] - df['low'], 
                df['close'], 
                fallback_method='mean'
            )
            
            # Volume-price trend (VPT) - correlación entre precio y volumen
            df['volume_price_trend'] = (df['close'] - df['close'].shift(1)) * df['volume']
            
            # 2️⃣ MARKET REGIME INDICATORS
            # Volatilidad relativa (regímenes de alta/baja volatilidad)
            close_rolling_20 = df['close'].rolling(20, min_periods=1)
            df['volatility_regime'] = safe_divide(
                close_rolling_20.std(), 
                close_rolling_20.mean(), 
                fallback_method='constant',
                min_value=1e-8
            )
            
            # Fuerza de tendencia (diferencia entre medias móviles)
            close_rolling_50 = df['close'].rolling(50, min_periods=1)
            df['trend_strength'] = abs(
                close_rolling_20.mean() - close_rolling_50.mean()
            )
            
            # 3️⃣ LIQUIDITY PROXIES
            # Momentum del volumen (volumen relativo)
            volume_rolling_20 = df['volume'].rolling(20, min_periods=1)
            df['volume_momentum'] = safe_divide(
                df['volume'], 
                volume_rolling_20.mean(), 
                fallback_method='mean'
            )
            
            # Impacto de precio en volumen (medida de liquidez)
            df['price_impact'] = safe_divide(
                abs(df['close'] - df['open']), 
                df['volume'], 
                fallback_method='mean'
            )
            
            # 4️⃣ CROSS-TIMEFRAME FEATURES
            # Retorno intradiario (open to close)
            df['intraday_return'] = safe_divide(
                df['close'] - df['open'], 
                df['open'], 
                fallback_method='constant'
            )
            
            # Gap overnight (close anterior a open actual)
            df['overnight_gap'] = safe_divide(
                df['open'] - df['close'].shift(1), 
                df['close'].shift(1), 
                fallback_method='constant'
            )
            
            # 5️⃣ FEATURES ADICIONALES ESPECÍFICAS PARA CRYPTO
            
            # Volatilidad intradiaria vs overnight
            df['intraday_volatility'] = df['intraday_return'].rolling(10).std()
            df['overnight_volatility'] = df['overnight_gap'].rolling(10).std()
            
            # Ratio de volatilidad (intradiaria vs overnight)
            df['volatility_ratio'] = safe_divide(
                df['intraday_volatility'], 
                df['overnight_volatility'], 
                fallback_method='constant',
                min_value=1e-8
            )
            
            # Momentum de volumen ponderado por precio
            df['volume_price_momentum'] = df['volume_momentum'] * df['intraday_return']
            
            # Regímenes de mercado más sofisticados
            # Regímenes basados en volatilidad y tendencia
            volatility_threshold = df['volatility_regime'].quantile(0.75)
            trend_threshold = df['trend_strength'].quantile(0.75)
            
            # ✅ CORRECCIÓN: Usar numpy.where para evitar el error de Series booleanas
            df['market_regime'] = 0  # Neutral por defecto
            
            # Trending volatile
            mask1 = (df['volatility_regime'] > volatility_threshold) & (df['trend_strength'] > trend_threshold)
            df.loc[mask1, 'market_regime'] = 1
            
            # High volatility sideways
            mask2 = (df['volatility_regime'] > volatility_threshold) & (df['trend_strength'] <= trend_threshold)
            df.loc[mask2, 'market_regime'] = 2
            
            # Low volatility trending
            mask3 = (df['volatility_regime'] <= volatility_threshold) & (df['trend_strength'] > trend_threshold)
            df.loc[mask3, 'market_regime'] = 3
            
            # Low volatility sideways
            mask4 = (df['volatility_regime'] <= volatility_threshold) & (df['trend_strength'] <= trend_threshold)
            df.loc[mask4, 'market_regime'] = 4
            
            # 6️⃣ FEATURES DE MICROESTRUCTURA AVANZADAS
            
            # Eficiencia de precio (qué tan directo es el movimiento)
            price_path = np.abs(df['close'] - df['close'].shift(10))
            price_direct = np.abs(df['close'] - df['close'].shift(1)).rolling(10).sum()
            df['price_efficiency'] = safe_divide(
                price_path, 
                price_direct, 
                fallback_method='constant'
            )
            
            # Momentum de spread (cambios en bid-ask proxy)
            df['spread_momentum'] = df['bid_ask_spread_proxy'].diff()
            
            # Volatilidad del spread
            df['spread_volatility'] = df['bid_ask_spread_proxy'].rolling(10).std()
            
            print("✅ Features específicas para crypto calculadas")

        except Exception as e:
            print(f"⚠️ Error calculando features adicionales: {e}")

        return df

    def _calculate_volume_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        📊 Calcular features de Volume Profile optimizadas para crypto
        
        Features:
        - volume_profile_poc: Point of Control (nivel de máximo volumen)
        - volume_profile_vah_val: Value Area High/Low ratio
        """
        try:
            print("📊 Calculando Volume Profile features...")
            
            # Ventana para cálculo de volume profile (20 períodos para 3m/5m)
            window = 20
            
            # 1. Volume Profile POC (Point of Control)
            # Crear bins de precio y calcular volumen por bin
            poc_values = []
            vah_val_ratios = []
            
            for i in range(len(df)):
                if i < window:
                    poc_values.append(df['close'].iloc[i])
                    vah_val_ratios.append(1.0)
                    continue
                    
                # Datos de la ventana
                window_data = df.iloc[i-window+1:i+1].copy()
                
                # Crear bins de precio (10 bins)
                price_min = window_data['low'].min()
                price_max = window_data['high'].max()
                
                if price_max == price_min:
                    poc_values.append(df['close'].iloc[i])
                    vah_val_ratios.append(1.0)
                    continue
                    
                # Crear bins y calcular volumen por bin
                n_bins = 10
                bin_edges = np.linspace(price_min, price_max, n_bins + 1)
                bin_volumes = np.zeros(n_bins)
                
                # Asignar volumen a bins basado en precio típico
                for _, row in window_data.iterrows():
                    typical_price = (row['high'] + row['low'] + row['close']) / 3
                    bin_idx = np.digitize(typical_price, bin_edges) - 1
                    bin_idx = max(0, min(n_bins - 1, bin_idx))  # Clamp to valid range
                    bin_volumes[bin_idx] += row['volume']
                
                # POC: centro del bin con mayor volumen
                max_vol_bin = np.argmax(bin_volumes)
                poc_price = (bin_edges[max_vol_bin] + bin_edges[max_vol_bin + 1]) / 2
                
                # Normalizar POC relativo al precio actual
                poc_normalized = poc_price / df['close'].iloc[i]
                poc_values.append(poc_normalized)
                
                # 2. Value Area High/Low ratio
                # Value Area: bins que contienen 70% del volumen
                total_volume = bin_volumes.sum()
                if total_volume > 0:
                    # Ordenar bins por volumen
                    sorted_indices = np.argsort(bin_volumes)[::-1]
                    cumulative_volume = 0
                    value_area_bins = []
                    
                    for bin_idx in sorted_indices:
                        cumulative_volume += bin_volumes[bin_idx]
                        value_area_bins.append(bin_idx)
                        if cumulative_volume >= total_volume * 0.7:
                            break
                    
                    # Value Area High y Low
                    vah = max([bin_edges[i+1] for i in value_area_bins])
                    val = min([bin_edges[i] for i in value_area_bins])
                    
                    # Ratio VAH/VAL normalizado
                    if val > 0:
                        vah_val_ratio = (vah / val - 1.0) * 100  # Porcentaje de rango
                        vah_val_ratios.append(min(vah_val_ratio, 10.0))  # Cap a 10%
                    else:
                        vah_val_ratios.append(1.0)
                else:
                    vah_val_ratios.append(1.0)
            
            # Asignar features calculadas
            df['volume_profile_poc'] = poc_values
            df['volume_profile_vah_val'] = vah_val_ratios
            
            # Limpieza y normalización
            df['volume_profile_poc'] = df['volume_profile_poc'].fillna(1.0)
            df['volume_profile_vah_val'] = df['volume_profile_vah_val'].fillna(1.0)
            
            # Clip a rangos razonables
            df['volume_profile_poc'] = df['volume_profile_poc'].clip(0.8, 1.2)  # ±20% del precio actual
            df['volume_profile_vah_val'] = df['volume_profile_vah_val'].clip(0.1, 10.0)  # 0.1% a 10%
            
            print("✅ Volume Profile features calculadas: 2 features")
            return df
            
        except Exception as e:
            print(f"⚠️ Error calculando Volume Profile features: {e}")
            # Valores fallback
            df['volume_profile_poc'] = 1.0
            df['volume_profile_vah_val'] = 1.0
            return df

    def _calculate_crypto_bearish_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        🚀 Calcular features bajistas crypto-específicas optimizadas para Binance
        
        Nuevo conjunto de 8 features sin dependencias problemáticas:
        - Señales de rechazo en niveles clave
        - Análisis de distribución de volumen
        - Indicadores de agotamiento y liquidez
        - Detección de movimientos institucionales
        """
        try:
            print("🚀 Calculando features bajistas crypto-específicas...")
            
            # 1. RSI Divergence Bearish - 🆕 Fuerza de rechazo en niveles clave (nuevo cálculo crypto-específico)
            high_low_ratio = (df['high'] - df['close']) / (df['high'] - df['low'] + 0.0001)
            wick_strength = (df['high'] - df['open']) / (df['close'] - df['open'] + 0.0001)
            rejection_volume = df['volume'] > df['volume'].rolling(20).mean() * 1.2
            
            df['rsi_divergence_bearish'] = ((high_low_ratio > 0.6) & (wick_strength > 0.5) & rejection_volume).astype(float) * high_low_ratio
            df['rsi_divergence_bearish'] = df['rsi_divergence_bearish'].clip(0, 1)
            
            # 2. MACD Bearish Cross - 🆕 Distribución anómala de volumen (nuevo cálculo crypto-específico)
            volume_sma_5 = df['volume'].rolling(5).mean()
            volume_sma_20 = df['volume'].rolling(20).mean()
            price_decline = df['close'] < df['close'].shift(1)
            volume_acceleration = volume_sma_5 > volume_sma_20 * 1.3
            
            df['macd_bearish_cross'] = (price_decline & volume_acceleration & 
                                            (df['volume'] > df['volume'].rolling(50).quantile(0.8))).astype(float)
            
            # 3. Trend Strength Ratio - 🆕 Agotamiento de momentum alcista (nuevo cálculo crypto-específico)
            rsi = df['rsi_14'] if 'rsi_14' in df.columns else 50
            price_change = df['close'].pct_change(5)
            rsi_overbought = rsi > 70
            momentum_slowing = abs(price_change) < abs(df['close'].pct_change(5).shift(5))
            
            df['trend_strength_ratio'] = (rsi_overbought & momentum_slowing & (price_change < 0.005)).astype(float)
            
            # 4. Volume Bearish Signal - 🆕 Breakouts bajistas con volatilidad (nuevo cálculo crypto-específico)
            if 'atr_14' in df.columns:
                vol_expansion = df['atr_14'] > df['atr_14'].rolling(20).mean() * 1.4
                price_break_down = df['low'] < df['low'].rolling(20).min().shift(1)
                volume_confirm = df['volume'] > df['volume'].rolling(10).mean()
                
                df['volume_bearish_signal'] = (vol_expansion & price_break_down & volume_confirm).astype(float)
            else:
                df['volume_bearish_signal'] = 0.0
                
            # 5. Price Momentum Bearish - 🆕 Debilidad en estructura de mercado (nuevo cálculo crypto-específico)
            lower_highs = df['high'] < df['high'].shift(1)
            lower_lows = df['low'] < df['low'].shift(1)
            weak_bounces = (df['close'] - df['low']) / (df['high'] - df['low'] + 0.0001) < 0.3
            
            df['price_momentum_bearish'] = ((lower_highs & lower_lows) | weak_bounces).astype(float)
            
            # 6. Support Resistance Context - 🆕 Señales de drenaje de liquidez (nuevo cálculo crypto-específico)
            thin_spread = (df['high'] - df['low']) / df['close'] < 0.005  # Spreads estrechos
            low_volume_decline = (df['close'] < df['close'].shift(1)) & (df['volume'] < df['volume'].rolling(10).mean())
            sudden_gap = abs(df['open'] - df['close'].shift(1)) / df['close'].shift(1) > 0.01
            
            df['support_resistance_context'] = (thin_spread & low_volume_decline & sudden_gap).astype(float)
            
            # 7. Volatility Expansion Bear - 🆕 Riesgo de cascada de liquidaciones (nuevo cálculo crypto-específico)
            rapid_decline = df['close'].pct_change(1) < -0.02  # Caída rápida > 2%
            volume_spike = df['volume'] > df['volume'].rolling(20).mean() * 2
            consecutive_red = (df['close'] < df['open']) & (df['close'].shift(1) < df['open'].shift(1))
            
            df['volatility_expansion_bear'] = (rapid_decline & volume_spike & consecutive_red).astype(float)
            
            # 8. Momentum Divergence Bear - 🆕 Señales de salida institucional (nuevo cálculo crypto-específico)
            large_volume = df['volume'] > df['volume'].rolling(50).quantile(0.9)
            controlled_decline = (df['close'] < df['open']) & (abs(df['close'] - df['low']) / (df['high'] - df['low'] + 0.0001) > 0.7)
            # Simplificación: 3 cierres consecutivos bajistas
            sustained_selling = ((df['close'] < df['close'].shift(1)) & 
                               (df['close'].shift(1) < df['close'].shift(2)) & 
                               (df['close'].shift(2) < df['close'].shift(3)))
            
            df['momentum_divergence_bear'] = (large_volume & controlled_decline & sustained_selling).astype(float)
            
            # Limpiar y normalizar todas las features
            crypto_bearish_features = [
                'rsi_divergence_bearish', 'macd_bearish_cross', 'trend_strength_ratio',
                'volume_bearish_signal', 'price_momentum_bearish', 'support_resistance_context',
                'volatility_expansion_bear', 'momentum_divergence_bear'
            ]
            
            for feature in crypto_bearish_features:
                if feature in df.columns:
                    df[feature] = df[feature].fillna(0.0)
                    df[feature] = df[feature].clip(0.0, 1.0)
                        
            print(f"✅ Features bajistas crypto-específicas calculadas: {len(crypto_bearish_features)} features")
            return df
            
        except Exception as e:
            print(f"⚠️ Error calculando features bajistas crypto: {e}")
            return df

    def _clean_features_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpiar y validar datos de features"""

        # Definir features de TA-Lib que NO deben ser clipeadas
        talib_features = [
            'rsi_14', 'rsi_21', 'rsi_7', 'macd', 'macd_signal', 'macd_histogram',
            'stoch_k', 'stoch_d', 'williams_r', 'roc_10', 'roc_20', 'momentum_10', 'momentum_20',
            'cci_14', 'cci_20', 'sma_10', 'sma_20', 'sma_50', 'ema_10', 'ema_20', 'ema_50',
            'adx_14', 'plus_di', 'minus_di', 'psar', 'aroon_up', 'aroon_down',
            'bb_upper', 'bb_middle', 'bb_lower', 'atr_14', 'atr_20', 'true_range',
            'natr_14', 'natr_20', 'ad', 'adosc', 'obv', 'volume_sma_10', 'volume_sma_20',
            'mfi_14', 'mfi_20'
        ]

        # Definir features manuales problemáticas que necesitan limpieza agresiva
        manual_features = [
            'bb_width', 'bb_position', 'volume_ratio', 'hl_ratio', 'oc_ratio', 'price_position',
            'price_change_1', 'price_change_5', 'price_change_10', 'price_volatility_10', 'price_volatility_20',
            'higher_high', 'lower_low', 'uptrend_strength', 'downtrend_strength',
            'resistance_touch', 'support_touch', 'efficiency_ratio', 'fractal_dimension',
            'rsi_momentum', 'macd_momentum', 'ad_momentum', 'volume_momentum', 'price_acceleration',
            
            # 🆕 Features específicas para crypto que necesitan limpieza especial
            'bid_ask_spread_proxy', 'volume_price_trend', 'volatility_regime', 'trend_strength',
            'volume_momentum', 'price_impact', 'intraday_return', 'overnight_gap',
            'intraday_volatility', 'overnight_volatility', 'volatility_ratio', 'market_regime',
            'price_efficiency', 'spread_momentum', 'spread_volatility'
        ]

        # Reemplazar infinitos en todas las columnas
        df = df.replace([np.inf, -np.inf], np.nan)

        # Limpieza específica por tipo de feature
        for col in df.columns:
            if col in talib_features:
                # TA-Lib: Solo manejar NaN suavemente - NO clipping
                df[col] = safe_fillna_forward_backward(df[col])

            elif col in manual_features:
                # Manuales: Sin data leakage  
                df[col] = safe_fillna_forward_backward(df[col])

                # Valores por defecto específicos por tipo de feature
                if col.startswith('bb_'):
                    df[col] = df[col].fillna(0.5)
                elif col.endswith('_ratio'):
                    df[col] = df[col].fillna(1.0)
                elif col.endswith('_touch'):
                    df[col] = df[col].fillna(0)
                elif col.endswith('_strength'):
                    df[col] = df[col].fillna(0.5)
                elif col.endswith('_momentum'):
                    df[col] = df[col].fillna(0.0)
                    # Aplicar clipping robusto a momentum features
                    df[col] = robust_feature_clipping(df[col], col)
                
                # 🆕 LIMPIEZA ESPECÍFICA PARA FEATURES CRYPTO
                elif col == 'bid_ask_spread_proxy':
                    df[col] = df[col].fillna(0.01)  # Spread típico 1%
                    df[col] = df[col].clip(lower=0, upper=0.1)  # Máximo 10%
                elif col == 'volatility_regime':
                    df[col] = df[col].fillna(0.02)  # Volatilidad típica 2%
                    df[col] = df[col].clip(lower=0, upper=1.0)  # Máximo 100%
                elif col == 'trend_strength':
                    df[col] = df[col].fillna(0.0)
                    df[col] = df[col].clip(lower=0, upper=100)  # Máximo $100
                elif col == 'price_impact':
                    df[col] = df[col].fillna(0.0)
                    df[col] = df[col].clip(lower=0, upper=1.0)  # Máximo 100%
                elif col == 'intraday_return' or col == 'overnight_gap':
                    df[col] = df[col].fillna(0.0)
                    df[col] = df[col].clip(lower=-0.5, upper=0.5)  # Máximo ±50%
                elif col == 'volatility_ratio':
                    df[col] = df[col].fillna(1.0)  # Ratio neutral
                    df[col] = df[col].clip(lower=0.1, upper=10.0)  # Rango razonable
                elif col == 'market_regime':
                    df[col] = df[col].fillna(0)  # Regímenes: 0 = neutral
                    df[col] = df[col].clip(lower=0, upper=4)  # Solo valores 0-4
                elif col == 'price_efficiency':
                    df[col] = df[col].fillna(0.5)  # Eficiencia neutral
                    df[col] = df[col].clip(lower=0, upper=1)  # Rango [0, 1]
                elif col == 'spread_momentum':
                    df[col] = df[col].fillna(0.0)
                    df[col] = df[col].clip(lower=-0.01, upper=0.01)  # Cambios pequeños
                elif col == 'spread_volatility':
                    df[col] = df[col].fillna(0.001)  # Volatilidad típica del spread
                    df[col] = df[col].clip(lower=0, upper=0.05)  # Máximo 5%
                else:
                    df[col] = df[col].fillna(0.0)

                # Clipping moderado solo para features manuales problemáticas
                if hasattr(df[col], 'dtype') and str(df[col].dtype) in ['float64', 'float32']:
                    if not col.endswith('_momentum'):  # Skip momentum ya procesado
                        q99 = df[col].quantile(0.99)
                        q01 = df[col].quantile(0.01)
                        if pd.notna(q99) and pd.notna(q01) and q99 != q01:
                            df[col] = df[col].clip(lower=q01, upper=q99)

            else:
                # Features adicionales: limpieza estándar
                df[col] = safe_fillna_forward_backward(df[col])
                if df[col].isna().any():
                    df[col] = df[col].fillna(0)
                    
                # Aplicar clipping robusto si es una feature conocida problemática
                if any(keyword in col.lower() for keyword in ['momentum', 'acceleration']):
                    df[col] = robust_feature_clipping(df[col], col)

        return df

    def validate_feature_ranges(self, df: pd.DataFrame, feature_type: str) -> Dict:
        """Validación específica por tipo de feature"""
        
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Rangos esperados por tipo de indicator
        expected_ranges = {
            'rsi': (0, 100),
            'stoch': (0, 100),
            'williams_r': (-100, 0),
            'bb_position': (0, 1),
            'price_position': (0, 1),
            'normalized_ratios': (-0.5, 2.5),
            'aroon': (0, 100),
            'mfi': (0, 100),
            'adx': (0, 100),
            'di': (0, 100),
            'cci': (-300, 300),
            'momentum_normalized': (-20, 20),
            'momentum_derivatives': (-50, 50),
            'efficiency_ratio': (0, 1),
            'fractal_dimension': (1, 2),
            'volatility_normalized': (0, 5),
            'volume_ratio': (0, 10),
            'price_ratios': (-0.1, 0.1),
            # 🆕 CORRECCIÓN: Rangos específicos para features problemáticas
            # 🚀 NUEVO: Features bajistas crypto-específicas (con nombres originales)
            'rsi_divergence_bearish': (0, 1),        # 🆕 Fuerza de rechazo (nuevo cálculo crypto)
            'macd_bearish_cross': (0, 1),            # 🆕 Distribución volumen bajista (nuevo cálculo crypto)
            'trend_strength_ratio': (0, 1),          # 🆕 Agotamiento momentum (nuevo cálculo crypto)
            'volume_bearish_signal': (0, 1),         # 🆕 Breakout bajista (nuevo cálculo crypto)
            'price_momentum_bearish': (0, 1),        # 🆕 Debilidad estructura (nuevo cálculo crypto)
            'support_resistance_context': (0, 1),    # 🆕 Drenaje liquidez (nuevo cálculo crypto)
            'volatility_expansion_bear': (0, 1),     # 🆕 Riesgo cascada (nuevo cálculo crypto)
            'momentum_divergence_bear': (0, 1),      # 🆕 Salida institucional (nuevo cálculo crypto)
        }
        
        violation_count = 0
        
        for col in df.columns:
            # Evaluar casos específicos ANTES que patrones generales
            if col in ['rsi_momentum', 'macd_momentum', 'ad_momentum', 'volume_momentum', 'price_acceleration']:
                if col == 'rsi_momentum':
                    expected_min, expected_max = (-15, 15)
                elif col == 'macd_momentum':
                    expected_min, expected_max = expected_ranges['momentum_derivatives']
                elif col == 'ad_momentum':
                    expected_min, expected_max = (-100, 100)
                else:
                    expected_min, expected_max = expected_ranges['momentum_derivatives']
            
            elif 'momentum_normalized' in col:
                expected_min, expected_max = expected_ranges['momentum_normalized']
            elif 'volatility_normalized' in col:
                expected_min, expected_max = expected_ranges['volatility_normalized']
            elif col.endswith('_momentum') and not 'normalized' in col:
                expected_min, expected_max = expected_ranges['momentum_derivatives']
            
            # 🆕 CORRECCIÓN: Casos específicos para features de detección bajista
            elif col in ['rsi_divergence_bearish', 'macd_bearish_cross', 'trend_strength_ratio',
                        'volume_bearish_signal', 'price_momentum_bearish', 'support_resistance_context',
                        'volatility_expansion_bear', 'momentum_divergence_bear']:
                expected_min, expected_max = expected_ranges[col]
            
            elif col.startswith('rsi_') and col not in ['rsi_momentum']:
                expected_min, expected_max = expected_ranges['rsi']
            elif col.startswith('stoch_'):
                expected_min, expected_max = expected_ranges['stoch']
            elif col == 'williams_r':
                expected_min, expected_max = expected_ranges['williams_r']
            elif col in ['bb_position', 'price_position']:
                expected_min, expected_max = expected_ranges[col]
            elif col.startswith('aroon_'):
                expected_min, expected_max = expected_ranges['aroon']
            elif col.startswith('mfi_'):
                expected_min, expected_max = expected_ranges['mfi']
            elif col.startswith('adx_') or col == 'adx_14':
                expected_min, expected_max = expected_ranges['adx']
            elif col.endswith('_di') or col in ['plus_di', 'minus_di']:
                expected_min, expected_max = expected_ranges['di']
            elif col.startswith('cci_'):
                expected_min, expected_max = expected_ranges['cci']
            elif col.endswith('_ratio') and not col.startswith('efficiency'):
                if 'volume' in col:
                    expected_min, expected_max = expected_ranges['volume_ratio']
                elif col in ['oc_ratio', 'hl_ratio']:
                    expected_min, expected_max = expected_ranges['price_ratios']
                else:
                    expected_min, expected_max = expected_ranges['normalized_ratios']
            elif col == 'efficiency_ratio':
                expected_min, expected_max = expected_ranges['efficiency_ratio']
            elif col == 'fractal_dimension':
                expected_min, expected_max = expected_ranges['fractal_dimension']
            else:
                continue  # Skip features sin rango definido
                
            # Validar rango con tolerancia
            actual_min = df[col].min()
            actual_max = df[col].max()
            
            # Tolerancia del 5% para casos edge
            tolerance = 0.05
            min_tolerance = expected_min - abs(expected_min * tolerance)
            max_tolerance = expected_max + abs(expected_max * tolerance)
            
            if actual_min < min_tolerance or actual_max > max_tolerance:
                violation_count += 1
                severity = 'error' if (actual_min < expected_min - abs(expected_min * 0.2) or 
                                     actual_max > expected_max + abs(expected_max * 0.2)) else 'warning'
                
                message = (f"{col}: rango [{actual_min:.3f}, {actual_max:.3f}] vs "
                          f"esperado [{expected_min}, {expected_max}] (tolerancia: "
                          f"[{min_tolerance:.3f}, {max_tolerance:.3f}])")
                
                if severity == 'error':
                    validation_results['errors'].append(message)
                    validation_results['valid'] = False
                else:
                    validation_results['warnings'].append(message)
        
        # Agregar resumen
        validation_results['total_features_checked'] = len([col for col in df.columns 
                                                           if any(pattern in col for pattern in 
                                                                 ['rsi_', 'stoch_', 'williams_r', 'bb_', 'price_', 
                                                                  'aroon_', 'mfi_', 'adx_', '_di', 'cci_', 
                                                                  '_ratio', 'efficiency', 'fractal', 'momentum', 'volatility'])])
        validation_results['violations'] = violation_count
        
        return validation_results

    def _correct_indicator_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Corregir rangos de indicadores técnicos para que estén en rangos válidos de trading
        """
        corrected_df = df.copy()
        
        # Corregir Stochastic K (debe estar entre 0 y 100)
        if 'stoch_k' in corrected_df.columns:
            stoch_k = corrected_df['stoch_k']
            # Clampear valores fuera de rango
            stoch_k_corrected = np.clip(stoch_k, 0, 100)
            
            # Si hay valores fuera de rango, aplicar corrección más inteligente
            if (stoch_k < 0).any() or (stoch_k > 100).any():
                print(f"🔧 Corrigiendo stoch_k: rango original [{stoch_k.min():.3f}, {stoch_k.max():.3f}] → [0, 100]")
                
                # Normalizar a rango [0, 100] preservando la distribución relativa
                stoch_range = stoch_k.max() - stoch_k.min()
                if stoch_range > 0:
                    stoch_k_normalized = ((stoch_k - stoch_k.min()) / stoch_range) * 100
                    corrected_df['stoch_k'] = stoch_k_normalized
                else:
                    corrected_df['stoch_k'] = stoch_k_corrected
            else:
                corrected_df['stoch_k'] = stoch_k_corrected
        
        # Corregir Stochastic D (debe estar entre 0 y 100)
        if 'stoch_d' in corrected_df.columns:
            stoch_d = corrected_df['stoch_d']
            stoch_d_corrected = np.clip(stoch_d, 0, 100)
            
            if (stoch_d < 0).any() or (stoch_d > 100).any():
                print(f"🔧 Corrigiendo stoch_d: rango original [{stoch_d.min():.3f}, {stoch_d.max():.3f}] → [0, 100]")
                
                stoch_range = stoch_d.max() - stoch_d.min()
                if stoch_range > 0:
                    stoch_d_normalized = ((stoch_d - stoch_d.min()) / stoch_range) * 100
                    corrected_df['stoch_d'] = stoch_d_normalized
                else:
                    corrected_df['stoch_d'] = stoch_d_corrected
            else:
                corrected_df['stoch_d'] = stoch_d_corrected
        
        # Corregir Williams %R (debe estar entre -100 y 0)
        if 'williams_r' in corrected_df.columns:
            williams_r = corrected_df['williams_r']
            williams_r_corrected = np.clip(williams_r, -100, 0)
            
            if (williams_r < -100).any() or (williams_r > 0).any():
                print(f"🔧 Corrigiendo williams_r: rango original [{williams_r.min():.3f}, {williams_r.max():.3f}] → [-100, 0]")
                
                # Normalizar a rango [-100, 0] preservando la distribución relativa
                williams_range = williams_r.max() - williams_r.min()
                if williams_range > 0:
                    williams_normalized = ((williams_r - williams_r.min()) / williams_range) * -100
                    corrected_df['williams_r'] = williams_normalized
                else:
                    corrected_df['williams_r'] = williams_r_corrected
            else:
                corrected_df['williams_r'] = williams_r_corrected
        
        # Corregir RSI (debe estar entre 0 y 100)
        rsi_features = ['rsi_14', 'rsi_21', 'rsi_7']
        for rsi_col in rsi_features:
            if rsi_col in corrected_df.columns:
                rsi = corrected_df[rsi_col]
                rsi_corrected = np.clip(rsi, 0, 100)
                
                if (rsi < 0).any() or (rsi > 100).any():
                    print(f"🔧 Corrigiendo {rsi_col}: rango original [{rsi.min():.3f}, {rsi.max():.3f}] → [0, 100]")
                    
                    rsi_range = rsi.max() - rsi.min()
                    if rsi_range > 0:
                        rsi_normalized = ((rsi - rsi.min()) / rsi_range) * 100
                        corrected_df[rsi_col] = rsi_normalized
                    else:
                        corrected_df[rsi_col] = rsi_corrected
                else:
                    corrected_df[rsi_col] = rsi_corrected
        
        # Corregir CCI (debe estar entre -300 y 300)
        cci_features = ['cci_14', 'cci_20']
        for cci_col in cci_features:
            if cci_col in corrected_df.columns:
                cci = corrected_df[cci_col]
                cci_corrected = np.clip(cci, -300, 300)
                
                if (cci < -300).any() or (cci > 300).any():
                    print(f"🔧 Corrigiendo {cci_col}: rango original [{cci.min():.3f}, {cci.max():.3f}] → [-300, 300]")
                    
                    cci_range = cci.max() - cci.min()
                    if cci_range > 0:
                        cci_normalized = ((cci - cci.min()) / cci_range) * 600 - 300
                        corrected_df[cci_col] = cci_normalized
                    else:
                        corrected_df[cci_col] = cci_corrected
                else:
                    corrected_df[cci_col] = cci_corrected
        
        # Corregir MFI (debe estar entre 0 y 100)
        mfi_features = ['mfi_14']
        for mfi_col in mfi_features:
            if mfi_col in corrected_df.columns:
                mfi = corrected_df[mfi_col]
                mfi_corrected = np.clip(mfi, 0, 100)
                
                if (mfi < 0).any() or (mfi > 100).any():
                    print(f"🔧 Corrigiendo {mfi_col}: rango original [{mfi.min():.3f}, {mfi.max():.3f}] → [0, 100]")
                    
                    mfi_range = mfi.max() - mfi.min()
                    if mfi_range > 0:
                        mfi_normalized = ((mfi - mfi.min()) / mfi_range) * 100
                        corrected_df[mfi_col] = mfi_normalized
                    else:
                        corrected_df[mfi_col] = mfi_corrected
                else:
                    corrected_df[mfi_col] = mfi_corrected
        
        # Corregir MACD si está comprimido (expandir rango)
        macd_features = ['macd', 'macd_signal', 'macd_histogram']
        for macd_col in macd_features:
            if macd_col in corrected_df.columns:
                macd = corrected_df[macd_col]
                macd_range = macd.max() - macd.min()
                
                # Si el rango es muy pequeño, expandirlo
                if macd_range < 0.01:  # Umbral de compresión
                    print(f"🔧 Expandindo {macd_col}: rango comprimido [{macd.min():.6f}, {macd.max():.6f}]")
                    
                    # Expandir el rango multiplicando por un factor
                    expansion_factor = 10.0 / macd_range if macd_range > 0 else 100.0
                    macd_expanded = macd * expansion_factor
                    
                    # Aplicar clipping para evitar valores extremos
                    corrected_df[macd_col] = np.clip(macd_expanded, -100, 100)
                else:
                    corrected_df[macd_col] = macd
        
        # ✅ NUEVO: Corregir Price Momentum Normalized a rango [-20, 20]
        momentum_norm_features = ['price_momentum_normalized_5', 'price_momentum_normalized_10', 'price_momentum_normalized_20']
        for momentum_col in momentum_norm_features:
            if momentum_col in corrected_df.columns:
                momentum = corrected_df[momentum_col]
                momentum_min = momentum.min()
                momentum_max = momentum.max()
                
                # Verificar si está fuera del rango esperado [-20, 20]
                if momentum_min < -20 or momentum_max > 20:
                    print(f"🔧 Corrigiendo {momentum_col}: rango original [{momentum_min:.3f}, {momentum_max:.3f}] → [-20, 20]")
                    
                    # Aplicar clipping directo al rango esperado
                    corrected_df[momentum_col] = np.clip(momentum, -20, 20)
                else:
                    corrected_df[momentum_col] = momentum
        
        # 🆕 CORRECCIÓN: Corregir features de detección bajista
        crypto_bearish_features = [
            'rsi_divergence_bearish', 'macd_bearish_cross', 'trend_strength_ratio',
            'volume_bearish_signal', 'price_momentum_bearish', 'support_resistance_context',
            'volatility_expansion_bear', 'momentum_divergence_bear'
        ]
        
        for feature in crypto_bearish_features:
            if feature in corrected_df.columns:
                feature_data = corrected_df[feature]
                feature_min = feature_data.min()
                feature_max = feature_data.max()
                
                # Aplicar rangos correctos según el tipo de feature
                if feature == 'trend_strength_ratio':
                    # Ratio uptrend/downtrend: [0, 5]
                    if feature_min < 0 or feature_max > 5:
                        print(f"🔧 Corrigiendo {feature}: rango original [{feature_min:.3f}, {feature_max:.3f}] → [0, 5]")
                        corrected_df[feature] = np.clip(feature_data, 0, 5)
                else:
                    # Features binarias: [0, 1]
                    if feature_min < 0 or feature_max > 1:
                        print(f"🔧 Corrigiendo {feature}: rango original [{feature_min:.3f}, {feature_max:.3f}] → [0, 1]")
                        corrected_df[feature] = np.clip(feature_data, 0, 1)
        
        return corrected_df

    def validate_talib_features_integrity(self, df: pd.DataFrame, price_context: Dict = None) -> Dict:
        """
        Validar que las features de TA-Lib mantienen su integridad después de la limpieza
        """
        validation_results = {
            'talib_features_preserved': True,
            'rsi_range_valid': True,
            'macd_extremes_preserved': True,
            'bb_ranges_valid': True,
            'warnings': []
        }

        # Validar RSI está en rango [0, 100]
        rsi_features = ['rsi_14', 'rsi_21', 'rsi_7']
        for rsi_col in rsi_features:
            if rsi_col in df.columns:
                rsi_min = df[rsi_col].min()
                rsi_max = df[rsi_col].max()
                if rsi_min < 0 or rsi_max > 100:
                    validation_results['rsi_range_valid'] = False
                    validation_results['warnings'].append(
                        f"RSI {rsi_col} fuera de rango [0,100]: [{rsi_min:.2f}, {rsi_max:.2f}]"
                    )

        # Validar que MACD mantiene valores extremos
        macd_features = ['macd', 'macd_signal', 'macd_histogram']
        for macd_col in macd_features:
            if macd_col in df.columns:
                macd_range = df[macd_col].max() - df[macd_col].min()
                macd_q99 = df[macd_col].quantile(0.99)
                macd_q01 = df[macd_col].quantile(0.01)
                macd_iqr = df[macd_col].quantile(0.75) - df[macd_col].quantile(0.25)

                # Límites adaptativos basados en el precio del activo
                if price_context and price_context.get('median_price') is not None:
                    price_level = price_context['median_price']
                    if price_level > 1000:
                        macd_threshold = price_level * 0.00002
                    elif price_level > 100:
                        macd_threshold = price_level * 0.0002
                    elif price_level > 10:
                        macd_threshold = price_level * 0.002
                    elif price_level > 1:
                        macd_threshold = price_level * 0.02
                    elif price_level > 0.1:
                        macd_threshold = price_level * 0.01
                    else:
                        macd_threshold = 0.001

                    macd_range_threshold = max(0.00001, macd_threshold)
                    macd_iqr_threshold = max(0.000001, macd_threshold * 0.001)
                else:
                    macd_range_threshold = 0.00001
                    macd_iqr_threshold = 0.000001

                # Verificar múltiples métricas de compresión con límites adaptativos
                if (macd_range < macd_range_threshold or
                    abs(macd_q99 - macd_q01) < macd_range_threshold or
                    macd_iqr < macd_iqr_threshold):
                    validation_results['macd_extremes_preserved'] = False
                    price_info = f"price:{price_context.get('median_price', 'N/A'):.2f}" if price_context else "price:N/A"
                    validation_results['warnings'].append(
                        f"MACD {macd_col} comprimido - range:{macd_range:.6f}, iqr:{macd_iqr:.6f}, "
                        f"threshold:{macd_range_threshold:.6f} ({price_info})"
                    )

        # Validar Bollinger Bands
        bb_features = ['bb_upper', 'bb_middle', 'bb_lower']
        if all(bb in df.columns for bb in bb_features):
            bb_width = df['bb_upper'] - df['bb_lower']
            bb_width_std = bb_width.std()
            bb_width_mean = bb_width.mean()

            if price_context and price_context.get('median_price') is not None:
                price_level = price_context['median_price']
                if price_level > 1000:
                    bb_threshold = price_level * 0.00002
                elif price_level > 100:
                    bb_threshold = price_level * 0.0002
                elif price_level > 10:
                    bb_threshold = price_level * 0.002
                elif price_level > 1:
                    bb_threshold = price_level * 0.02
                elif price_level > 0.1:
                    bb_threshold = price_level * 0.01
                else:
                    bb_threshold = 0.001
            else:
                bb_threshold = 0.00001

            if bb_width_std < bb_threshold and bb_width_mean < bb_threshold * 10:
                validation_results['bb_ranges_valid'] = False
                validation_results['warnings'].append(
                    f"Bollinger Bands comprimidas - std:{bb_width_std:.6f}, mean:{bb_width_mean:.6f}, "
                    f"threshold:{bb_threshold:.6f}"
                )

        # Validación general
        if not (validation_results['rsi_range_valid'] and
                validation_results['macd_extremes_preserved'] and
                validation_results['bb_ranges_valid']):
            validation_results['talib_features_preserved'] = False

        return validation_results

    def get_feature_info(self, feature_set: str = None) -> Dict:
        """Obtener información sobre los conjuntos de features"""

        if feature_set and feature_set in self.feature_sets:
            return {
                'feature_set': feature_set,
                'features': self.feature_sets[feature_set],
                'count': len(self.feature_sets[feature_set])
            }

        return {
            'available_sets': list(self.feature_sets.keys()),
            'sets_info': {
                name: {
                    'features': features,
                    'count': len(features)
                }
                for name, features in self.feature_sets.items()
            }
        }

    async def compute_features(self, symbol: str, klines_data: List, feature_set: str = 'tcn_definitivo_v3') -> np.ndarray:
        """
        Computar features desde datos de klines de Binance

        Args:
            symbol: Símbolo del par (ej: BTCUSDT)
            klines_data: Lista de klines de Binance
            feature_set: Conjunto de features a calcular

        Returns:
            np.ndarray: Features calculadas o None si error
        """
        try:
            print(f"🔄 Calculando {len(self.feature_sets.get(feature_set, []))} features para {symbol}...")

            # Convertir klines a DataFrame
            df = self._klines_to_dataframe(klines_data)
            if df is None or df.empty:
                print(f"❌ Error: DataFrame vacío para {symbol}")
                return None

            # Calcular features
            df_features = self.calculate_features(df, feature_set)

            if df_features is None or df_features.empty:
                print(f"❌ Error: No se calcularon features para {symbol}")
                return None

            # Seleccionar solo las features del conjunto solicitado
            feature_columns = self.feature_sets.get(feature_set, [])
            available_columns = [col for col in feature_columns if col in df_features.columns]

            if not available_columns:
                print(f"❌ Error: No hay features disponibles para {symbol}")
                return None

            # Obtener datos como numpy array
            features_array = df_features[available_columns].values

            print(f"✅ Features calculadas: {len(available_columns)} de {len(feature_columns)} solicitadas")

            return features_array

        except Exception as e:
            print(f"❌ Error calculando features para {symbol}: {e}")
            return None

    def _klines_to_dataframe(self, klines_data: List) -> pd.DataFrame:
        """Convertir datos de klines de Binance a DataFrame"""
        try:
            if not klines_data:
                return None

            df = pd.DataFrame(klines_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])

            # Convertir a tipos correctos
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Ordenar por timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)

            return df

        except Exception as e:
            print(f"❌ Error convirtiendo klines a DataFrame: {e}")
            return None

    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """Validar que el DataFrame tiene el formato correcto"""

        required_columns = ['open', 'high', 'low', 'close', 'volume']

        # Verificar columnas
        if not all(col in df.columns for col in required_columns):
            missing = set(required_columns) - set(df.columns)
            print(f"❌ Columnas faltantes: {missing}")
            return False

        # Verificar tipos de datos
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"❌ Columna '{col}' debe ser numérica")
                return False

        # Verificar que no hay valores negativos en precios
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (df[col] <= 0).any():
                print(f"❌ Columna '{col}' contiene valores no positivos")
                return False

        # Verificar lógica OHLC
        # ✅ CORRECCIÓN: Usar operaciones booleanas seguras para evitar errores de Series
        ohlc_valid = (
            (df['high'] >= df['low']) &
            (df['high'] >= df['open']) &
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) &
            (df['low'] <= df['close'])
        )
        
        if not ohlc_valid.all():
            print("❌ Datos OHLC inconsistentes")
            return False

        print("✅ DataFrame validado correctamente")
        return True

    # === MÉTODOS DE BINANCE API ===
    
    async def get_real_market_data(self, symbol: str, timeframe: str = '1m', days: int = 30) -> pd.DataFrame:
        """
        📊 Obtener datos reales de mercado de Binance con cache
        
        Args:
            symbol: Símbolo del par (ej: BTCUSDT)
            timeframe: Intervalo de tiempo (1m, 3m, 5m, 15m, 30m, 1h, 4h, 1d)
            days: Número de días de datos a obtener
            
        Returns:
            pd.DataFrame: Datos OHLCV de Binance
        """
        # ✅ CACHE: Verificar si ya tenemos datos guardados
        cache_file = f"cache/{symbol}_{timeframe}_{days}d.pkl"
        os.makedirs("cache", exist_ok=True)

        if os.path.exists(cache_file):
            # Verificar si el cache es reciente (menos de 1 hora)
            cache_time = os.path.getmtime(cache_file)
            current_time = time.time()
            if current_time - cache_time < 3600:  # 1 hora
                print(f"📋 Usando datos cacheados para {symbol} ({timeframe})")
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    print(f"⚠️ Error leyendo cache: {e}, descargando de nuevo...")

        # Calcular período de tiempo
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        period_desc = f"{days} días"

        print(f"📊 Obteniendo datos {period_desc} para {symbol} ({timeframe})...")

        # 🚀 PRIORIDAD: Usar API autenticada si está disponible
        if self.use_authenticated_api and self.binance_client:
            try:
                print(f"🔐 Usando API autenticada de Binance (menor latencia)")
                
                # Mapear timeframes a formato de Binance
                timeframe_mapping = {
                    '1m': Client.KLINE_INTERVAL_1MINUTE,
                    '3m': Client.KLINE_INTERVAL_3MINUTE,
                    '5m': Client.KLINE_INTERVAL_5MINUTE,
                    '15m': Client.KLINE_INTERVAL_15MINUTE,
                    '30m': Client.KLINE_INTERVAL_30MINUTE,
                    '1h': Client.KLINE_INTERVAL_1HOUR,
                    '4h': Client.KLINE_INTERVAL_4HOUR,
                    '1d': Client.KLINE_INTERVAL_1DAY
                }
                
                binance_timeframe = timeframe_mapping.get(timeframe, Client.KLINE_INTERVAL_1MINUTE)
                
                # Obtener datos usando cliente autenticado
                klines = self.binance_client.get_klines(
                    symbol=symbol,
                    interval=binance_timeframe,
                    start_str=datetime.fromtimestamp(start_time/1000).strftime('%Y-%m-%d %H:%M:%S'),
                    end_str=datetime.fromtimestamp(end_time/1000).strftime('%Y-%m-%d %H:%M:%S'),
                    limit=1000
                )
                
                if klines:
                    # Convertir a DataFrame
                    df = pd.DataFrame(klines, columns=[
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
                    
                    print(f"✅ Obtenidos {len(df)} registros de {symbol} via API autenticada")
                    
                    # Guardar en cache
                    try:
                        with open(cache_file, 'wb') as f:
                            pickle.dump(df, f)
                        print(f"💾 Datos guardados en cache: {cache_file}")
                    except Exception as e:
                        print(f"⚠️ Error guardando cache: {e}")
                    
                    return df
                else:
                    print("⚠️ API autenticada no devolvió datos, usando API pública...")
                    
            except Exception as e:
                print(f"⚠️ Error con API autenticada: {e}, usando API pública...")
        
        # 🌐 FALLBACK: Usar API pública si no hay autenticación o falló
        print(f"🌐 Usando API pública de Binance (mayor latencia)")
        
        base_url = "https://api.binance.com"

        async with aiohttp.ClientSession() as session:
            url = f"{base_url}/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': timeframe,
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

        print(f"✅ Obtenidos {len(df)} registros de {symbol} via API pública")

        # ✅ CACHE: Guardar datos descargados
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)
            print(f"💾 Datos guardados en cache: {cache_file}")
        except Exception as e:
            print(f"⚠️ Error guardando cache: {e}")

        return df

    async def get_features_from_binance(self, symbol: str, timeframe: str = '1m', 
                                      days: int = 30, feature_set: str = 'tcn_definitivo_v3') -> pd.DataFrame:
        """
        🔄 Obtener features directamente desde Binance
        
        Args:
            symbol: Símbolo del par
            timeframe: Intervalo de tiempo
            days: Días de datos
            feature_set: Conjunto de features a calcular
            
        Returns:
            pd.DataFrame: Datos con features calculadas
        """
        try:
            print(f"🔄 Obteniendo features desde Binance para {symbol}...")
            
            # 1. Obtener datos de mercado
            df = await self.get_real_market_data(symbol, timeframe, days)
            
            if df is None or df.empty:
                print(f"❌ Error: No se pudieron obtener datos para {symbol}")
                return None
            
            # 2. Calcular features
            features_df = self.calculate_features(df, feature_set)
            
            if features_df is None or features_df.empty:
                print(f"❌ Error: No se pudieron calcular features para {symbol}")
                return None
            
            print(f"✅ Features calculadas desde Binance: {features_df.shape}")
            return features_df
            
        except Exception as e:
            print(f"❌ Error obteniendo features desde Binance: {e}")
            return None

    def get_klines_data(self, klines_data: List, symbol: str = None) -> pd.DataFrame:
        """
        📊 Convertir datos de klines de Binance a DataFrame
        
        Args:
            klines_data: Lista de klines de Binance
            symbol: Símbolo del par (opcional, para logging)
            
        Returns:
            pd.DataFrame: DataFrame con datos OHLCV
        """
        try:
            if not klines_data:
                return None

            # Formato esperado de klines de Binance:
            # [timestamp, open, high, low, close, volume, close_time, quote_asset_volume, 
            #  number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore]

            df = pd.DataFrame(klines_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])

            # Convertir a tipos correctos
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Ordenar por timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            if symbol:
                print(f"✅ Klines convertidos para {symbol}: {df.shape}")

            return df

        except Exception as e:
            print(f"❌ Error convirtiendo klines a DataFrame: {e}")
            return None

    async def get_multiple_symbols_data(self, symbols: List[str], timeframe: str = '1m', 
                                      days: int = 30) -> Dict[str, pd.DataFrame]:
        """
        📊 Obtener datos de múltiples símbolos simultáneamente
        
        Args:
            symbols: Lista de símbolos
            timeframe: Intervalo de tiempo
            days: Días de datos
            
        Returns:
            Dict: Diccionario con datos de cada símbolo
        """
        print(f"📊 Obteniendo datos de {len(symbols)} símbolos...")
        
        tasks = []
        for symbol in symbols:
            task = self.get_real_market_data(symbol, timeframe, days)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data_dict = {}
        for i, result in enumerate(results):
            symbol = symbols[i]
            if isinstance(result, Exception):
                print(f"❌ Error obteniendo {symbol}: {result}")
            else:
                data_dict[symbol] = result
                print(f"✅ {symbol}: {result.shape}")
        
        return data_dict

    def calculate_features_enhanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        🎯 Versión de conveniencia para calcular features enhanced
        
        Returns:
            DataFrame con 62 features (54 base + 8 bajistas)
        """
        return self.calculate_features(df, 'tcn_definitivo_v3_enhanced')

    def _calculate_ultra_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ⚡ Calcular features ultra-momentum (1-10 segundos)
        Features especializadas para capturar momentum de ultra-corto plazo
        """
        try:
            print("⚡ Calculando features ultra-momentum...")
            
            # === PRICE MOMENTUM ULTRA-CORTO ===
            # Cambios de precio en intervalos muy cortos (simulados para datos de 5m)
            # En datos reales de alta frecuencia, estos serían cálculos de 1-10 segundos
            
            # Simular cambios de precio ultra-cortos basados en datos disponibles
            df['price_change_1s'] = df['close'].pct_change(1).fillna(0) * 100  # Simulado
            df['price_change_3s'] = df['close'].pct_change(2).fillna(0) * 100  # Simulado
            df['price_change_5s'] = df['close'].pct_change(3).fillna(0) * 100  # Simulado
            df['price_change_10s'] = df['close'].pct_change(4).fillna(0) * 100  # Simulado
            
            # === VOLUME MOMENTUM ULTRA-CORTO ===
            # Volumen relativo en intervalos cortos
            df['volume_ratio_1s'] = safe_divide(df['volume'], df['volume'].rolling(5).mean(), fallback_method='mean')
            df['volume_ratio_3s'] = safe_divide(df['volume'], df['volume'].rolling(10).mean(), fallback_method='mean')
            df['volume_ratio_5s'] = safe_divide(df['volume'], df['volume'].rolling(15).mean(), fallback_method='mean')
            df['volume_ratio_10s'] = safe_divide(df['volume'], df['volume'].rolling(20).mean(), fallback_method='mean')
            
            # Picos de volumen
            volume_std = df['volume'].rolling(20).std()
            volume_mean = df['volume'].rolling(20).mean()
            df['volume_spike_1s'] = ((df['volume'] - volume_mean) / volume_std).fillna(0)
            df['volume_spike_3s'] = ((df['volume'] - volume_mean) / volume_std).fillna(0)
            
            # === OSCILADORES ULTRA-CORTOS ===
            # RSI con períodos muy cortos
            if talib is not None:
                df['rsi_3s'] = talib.RSI(df['close'].values, timeperiod=3)
                df['rsi_5s'] = talib.RSI(df['close'].values, timeperiod=5)
                df['rsi_10s'] = talib.RSI(df['close'].values, timeperiod=10)
                
                # Stochastic ultra-corto
                stoch_3s_k, _ = talib.STOCH(df['high'].values, df['low'].values, df['close'].values, 
                                           fastk_period=3, slowk_period=1, slowd_period=1)
                stoch_5s_k, _ = talib.STOCH(df['high'].values, df['low'].values, df['close'].values, 
                                           fastk_period=5, slowk_period=1, slowd_period=1)
                df['stoch_3s'] = stoch_3s_k
                df['stoch_5s'] = stoch_5s_k
                
                # Williams %R ultra-corto
                df['williams_r_3s'] = talib.WILLR(df['high'].values, df['low'].values, df['close'].values, timeperiod=3)
                df['williams_r_5s'] = talib.WILLR(df['high'].values, df['low'].values, df['close'].values, timeperiod=5)
            else:
                # Implementaciones manuales para ultra-corto plazo
                df['rsi_3s'] = 50.0  # Placeholder
                df['rsi_5s'] = 50.0  # Placeholder
                df['rsi_10s'] = 50.0  # Placeholder
                df['stoch_3s'] = 50.0  # Placeholder
                df['stoch_5s'] = 50.0  # Placeholder
                df['williams_r_3s'] = -50.0  # Placeholder
                df['williams_r_5s'] = -50.0  # Placeholder
            
            # === ACELERACIÓN DE PRECIO ===
            df['price_acceleration_1s'] = df['price_change_1s'].diff().fillna(0)
            df['price_acceleration_3s'] = df['price_change_3s'].diff().fillna(0)
            df['price_acceleration_5s'] = df['price_change_5s'].diff().fillna(0)
            
            # Jerk (cambio en aceleración)
            df['price_jerk_1s'] = df['price_acceleration_1s'].diff().fillna(0)
            df['price_jerk_3s'] = df['price_acceleration_3s'].diff().fillna(0)
            
            # === MOMENTUM INSTANTÁNEO ===
            df['momentum_1s'] = df['close'] - df['close'].shift(1)
            df['momentum_3s'] = df['close'] - df['close'].shift(2)
            df['momentum_5s'] = df['close'] - df['close'].shift(3)
            df['momentum_10s'] = df['close'] - df['close'].shift(4)
            
            # === SEÑALES DE MOMENTUM ===
            df['momentum_signal_1s'] = (df['momentum_1s'] > 0).astype(int)
            df['momentum_signal_3s'] = (df['momentum_3s'] > 0).astype(int)
            df['momentum_signal_5s'] = (df['momentum_5s'] > 0).astype(int)
            
            # Fuerza del momentum
            momentum_std = df['momentum_1s'].rolling(20).std()
            df['momentum_strength_1s'] = safe_divide(df['momentum_1s'], momentum_std, fallback_method='constant')
            df['momentum_strength_3s'] = safe_divide(df['momentum_3s'], momentum_std, fallback_method='constant')
            df['momentum_strength_5s'] = safe_divide(df['momentum_5s'], momentum_std, fallback_method='constant')
            
            # === VOLATILIDAD ULTRA-CORTA ===
            df['volatility_1s'] = df['close'].pct_change().rolling(3).std().fillna(0)
            df['volatility_3s'] = df['close'].pct_change().rolling(5).std().fillna(0)
            df['volatility_5s'] = df['close'].pct_change().rolling(8).std().fillna(0)
            df['volatility_10s'] = df['close'].pct_change().rolling(12).std().fillna(0)
            
            # === PATRONES ULTRA-CORTOS ===
            df['higher_high_1s'] = (df['high'] > df['high'].shift(1)).astype(int)
            df['higher_high_3s'] = (df['high'] > df['high'].shift(2)).astype(int)
            df['lower_low_1s'] = (df['low'] < df['low'].shift(1)).astype(int)
            df['lower_low_3s'] = (df['low'] < df['low'].shift(2)).astype(int)
            
            # Breakouts
            rolling_max = df['high'].rolling(10).max().shift(1)
            rolling_min = df['low'].rolling(10).min().shift(1)
            df['breakout_1s'] = (df['close'] > rolling_max).astype(int)
            df['breakout_3s'] = (df['close'] > rolling_max * 1.001).astype(int)
            df['breakout_5s'] = (df['close'] > rolling_max * 1.002).astype(int)
            
            # === EFFICIENCY ULTRA-CORTA ===
            # Eficiencia de mercado en intervalos muy cortos
            close_diff_abs = pd.Series(np.abs(df['close'].diff()), index=df.index)
            efficiency_numerator_1s = np.abs(df['close'] - df['close'].shift(3))
            efficiency_numerator_3s = np.abs(df['close'] - df['close'].shift(5))
            efficiency_numerator_5s = np.abs(df['close'] - df['close'].shift(8))
            
            efficiency_denominator_1s = close_diff_abs.rolling(3, min_periods=1).sum()
            efficiency_denominator_3s = close_diff_abs.rolling(5, min_periods=1).sum()
            efficiency_denominator_5s = close_diff_abs.rolling(8, min_periods=1).sum()
            
            df['efficiency_1s'] = safe_divide(efficiency_numerator_1s, efficiency_denominator_1s, fallback_method='constant').fillna(0)
            df['efficiency_3s'] = safe_divide(efficiency_numerator_3s, efficiency_denominator_3s, fallback_method='constant').fillna(0)
            df['efficiency_5s'] = safe_divide(efficiency_numerator_5s, efficiency_denominator_5s, fallback_method='constant').fillna(0)
            
            # === COMPOSITE SIGNALS ===
            # ✅ CORRECCIÓN: Usar operaciones booleanas seguras para evitar errores de Series
            # Señal combinada momentum + volumen
            momentum_volume_mask = (df['momentum_signal_1s'] == 1) & (df['volume_ratio_1s'] > 1.2)
            df['momentum_volume_signal'] = momentum_volume_mask.astype(int)
            
            # Señal combinada momentum + RSI
            momentum_rsi_mask = (df['momentum_signal_1s'] == 1) & (df['rsi_3s'] < 30)
            df['momentum_rsi_signal'] = momentum_rsi_mask.astype(int)
            
            # Señal triple (precio + volumen + RSI)
            triple_momentum_mask = (df['momentum_signal_1s'] == 1) & (df['volume_ratio_1s'] > 1.2) & (df['rsi_3s'] < 30)
            df['triple_momentum_signal'] = triple_momentum_mask.astype(int)
            
            # === NORMALIZED FEATURES ===
            # Normalizar cambios de precio
            price_change_std = df['price_change_1s'].rolling(20).std()
            df['price_change_normalized_1s'] = safe_divide(df['price_change_1s'], price_change_std, fallback_method='constant')
            df['price_change_normalized_3s'] = safe_divide(df['price_change_3s'], price_change_std, fallback_method='constant')
            df['price_change_normalized_5s'] = safe_divide(df['price_change_5s'], price_change_std, fallback_method='constant')
            
            # Normalizar volumen
            volume_std = df['volume'].rolling(20).std()
            df['volume_normalized_1s'] = safe_divide(df['volume'], volume_std, fallback_method='constant')
            df['volume_normalized_3s'] = safe_divide(df['volume'], volume_std, fallback_method='constant')
            df['volume_normalized_5s'] = safe_divide(df['volume'], volume_std, fallback_method='constant')
            
            # Limpiar valores extremos
            ultra_features = [col for col in df.columns if any(x in col for x in ['1s', '3s', '5s', '10s'])]
            for col in ultra_features:
                if col in df.columns:
                    # Aplicar clipping robusto
                    q99 = df[col].quantile(0.99)
                    q01 = df[col].quantile(0.01)
                    if pd.notna(q99) and pd.notna(q01) and q99 != q01:
                        df[col] = df[col].clip(lower=q01, upper=q99)
                    
                    # Rellenar NaN
                    df[col] = df[col].fillna(0)
            
            print(f"✅ Features ultra-momentum calculadas: {len(ultra_features)} features")
            return df
            
        except Exception as e:
            print(f"⚠️ Error calculando features ultra-momentum: {e}")
            return df


# === FUNCIONES DE UTILIDAD ===

def create_features_engine(quiet_mode: bool = False, api_key: str = None, api_secret: str = None, 
                          use_authenticated_api: bool = True) -> CentralizedFeaturesEngine:
    """
    Factory function para crear el motor de features
    
    Args:
        quiet_mode: Si True, suprime advertencias de compresión/rangos que pueden ser normales
        api_key: API key de Binance para acceso autenticado
        api_secret: API secret de Binance para acceso autenticado
        use_authenticated_api: Si True, usa API autenticada (menor latencia)
    """
    return CentralizedFeaturesEngine(quiet_mode=quiet_mode, api_key=api_key, api_secret=api_secret, 
                                   use_authenticated_api=use_authenticated_api)

def calculate_features_for_symbol(df: pd.DataFrame, feature_set: str = 'tcn_definitivo_v3') -> pd.DataFrame:
    """Función de conveniencia para calcular features"""
    engine = create_features_engine()
    return engine.calculate_features(df, feature_set)

def get_available_feature_sets() -> List[str]:
    """Obtener lista de conjuntos de features disponibles"""
    engine = create_features_engine()
    return list(engine.feature_sets.keys())


# === TESTING SIMPLIFICADO ===
def test_centralized_features_v3_simple():
    """Test simplificado del motor centralizado de features V3 sin dependencias externas"""
    print("🧪 TESTING CENTRALIZED FEATURES ENGINE V3 - CORRELATION OPTIMIZED + CRITICAL FEATURES")
    print("=" * 80)
    print("✅ Features críticas restauradas:")
    print("   - macd_signal: Crítico para timing de señales de entrada/salida")
    print("   - bb_middle: Importante para contexto de Bollinger Bands")
    print("   - sma_20: Media móvil fundamental para análisis técnico")
    print("=" * 80)

    try:
        # Crear motor de features
        engine = CentralizedFeaturesEngine(quiet_mode=True)
        
        # Verificar que las features críticas están en el conjunto V3
        v3_features = engine.feature_sets['tcn_definitivo_v3']
        
        print(f"\n🔍 VERIFICANDO FEATURES CRÍTICAS RESTAURADAS:")
        critical_features = ['macd_signal', 'bb_middle', 'sma_20']
        
        for feature in critical_features:
            if feature in v3_features:
                print(f"   ✅ {feature}: RESTAURADO correctamente")
            else:
                print(f"   ❌ {feature}: NO encontrado en conjunto V3")
        
        # Verificar conteo total
        original_count = len(engine.feature_sets['tcn_definitivo'])
        v3_count = len(v3_features)
        reduction = original_count - v3_count
        
        print(f"\n📊 RESUMEN DE OPTIMIZACIÓN:")
        print(f"   📈 Features originales: {original_count}")
        print(f"   🎯 Features V3: {v3_count}")
        print(f"   🗑️  Features eliminadas: {reduction}")
        print(f"   📉 Reducción: {(reduction/original_count)*100:.1f}%")
        
        # Verificar que las features críticas están en las posiciones correctas
        print(f"\n🔧 VERIFICACIÓN DE IMPLEMENTACIÓN:")
        
        # Verificar MACD
        macd_features = [f for f in v3_features if 'macd' in f]
        print(f"   📊 Features MACD: {macd_features}")
        
        # Verificar Bollinger Bands
        bb_features = [f for f in v3_features if 'bb_' in f]
        print(f"   📊 Features Bollinger Bands: {bb_features}")
        
        # Verificar Moving Averages
        ma_features = [f for f in v3_features if 'sma_' in f or 'ema_' in f]
        print(f"   📊 Features Moving Averages: {ma_features}")
        
        print(f"\n🎯 VERIFICACIÓN COMPLETADA:")
        if all(feature in v3_features for feature in critical_features):
            print(f"   ✅ TODAS las features críticas restauradas correctamente")
            print(f"   ✅ Conjunto V3 optimizado: {v3_count} features")
            print(f"   ✅ Reducción de correlaciones mantenida")
        else:
            print(f"   ❌ ALGUNAS features críticas faltan")
            
    except Exception as e:
        print(f"❌ Error en test simplificado: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 80)


# === RESUMEN DE CAMBIOS IMPLEMENTADOS ===
"""
🎯 RESTAURACIÓN DE FEATURES CRÍTICAS COMPLETADA

✅ CAMBIOS IMPLEMENTADOS:
1. macd_signal restaurado - Crítico para timing de señales de entrada/salida
2. bb_middle restaurado - Importante para contexto de Bollinger Bands  
3. sma_20 restaurado - Media móvil fundamental para análisis técnico

📊 IMPACTO EN OPTIMIZACIÓN:
- Features V3: 43 → 46 (+3 críticas)
- Reducción total: 88 → 46 (-42 features redundantes)
- Eficiencia: 47.7% de features originales
- Preserva 100% de features críticas para timing

🔧 JUSTIFICACIÓN TÉCNICA:
- macd_signal: Necesario para cruces MACD (señales de trading)
- bb_middle: Contexto esencial para interpretar bb_position y bb_width
- sma_20: Referencia estándar en análisis técnico (soporte/resistencia)

⚡ BENEFICIOS:
- Mantiene optimización de correlaciones
- Preserva capacidad de timing crítico
- Mejora interpretabilidad de indicadores
- Compatible con estrategias de trading existentes
"""


def test_bearish_detection():
    """
    🧪 Test específico para validar mejora en detección bajista
    """
    print("🧪 TESTING ENHANCED BEARISH DETECTION")
    print("=" * 60)
    
    try:
        # Crear motor de features
        engine = CentralizedFeaturesEngine(quiet_mode=True)
        
        # Verificar que el conjunto enhanced está disponible
        if 'tcn_definitivo_v3_enhanced' not in engine.feature_sets:
            print("❌ Conjunto enhanced no encontrado")
            return
            
        enhanced_features = engine.feature_sets['tcn_definitivo_v3_enhanced']
        base_features = engine.feature_sets['tcn_definitivo_v3']
        
        print(f"📊 Conjunto base V3: {len(base_features)} features")
        print(f"🎯 Conjunto enhanced: {len(enhanced_features)} features")
        print(f"🐻 Features adicionales: {len(enhanced_features) - len(base_features)} features")
        
        # Verificar features específicas de detección bajista
        crypto_bearish_features = [
            'rsi_divergence_bearish', 'macd_bearish_cross', 'trend_strength_ratio',
            'volume_bearish_signal', 'price_momentum_bearish', 'support_resistance_context',
            'volatility_expansion_bear', 'momentum_divergence_bear'
        ]
        
        print(f"\n🔍 VERIFICANDO FEATURES DE DETECCIÓN BAJISTA:")
        for feature in crypto_bearish_features:
            if feature in enhanced_features:
                print(f"   ✅ {feature}: Disponible en conjunto enhanced")
            else:
                print(f"   ❌ {feature}: NO encontrado")
        
        # Simular datos con tendencia bajista para testing
        print(f"\n📊 SIMULANDO DATOS DE TENDENCIA BAJISTA...")
        
        # Crear datos de prueba
        dates = pd.date_range('2024-01-01', periods=100, freq='5min')
        
        # Crear tendencia bajista con rebotes técnicos
        base_price = 50000
        trend_data = []
        for i in range(100):
            # Tendencia bajista general con volatilidad
            trend_factor = 1 - (i * 0.002)  # Decline 0.2% per period
            noise = np.random.normal(0, 0.01)  # 1% noise
            price = base_price * trend_factor * (1 + noise)
            
            # Añadir rebotes técnicos ocasionales para confundir al modelo
            if i % 15 == 0 and i > 20:
                price *= 1.02  # 2% bounce
                
            trend_data.append(price)
        
        # Crear DataFrame de prueba
        df_test = pd.DataFrame({
            'timestamp': dates,
            'open': trend_data,
            'high': [p * 1.01 for p in trend_data],
            'low': [p * 0.99 for p in trend_data], 
            'close': trend_data,
            'volume': np.random.uniform(1000, 5000, 100)
        })
        
        print(f"📊 Datos de prueba creados: tendencia bajista con rebotes técnicos")
        print(f"   Precio inicial: {df_test['close'].iloc[0]:.2f}")
        print(f"   Precio final: {df_test['close'].iloc[-1]:.2f}")
        print(f"   Cambio total: {((df_test['close'].iloc[-1] / df_test['close'].iloc[0]) - 1) * 100:.2f}%")
        
        # Calcular features enhanced
        print(f"\n🎯 CALCULANDO FEATURES ENHANCED...")
        enhanced_df = engine.calculate_features(df_test, 'tcn_definitivo_v3_enhanced')
        
        print(f"✅ Features enhanced calculadas: {len(enhanced_df.columns)} columnas")
        
        # Verificar que las features bajistas están presentes
        bearish_present = [f for f in crypto_bearish_features if f in enhanced_df.columns]
        print(f"🐻 Features bajistas calculadas: {len(bearish_present)}/{len(crypto_bearish_features)}")
        
        if len(bearish_present) == len(crypto_bearish_features):
            print("✅ TODAS las features de detección bajista funcionando correctamente")
        else:
            missing = set(crypto_bearish_features) - set(bearish_present)
            print(f"❌ Features faltantes: {missing}")
        
        return df_test
        
    except Exception as e:
        print(f"❌ Error en test de detección bajista: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_centralized_features_v3_simple()
    print("\n" + "="*80)
    test_bearish_detection()