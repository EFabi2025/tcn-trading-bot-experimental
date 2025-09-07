#!/usr/bin/env python3
"""
🚀 TCN ADAPTATIVE TRAINER - VERSIÓN CONFIGURABLE CON ARQUITECTURA TCN MEJORADA
Entrenador con thresholds adaptativos, profit-aware loss y nueva arquitectura TCN

✅ NUEVA FUNCIONALIDAD: ARQUITECTURA TCN MEJORADA (2024)
- Bloques TCN residuales con normalización de capas
- Atención temporal específica para crypto con ponderación por volatilidad
- Procesamiento multi-escala con dilataciones crecientes (1, 2, 4, 8, 16, 32)
- Activación Swish para mejor convergencia
- Arquitectura más profunda y robusta (~200K parámetros)

✅ FUNCIONALIDADES EXISTENTES:
- Profit-aware loss functions para maximizar rentabilidad
- Selección de regímenes de mercado equilibrados
- Thresholds adaptativos optimizados
- Features críticas restauradas (macd_signal, bb_middle, sma_20)
- Percentiles dinámicos corregidos (BUY: 15%, SELL: 15%, HOLD: 70%)

🎯 ARQUITECTURAS TCN DISPONIBLES:
1. 'enhanced': Nueva arquitectura TCN mejorada (RECOMENDADO)
2. 'original': Arquitectura crypto optimizada probada
3. 'hybrid': Combinación de ambas arquitecturas
4. 'efficient_v3': Arquitectura TCN V3 eficiente y rápida (NUEVA - 2-3x más rápido)

🎯 EJEMPLOS DE USO:

1. ENTRENAMIENTO CON TCN MEJORADO:
   python tcn_adaptative_trainer_v2.py --non_interactive --enhanced_tcn

2. ENTRENAMIENTO COMPLETO OPTIMIZADO:
   python tcn_adaptative_trainer_v2.py --non_interactive --profit_aware_loss --balanced_regimes --enhanced_tcn

3. ENTRENAMIENTO RÁPIDO CON TCN V3 EFICIENTE:
   python tcn_adaptative_trainer_v2.py --non_interactive --use_efficient_tcn_v3

4. ENTRENAMIENTO ULTRA-RÁPIDO PARA PROTOTIPADO:
   python tcn_adaptative_trainer_v2.py --non_interactive --use_efficient_tcn_v3 --tcn_v3_filters 32 --tcn_v3_dilations 1,2,4

🎯 BENEFICIOS DE LA ARQUITECTURA TCN MEJORADA:
- Mejor captura de patrones temporales complejos
- Convergencia más estable durante el entrenamiento
- Manejo robusto de volatilidad del mercado
- Arquitectura especializada para trading de crypto
- Bloques residuales para entrenamiento más profundo

🎯 BENEFICIOS DE LA ARQUITECTURA TCN V3 EFICIENTE:
- Entrenamiento 2-3x más rápido que enhanced
- Menor uso de memoria GPU/RAM
- Mejor generalización en datasets pequeños
- Ideal para prototipado y experimentación rápida
- Arquitectura limpia y fácil de mantener
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd
import tensorflow as tf
import time
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import talib
import warnings
import pickle
import os
from typing import List, Optional, Union, Dict, Tuple
from collections import Counter
warnings.filterwarnings('ignore')

# Importar motor de features actual (sin cambios)
from centralized_features_engine3 import CentralizedFeaturesEngine

# ✅ NUEVA CONFIGURACIÓN: CARGAR DESDE config_example.env PARA ENTRENAMIENTO
def load_training_config():
    """🔧 Cargar configuración de entrenamiento desde training_config.env"""
    
    # Prioridad: 1. training_config.env, 2. config_example.env, 3. valores por defecto
    config_files = ["training_config.env", "config_example.env"]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"🔧 Cargando configuración de entrenamiento desde {config_file}")
            break
    else:
        print(f"⚠️ No se encontraron archivos de configuración, usando valores por defecto")
        return {
            'MIN_CONFIDENCE_THRESHOLD': 0.65,
            'MIN_SELL_CONFIDENCE_THRESHOLD': 0.75,
            'SIGNAL_REVERSAL_THRESHOLD': 0.85,
            'STOP_LOSS_PERCENT': 1.5,
            'TAKE_PROFIT_PERCENT': 1.9,
            'TRAILING_STOP_PERCENT': 1.2
        }
    
    if not os.path.exists(config_file):
        print(f"⚠️ Archivo {config_file} no encontrado, usando valores por defecto")
        return {
            'MIN_CONFIDENCE_THRESHOLD': 0.65,
            'MIN_SELL_CONFIDENCE_THRESHOLD': 0.75,
            'SIGNAL_REVERSAL_THRESHOLD': 0.85,
            'STOP_LOSS_PERCENT': 1.5,
            'TAKE_PROFIT_PERCENT': 1.9,
            'TRAILING_STOP_PERCENT': 1.2
        }
    
    print(f"🔧 Cargando configuración de entrenamiento desde {config_file}")
    
    config = {}
    try:
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remover comentarios del valor
                    if '#' in value:
                        value = value.split('#')[0].strip()
                    config[key] = value
        
        # Convertir valores numéricos
        numeric_keys = [
            'MIN_CONFIDENCE_THRESHOLD', 'MIN_SELL_CONFIDENCE_THRESHOLD',
            'SIGNAL_REVERSAL_THRESHOLD', 'STOP_LOSS_PERCENT',
            'TAKE_PROFIT_PERCENT', 'TRAILING_STOP_PERCENT'
        ]
        
        for key in numeric_keys:
            if key in config:
                try:
                    config[key] = float(config[key])
                except ValueError:
                    print(f"⚠️ Valor inválido para {key}: {config[key]}, usando por defecto")
                    if key == 'MIN_CONFIDENCE_THRESHOLD':
                        config[key] = 0.65
                    elif key == 'MIN_SELL_CONFIDENCE_THRESHOLD':
                        config[key] = 0.50
                    elif key == 'SIGNAL_REVERSAL_THRESHOLD':
                        config[key] = 0.85
                    elif key == 'STOP_LOSS_PERCENT':
                        config[key] = 1.5
                    elif key == 'TAKE_PROFIT_PERCENT':
                        config[key] = 1.9
                    elif key == 'TRAILING_STOP_PERCENT':
                        config[key] = 1.2
        
        print(f"✅ Configuración cargada exitosamente:")
        for key, value in config.items():
            if key in numeric_keys:
                print(f"   📊 {key}: {value}")
        
        return config
        
    except Exception as e:
        print(f"❌ Error cargando configuración: {e}")
        print(f"🔄 Usando valores por defecto")
        return {
            'MIN_CONFIDENCE_THRESHOLD': 0.65,
            'MIN_SELL_CONFIDENCE_THRESHOLD': 0.75,
            'SIGNAL_REVERSAL_THRESHOLD': 0.85,
            'STOP_LOSS_PERCENT': 1.5,
            'TAKE_PROFIT_PERCENT': 1.9,
            'TRAILING_STOP_PERCENT': 1.2
        }

# Cargar configuración al importar el módulo
TRAINING_CONFIG = load_training_config()

# ✅ NUEVA IMPORTACIÓN: FEATURES 3M ESPECIALIZADAS
try:
    from features3m import AdvancedFeaturesEngine3m, TechnicalIndicatorsBridge3m
    FEATURES_3M_AVAILABLE = True
    print("✅ Features 3M especializadas disponibles")
except ImportError as e:
    FEATURES_3M_AVAILABLE = False
    print(f"⚠️ Features 3M no disponibles: {e}")

# ✅ NUEVAS IMPORTACIONES PARA PROFIT-AWARE LOSS Y MÉTRICAS AVANZADAS
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score

# Importación condicional de profit-aware loss
try:
    from profit_aware_loss import create_profit_aware_loss, prepare_future_returns_tensor, evaluate_profit_performance
    PROFIT_AWARE_AVAILABLE = True
except ImportError:
    print("⚠️ Profit-aware loss no disponible, usando funciones estándar")
    PROFIT_AWARE_AVAILABLE = False

# Importación condicional de métricas mejoradas
try:
    from trading_metrics_enhanced import TradingMetricsEnhanced, evaluate_model_with_enhanced_metrics
    ENHANCED_METRICS_AVAILABLE = True
except ImportError:
    print("⚠️ Métricas mejoradas no disponibles, usando funciones estándar")
    ENHANCED_METRICS_AVAILABLE = False

# ✅ NUEVA IMPORTACIÓN PARA DETECCIÓN DE REGÍMENES DE MERCADO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ✅ IMPORTACIONES OPCIONALES PARA VISUALIZACIÓN
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib no disponible, gráficos deshabilitados")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    if MATPLOTLIB_AVAILABLE:
        print("⚠️  seaborn no disponible, usando matplotlib básico para gráficos")
    else:
        print("⚠️  seaborn no disponible, gráficos deshabilitados")

# ✅ IMPORTACIÓN OPCIONAL DE PSUTIL
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️  psutil no disponible, monitoreo de memoria deshabilitado")

# Variable global para controlar si se pueden generar gráficos
PLOTTING_AVAILABLE = MATPLOTLIB_AVAILABLE


# ✅ ELIMINADAS LAS FUNCIONES PERSONALIZADAS QUE CAUSABAN ERRORES
# Usar únicamente funciones estándar de Keras para máxima estabilidad


class TrainingConfig:
    """🔧 Configuración completa de entrenamiento - TOTALMENTE CONFIGURABLE"""

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
        self.use_adaptive_thresholds = False  # 🎯 FORZAR thresholds fijos rentables
        
        # 🚀 NUEVO: SISTEMA DE PERCENTILES DINÁMICOS (PROFESIONAL)
        self.use_dynamic_percentiles = True  # Usar percentiles en lugar de thresholds fijos
        self.percentile_window = 1000  # Ventana rolling para calcular percentiles
        self.buy_percentile = 80   # Top 15% = BUY (percentil 85+)
        self.sell_percentile = 20  # Bottom 15% = SELL (percentil 0-15)
        self.min_samples_for_percentiles = 500  # Mínimo de datos para usar percentiles
        
        # 🎯 FEATURE SET - AGREGADO
        self.feature_set = 'tcn_definitivo_v3_enhanced'  # Por defecto - V3 Enhanced con detección bajista
        
        # 🚀 NUEVA ARQUITECTURA TCN MEJORADA
        self.use_enhanced_tcn = True  # Usar arquitectura TCN mejorada por defecto
        self.tcn_architecture = 'enhanced'  # 'enhanced', 'original', 'hybrid'

        # ✅ NUEVO: SELECCIÓN DE REGÍMENES DE MERCADO EQUILIBRADOS
        self.use_balanced_regimes = False  # Por defecto deshabilitado
        self.regime_balance_method = 'auto'  # 'auto', 'manual', 'stratified'
        self.target_samples_per_regime = None  # Para método manual
        
        # 🎯 NUEVO: PROFIT-AWARE LOSS CONFIGURATION
        self.use_profit_aware_loss = True  # Por defecto HABILITADO para maximizar rentabilidad
        self.loss_type = 'profit_weighted'  # Más agresivo en rentabilidad
        self.base_fee = 0.001  # 0.1% fee por trade
        self.spread_cost = 0.0005  # 0.05% spread estimado
        self.profit_amplifier = 3.0  # MÁS agresivo en recompensas por trades rentables
        self.loss_amplifier = 2.5  # MÁS agresivo en penalizaciones por pérdidas
        self.use_enhanced_metrics = True  # Usar métricas de trading mejoradas
        
        # ✅ NUEVO: SELECCIÓN POR PERÍODOS DE FECHAS ESPECÍFICOS
        self.use_date_periods = False  # Por defecto deshabilitado
        self.date_periods = []  # Lista de períodos de fechas para selección
        self.date_periods_method = 'manual'  # 'manual', 'preset', 'custom'
        
        # ✅ PERÍODOS PREDEFINIDOS PARA DIFERENTES REGÍMENES DE MERCADO
        self.preset_date_periods = {
            'crypto_bull_bear_2021_2023': [
                {'start_date': '2021-01-01', 'end_date': '2021-11-30', 'description': 'Mercado alcista 2021'},
                {'start_date': '2022-01-01', 'end_date': '2022-12-31', 'description': 'Mercado bajista 2022'},
                {'start_date': '2023-01-01', 'end_date': '2023-06-30', 'description': 'Recuperación 2023'}
            ],
            'crypto_volatility_2020_2023': [
                {'start_date': '2020-03-01', 'end_date': '2020-12-31', 'description': 'COVID y recuperación'},
                {'start_date': '2021-01-01', 'end_date': '2021-12-31', 'description': 'Bull market completo'},
                {'start_date': '2022-01-01', 'end_date': '2022-12-31', 'description': 'Bear market completo'},
                {'start_date': '2023-01-01', 'end_date': '2023-12-31', 'description': 'Mercado mixto 2023'}
            ],
            'btc_cycles_2017_2023': [
                {'start_date': '2017-01-01', 'end_date': '2017-12-31', 'description': 'Bull run 2017'},
                {'start_date': '2018-01-01', 'end_date': '2018-12-31', 'description': 'Bear market 2018'},
                {'start_date': '2019-01-01', 'end_date': '2020-02-29', 'description': 'Acumulación 2019-2020'},
                {'start_date': '2020-03-01', 'end_date': '2021-11-30', 'description': 'COVID bull run'},
                {'start_date': '2022-01-01', 'end_date': '2022-12-31', 'description': 'Crypto winter 2022'},
                {'start_date': '2023-01-01', 'end_date': '2023-12-31', 'description': 'Recuperación 2023'}
            ]
        }
        
        # ✅ NUEVO: ATRIBUTO PRESET_PERIODS PARA COMPATIBILIDAD
        self.preset_periods = None  # Se establece cuando se selecciona un preset

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

        # ✅ NUEVO: MANEJAR PARÁMETROS DE PERÍODOS DE FECHAS
        if hasattr(args, 'use_date_periods') and args.use_date_periods:
            self.use_date_periods = True
            
        if hasattr(args, 'date_periods_method') and args.date_periods_method:
            if args.date_periods_method in ['manual', 'preset', 'custom']:
                self.date_periods_method = args.date_periods_method
                
        if hasattr(args, 'preset_periods') and args.preset_periods:
            if args.preset_periods in self.preset_date_periods:
                self.date_periods = self.preset_date_periods[args.preset_periods]
                self.date_periods_method = 'preset'
                self.preset_periods = args.preset_periods  # ✅ AGREGADO: Establecer preset_periods
                print(f"✅ Períodos predefinidos cargados: {args.preset_periods}")
                
        if hasattr(args, 'custom_periods') and args.custom_periods:
            try:
                # Formato esperado: "2023-01-01:2023-03-31,2023-04-01:2023-06-30"
                periods = []
                for period_str in args.custom_periods.split(','):
                    start, end = period_str.split(':')
                    periods.append({
                        'start_date': start.strip(),
                        'end_date': end.strip(),
                        'description': f'Período personalizado {start} a {end}'
                    })
                self.date_periods = periods
                self.date_periods_method = 'custom'
                print(f"✅ {len(periods)} períodos personalizados cargados")
            except Exception as e:
                print(f"⚠️  Error procesando períodos personalizados: {e}")
                print(f"   📋 Formato esperado: '2023-01-01:2023-03-31,2023-04-01:2023-06-30'")

    def print_config(self):
        """Mostrar configuración actual"""
        print("\n🔧 CONFIGURACIÓN DE ENTRENAMIENTO:")
        print("=" * 50)
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
        
        # 🎯 NUEVO: MOSTRAR CONFIGURACIÓN DE PROFIT-AWARE LOSS
        use_profit_aware = getattr(self, 'use_profit_aware_loss', False)
        print(f"🎯 Profit-Aware Loss: {'✅' if use_profit_aware else '❌'}")
        if use_profit_aware:
            print(f"   📊 Tipo de loss: {getattr(self, 'loss_type', 'combined')}")
            print(f"   💰 Fee base: {getattr(self, 'base_fee', 0.001):.3f} ({getattr(self, 'base_fee', 0.001)*100:.1f}%)")
            print(f"   📊 Spread cost: {getattr(self, 'spread_cost', 0.0005):.3f} ({getattr(self, 'spread_cost', 0.0005)*100:.2f}%)")
            print(f"   📈 Amplificador profit: {getattr(self, 'profit_amplifier', 2.0)}x")
            print(f"   📉 Amplificador loss: {getattr(self, 'loss_amplifier', 1.5)}x")
            print(f"   📊 Métricas mejoradas: {'✅' if getattr(self, 'use_enhanced_metrics', True) else '❌'}")
        
        # ✅ NUEVO: MOSTRAR CONFIGURACIÓN DE PERÍODOS DE FECHAS
        print(f"📅 Períodos de fechas: {'✅' if self.use_date_periods else '❌'}")
        if self.use_date_periods:
            print(f"   📊 Método: {self.date_periods_method}")
            if self.date_periods:
                print(f"   📅 Períodos configurados: {len(self.date_periods)}")
                for i, period in enumerate(self.date_periods[:3]):  # Mostrar solo los primeros 3
                    start = period.get('start_date', 'N/A')
                    end = period.get('end_date', 'N/A')
                    desc = period.get('description', 'Sin descripción')
                    print(f"      {i+1}. {start} a {end}: {desc}")
                if len(self.date_periods) > 3:
                    print(f"      ... y {len(self.date_periods) - 3} períodos más")
            elif self.date_periods_method == 'preset':
                print(f"   📅 Usando períodos predefinidos")
        print("=" * 50)


class TradingMetrics:
    """📊 Métricas específicas para trading con análisis detallado por clase"""

    def __init__(self):
        self.class_names = ['SELL', 'HOLD', 'BUY']
        self.class_colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']

    def calculate_trading_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                y_pred_proba: np.ndarray = None) -> Dict:
        """🎯 Calcular métricas específicas para trading"""

        # Métricas básicas
        accuracy = np.mean(y_true == y_pred)

        # Reporte de clasificación detallado
        report = classification_report(y_true, y_pred,
                                    target_names=self.class_names,
                                    output_dict=True)

        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)

        # Métricas por clase
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        # ✅ MÉTRICAS ESPECÍFICAS PARA TRADING
        trading_metrics = {
            'accuracy': accuracy,
            'precision_per_class': dict(zip(self.class_names, precision)),
            'recall_per_class': dict(zip(self.class_names, recall)),
            'f1_per_class': dict(zip(self.class_names, f1)),
            'support_per_class': dict(zip(self.class_names, support)),
            'confusion_matrix': cm,
            'classification_report': report,
            'total_samples': len(y_true)
        }

        # ✅ MÉTRICAS DE CONFIANZA (si hay probabilidades)
        if y_pred_proba is not None:
            confidence_metrics = self.calculate_confidence_metrics(y_true, y_pred, y_pred_proba)
            trading_metrics.update(confidence_metrics)

        return trading_metrics

    def calculate_confidence_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   y_pred_proba: np.ndarray) -> Dict:
        """🎯 Calcular métricas de confianza de las predicciones"""

        # Confianza promedio por predicción correcta/incorrecta
        correct_mask = y_true == y_pred
        incorrect_mask = ~correct_mask

        confidence_metrics = {
            'avg_confidence_correct': np.mean(np.max(y_pred_proba[correct_mask], axis=1)) if np.any(correct_mask) else 0,
            'avg_confidence_incorrect': np.mean(np.max(y_pred_proba[incorrect_mask], axis=1)) if np.any(incorrect_mask) else 0,
            'confidence_threshold_80': np.mean(np.max(y_pred_proba, axis=1) > 0.8),
            'confidence_threshold_90': np.mean(np.max(y_pred_proba, axis=1) > 0.9),
            'high_confidence_accuracy': self.calculate_high_confidence_accuracy(y_true, y_pred, y_pred_proba, threshold=0.8)
        }

        return confidence_metrics

    def calculate_high_confidence_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray,
                                         y_pred_proba: np.ndarray, threshold: float = 0.8) -> float:
        """🎯 Calcular accuracy solo para predicciones con alta confianza"""
        high_conf_mask = np.max(y_pred_proba, axis=1) > threshold
        if np.any(high_conf_mask):
            return np.mean(y_true[high_conf_mask] == y_pred[high_conf_mask])
        return 0.0

    def print_trading_report(self, metrics: Dict, symbol: str, timeframe: str):
        """📊 Imprimir reporte detallado de métricas de trading"""

        print(f"\n📊 REPORTE DE MÉTRICAS DE TRADING - {symbol} ({timeframe})")
        print("=" * 70)

        # Accuracy general
        print(f"🎯 ACCURACY GENERAL: {metrics['accuracy']:.3f}")

        # Métricas por clase
        print(f"\n📈 MÉTRICAS POR CLASE:")
        for i, class_name in enumerate(self.class_names):
            precision = metrics['precision_per_class'][class_name]
            recall = metrics['recall_per_class'][class_name]
            f1 = metrics['f1_per_class'][class_name]
            support = metrics['support_per_class'][class_name]

            print(f"   {class_name:>5}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, Support={support}")

        # Métricas de confianza
        if 'avg_confidence_correct' in metrics:
            print(f"\n🎯 MÉTRICAS DE CONFIANZA:")
            print(f"   Confianza promedio (correctas): {metrics['avg_confidence_correct']:.3f}")
            print(f"   Confianza promedio (incorrectas): {metrics['avg_confidence_incorrect']:.3f}")
            print(f"   Predicciones >80% confianza: {metrics['confidence_threshold_80']:.1%}")
            print(f"   Predicciones >90% confianza: {metrics['confidence_threshold_90']:.1%}")
            print(f"   Accuracy alta confianza (>80%): {metrics['high_confidence_accuracy']:.3f}")

        # Análisis de trading
        self.print_trading_analysis(metrics, symbol)

    def print_trading_analysis(self, metrics: Dict, symbol: str):
        """🎯 Análisis específico para trading"""

        print(f"\n🎯 ANÁLISIS DE TRADING - {symbol}:")

        # Análisis de señales de compra
        buy_precision = metrics['precision_per_class']['BUY']
        buy_recall = metrics['recall_per_class']['BUY']

        if buy_precision > 0.6 and buy_recall > 0.5:
            print(f"   ✅ BUY: Buena precisión ({buy_precision:.3f}) y recall ({buy_recall:.3f})")
        elif buy_precision < 0.4:
            print(f"   ⚠️  BUY: Baja precisión ({buy_precision:.3f}) - muchas falsas alarmas")
        elif buy_recall < 0.3:
            print(f"   ⚠️  BUY: Bajo recall ({buy_recall:.3f}) - se pierden oportunidades")

        # Análisis de señales de venta
        sell_precision = metrics['precision_per_class']['SELL']
        sell_recall = metrics['recall_per_class']['SELL']

        if sell_precision > 0.6 and sell_recall > 0.5:
            print(f"   ✅ SELL: Buena precisión ({sell_precision:.3f}) y recall ({sell_recall:.3f})")
        elif sell_precision < 0.4:
            print(f"   ⚠️  SELL: Baja precisión ({sell_precision:.3f}) - muchas falsas alarmas")
        elif sell_recall < 0.3:
            print(f"   ⚠️  SELL: Bajo recall ({sell_recall:.3f}) - se pierden oportunidades")

        # Análisis de HOLD
        hold_f1 = metrics['f1_per_class']['HOLD']
        if hold_f1 > 0.6:
            print(f"   ✅ HOLD: Buen balance ({hold_f1:.3f})")
        else:
            print(f"   ⚠️  HOLD: Balance pobre ({hold_f1:.3f})")

    def save_metrics_plot(self, metrics: Dict, symbol: str, timeframe: str, save_path: str):
        """📊 Guardar gráfico de métricas (opcional)"""

        # ✅ CORRECCIÓN: Verificar si se pueden generar gráficos
        if not PLOTTING_AVAILABLE:
            print(f"⚠️  Gráficos deshabilitados - matplotlib no disponible")
            print(f"   📊 Métricas disponibles en: {save_path.replace('.png', '.json')}")
            return

        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Métricas de Trading - {symbol} ({timeframe})', fontsize=16)

            # 1. Matriz de confusión
            cm = metrics['confusion_matrix']

            # ✅ CORRECCIÓN: Usar matplotlib si seaborn no está disponible
            if SEABORN_AVAILABLE:
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=self.class_names, yticklabels=self.class_names,
                           ax=axes[0,0])
            else:
                # Usar matplotlib básico para matriz de confusión
                im = axes[0,0].imshow(cm, cmap='Blues', interpolation='nearest')
                axes[0,0].set_xticks(range(len(self.class_names)))
                axes[0,0].set_yticks(range(len(self.class_names)))
                axes[0,0].set_xticklabels(self.class_names)
                axes[0,0].set_yticklabels(self.class_names)

                # Agregar texto en las celdas
                for i in range(len(self.class_names)):
                    for j in range(len(self.class_names)):
                        text = axes[0,0].text(j, i, str(cm[i, j]),
                                             ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")

                plt.colorbar(im, ax=axes[0,0])

            axes[0,0].set_title('Matriz de Confusión')
            axes[0,0].set_ylabel('Real')
            axes[0,0].set_xlabel('Predicción')

            # 2. Métricas por clase
            classes = list(metrics['precision_per_class'].keys())
            precision_values = list(metrics['precision_per_class'].values())
            recall_values = list(metrics['recall_per_class'].values())
            f1_values = list(metrics['f1_per_class'].values())

            x = np.arange(len(classes))
            width = 0.25

            axes[0,1].bar(x - width, precision_values, width, label='Precision', color='#ff6b6b')
            axes[0,1].bar(x, recall_values, width, label='Recall', color='#4ecdc4')
            axes[0,1].bar(x + width, f1_values, width, label='F1-Score', color='#45b7d1')

            axes[0,1].set_xlabel('Clases')
            axes[0,1].set_ylabel('Score')
            axes[0,1].set_title('Métricas por Clase')
            axes[0,1].set_xticks(x)
            axes[0,1].set_xticklabels(classes)
            axes[0,1].legend()

            # 3. Distribución de predicciones (simplificada)
            # Como no tenemos las predicciones reales, mostrar distribución de clases
            class_counts = [metrics['support_per_class'][name] for name in self.class_names]
            axes[1,0].pie(class_counts, labels=self.class_names, autopct='%1.1f%%',
                         colors=self.class_colors)
            axes[1,0].set_title('Distribución de Clases')

            # 4. Métricas de confianza (si están disponibles)
            if 'avg_confidence_correct' in metrics:
                conf_metrics = ['Correctas', 'Incorrectas']
                conf_values = [metrics['avg_confidence_correct'], metrics['avg_confidence_incorrect']]
                axes[1,1].bar(conf_metrics, conf_values, color=['#4ecdc4', '#ff6b6b'])
                axes[1,1].set_title('Confianza Promedio')
                axes[1,1].set_ylabel('Confianza')
            else:
                axes[1,1].text(0.5, 0.5, 'Métricas de confianza\nno disponibles',
                              ha='center', va='center', transform=axes[1,1].transAxes)
                axes[1,1].set_title('Métricas de Confianza')

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ Gráfico guardado: {save_path}")

        except Exception as e:
            print(f"⚠️  Error guardando gráfico: {e}")
            print(f"   📊 Error específico: {str(e)}")
            print(f"   💡 El entrenamiento continuará sin gráfico")
            print(f"   📊 Métricas disponibles en: {save_path.replace('.png', '.json')}")


class MarketRegimeSelector:
    """
    🚀 TCN ADAPTATIVE TRAINER - VERSIÓN CONFIGURABLE CON ARQUITECTURA TCN MEJORADA
    Entrenador con thresholds adaptativos, profit-aware loss y nueva arquitectura TCN

✅ NUEVA FUNCIONALIDAD: ARQUITECTURA TCN MEJORADA (2024)
- Bloques TCN residuales con normalización de capas
- Atención temporal específica para crypto con ponderación por volatilidad
- Procesamiento multi-escala con dilataciones crecientes (1, 2, 4, 8, 16, 32)
- Activación Swish para mejor convergencia
- Arquitectura más profunda y robusta (~200K parámetros)

✅ FUNCIONALIDADES EXISTENTES:
- Profit-aware loss functions para maximizar rentabilidad
- Selección de regímenes de mercado equilibrados
- Thresholds adaptativos optimizados
- Features críticas restauradas (macd_signal, bb_middle, sma_20)
- Percentiles dinámicos corregidos (BUY: 15%, SELL: 15%, HOLD: 70%)

🎯 ARQUITECTURAS TCN DISPONIBLES:
1. 'enhanced': Nueva arquitectura TCN mejorada (RECOMENDADO)
2. 'original': Arquitectura crypto optimizada probada
3. 'hybrid': Combinación de ambas arquitecturas
4. 'efficient_v3': Arquitectura TCN V3 eficiente y rápida (NUEVA - 2-3x más rápido)

🎯 EJEMPLOS DE USO:

1. ENTRENAMIENTO CON TCN MEJORADO:
   python tcn_adaptative_trainer_v2.py --non_interactive --enhanced_tcn

2. ENTRENAMIENTO COMPLETO OPTIMIZADO:
   python tcn_adaptative_trainer_v2.py --non_interactive --profit_aware_loss --balanced_regimes --enhanced_tcn

3. ENTRENAMIENTO RÁPIDO CON TCN V3 EFICIENTE:
   python tcn_adaptative_trainer_v2.py --non_interactive --use_efficient_tcn_v3

4. ENTRENAMIENTO ULTRA-RÁPIDO PARA PROTOTIPADO:
   python tcn_adaptative_trainer_v2.py --non_interactive --use_efficient_tcn_v3 --tcn_v3_filters 32 --tcn_v3_dilations 1,2,4

🎯 BENEFICIOS DE LA ARQUITECTURA TCN MEJORADA:
- Mejor captura de patrones temporales complejos
- Convergencia más estable durante el entrenamiento
- Manejo robusto de volatilidad del mercado
- Arquitectura especializada para trading de crypto
- Bloques residuales para entrenamiento más profundo

🎯 BENEFICIOS DE LA ARQUITECTURA TCN V3 EFICIENTE:
- Entrenamiento 2-3x más rápido que enhanced
- Menor uso de memoria GPU/RAM
- Mejor generalización en datasets pequeños
- Ideal para prototipado y experimentación rápida
- Arquitectura limpia y fácil de mantener
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd
import tensorflow as tf
import time
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import talib
import warnings
import pickle
import os
from typing import List, Optional, Union, Dict, Tuple
from collections import Counter
warnings.filterwarnings('ignore')

# Importar motor de features actual (sin cambios)
from centralized_features_engine3 import CentralizedFeaturesEngine

# ✅ NUEVAS IMPORTACIONES PARA PROFIT-AWARE LOSS Y MÉTRICAS AVANZADAS
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score

# Importación condicional de profit-aware loss
try:
    from profit_aware_loss import create_profit_aware_loss, prepare_future_returns_tensor, evaluate_profit_performance
    PROFIT_AWARE_AVAILABLE = True
except ImportError:
    print("⚠️ Profit-aware loss no disponible, usando funciones estándar")
    PROFIT_AWARE_AVAILABLE = False

# Importación condicional de métricas mejoradas
try:
    from trading_metrics_enhanced import TradingMetricsEnhanced, evaluate_model_with_enhanced_metrics
    ENHANCED_METRICS_AVAILABLE = True
except ImportError:
    print("⚠️ Métricas mejoradas no disponibles, usando funciones estándar")
    ENHANCED_METRICS_AVAILABLE = False

# ✅ NUEVA IMPORTACIÓN PARA DETECCIÓN DE REGÍMENES DE MERCADO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ✅ IMPORTACIONES OPCIONALES PARA VISUALIZACIÓN
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib no disponible, gráficos deshabilitados")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    if MATPLOTLIB_AVAILABLE:
        print("⚠️  seaborn no disponible, usando matplotlib básico para gráficos")
    else:
        print("⚠️  seaborn no disponible, gráficos deshabilitados")

# ✅ IMPORTACIÓN OPCIONAL DE PSUTIL
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️  psutil no disponible, monitoreo de memoria deshabilitado")

# Variable global para controlar si se pueden generar gráficos
PLOTTING_AVAILABLE = MATPLOTLIB_AVAILABLE


# ✅ ELIMINADAS LAS FUNCIONES PERSONALIZADAS QUE CAUSABAN ERRORES
# Usar únicamente funciones estándar de Keras para máxima estabilidad


class TrainingConfig:
    """🔧 Configuración completa de entrenamiento - TOTALMENTE CONFIGURABLE"""

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
        self.use_adaptive_thresholds = False  # 🎯 FORZAR thresholds fijos rentables
        
        # 🚀 NUEVO: SISTEMA DE PERCENTILES DINÁMICOS (PROFESIONAL)
        self.use_dynamic_percentiles = True  # Usar percentiles en lugar de thresholds fijos
        self.percentile_window = 1000  # Ventana rolling para calcular percentiles
        self.buy_percentile = 85   # Top 15% = BUY (percentil 85+)
        self.sell_percentile = 15  # Bottom 15% = SELL (percentil 0-15)
        self.min_samples_for_percentiles = 500  # Mínimo de datos para usar percentiles
        
        # 🎯 FEATURE SET - AGREGADO
        self.feature_set = 'tcn_definitivo_v3_enhanced'  # Por defecto - V3 Enhanced con detección bajista
        
        # 🚀 NUEVA ARQUITECTURA TCN MEJORADA
        self.use_enhanced_tcn = True  # Usar arquitectura TCN mejorada por defecto
        self.tcn_architecture = 'enhanced'  # 'enhanced', 'original', 'hybrid'

        # ✅ NUEVO: SELECCIÓN DE REGÍMENES DE MERCADO EQUILIBRADOS
        self.use_balanced_regimes = False  # Por defecto deshabilitado
        self.regime_balance_method = 'auto'  # 'auto', 'manual', 'stratified'
        self.target_samples_per_regime = None  # Para método manual
        
        # 🎯 NUEVO: PROFIT-AWARE LOSS CONFIGURATION
        self.use_profit_aware_loss = True  # Por defecto HABILITADO para maximizar rentabilidad
        self.loss_type = 'profit_weighted'  # Más agresivo en rentabilidad
        self.base_fee = 0.001  # 0.1% fee por trade
        self.spread_cost = 0.0005  # 0.05% spread estimado
        self.profit_amplifier = 3.0  # MÁS agresivo en recompensas por trades rentables
        self.loss_amplifier = 2.5  # MÁS agresivo en penalizaciones por pérdidas
        self.use_enhanced_metrics = True  # Usar métricas de trading mejoradas
        
        # ✅ NUEVO: SELECCIÓN POR PERÍODOS DE FECHAS ESPECÍFICOS
        self.use_date_periods = False  # Por defecto deshabilitado
        self.date_periods = []  # Lista de períodos de fechas para selección
        self.date_periods_method = 'manual'  # 'manual', 'preset', 'custom'
        
        # ✅ PERÍODOS PREDEFINIDOS PARA DIFERENTES REGÍMENES DE MERCADO
        self.preset_date_periods = {
            'crypto_bull_bear_2021_2023': [
                {'start_date': '2021-01-01', 'end_date': '2021-11-30', 'description': 'Mercado alcista 2021'},
                {'start_date': '2022-01-01', 'end_date': '2022-12-31', 'description': 'Mercado bajista 2022'},
                {'start_date': '2023-01-01', 'end_date': '2023-06-30', 'description': 'Recuperación 2023'}
            ],
            'crypto_volatility_2020_2023': [
                {'start_date': '2020-03-01', 'end_date': '2020-12-31', 'description': 'COVID y recuperación'},
                {'start_date': '2021-01-01', 'end_date': '2021-12-31', 'description': 'Bull market completo'},
                {'start_date': '2022-01-01', 'end_date': '2022-12-31', 'description': 'Bear market completo'},
                {'start_date': '2023-01-01', 'end_date': '2023-12-31', 'description': 'Mercado mixto 2023'}
            ],
            'btc_cycles_2017_2023': [
                {'start_date': '2017-01-01', 'end_date': '2017-12-31', 'description': 'Bull run 2017'},
                {'start_date': '2018-01-01', 'end_date': '2018-12-31', 'description': 'Bear market 2018'},
                {'start_date': '2019-01-01', 'end_date': '2020-02-29', 'description': 'Acumulación 2019-2020'},
                {'start_date': '2020-03-01', 'end_date': '2021-11-30', 'description': 'COVID bull run'},
                {'start_date': '2022-01-01', 'end_date': '2022-12-31', 'description': 'Crypto winter 2022'},
                {'start_date': '2023-01-01', 'end_date': '2023-12-31', 'description': 'Recuperación 2023'}
            ]
        }
        
        # ✅ NUEVO: ATRIBUTO PRESET_PERIODS PARA COMPATIBILIDAD
        self.preset_periods = None  # Se establece cuando se selecciona un preset

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

        # ✅ NUEVO: MANEJAR PARÁMETROS DE PERÍODOS DE FECHAS
        if hasattr(args, 'use_date_periods') and args.use_date_periods:
            self.use_date_periods = True
            
        if hasattr(args, 'date_periods_method') and args.date_periods_method:
            if args.date_periods_method in ['manual', 'preset', 'custom']:
                self.date_periods_method = args.date_periods_method
                
        if hasattr(args, 'preset_periods') and args.preset_periods:
            if args.preset_periods in self.preset_date_periods:
                self.date_periods = self.preset_date_periods[args.preset_periods]
                self.date_periods_method = 'preset'
                self.preset_periods = args.preset_periods  # ✅ AGREGADO: Establecer preset_periods
                print(f"✅ Períodos predefinidos cargados: {args.preset_periods}")
                
        if hasattr(args, 'custom_periods') and args.custom_periods:
            try:
                # Formato esperado: "2023-01-01:2023-03-31,2023-04-01:2023-06-30"
                periods = []
                for period_str in args.custom_periods.split(','):
                    start, end = period_str.split(':')
                    periods.append({
                        'start_date': start.strip(),
                        'end_date': end.strip(),
                        'description': f'Período personalizado {start} a {end}'
                    })
                self.date_periods = periods
                self.date_periods_method = 'custom'
                print(f"✅ {len(periods)} períodos personalizados cargados")
            except Exception as e:
                print(f"⚠️  Error procesando períodos personalizados: {e}")
                print(f"   📋 Formato esperado: '2023-01-01:2023-03-31,2023-04-01:2023-06-30'")

    def print_config(self):
        """Mostrar configuración actual"""
        print("\n🔧 CONFIGURACIÓN DE ENTRENAMIENTO:")
        print("=" * 50)
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
        
        # 🎯 NUEVO: MOSTRAR CONFIGURACIÓN DE PROFIT-AWARE LOSS
        use_profit_aware = getattr(self, 'use_profit_aware_loss', False)
        print(f"🎯 Profit-Aware Loss: {'✅' if use_profit_aware else '❌'}")
        if use_profit_aware:
            print(f"   📊 Tipo de loss: {getattr(self, 'loss_type', 'combined')}")
            print(f"   💰 Fee base: {getattr(self, 'base_fee', 0.001):.3f} ({getattr(self, 'base_fee', 0.001)*100:.1f}%)")
            print(f"   📊 Spread cost: {getattr(self, 'spread_cost', 0.0005):.3f} ({getattr(self, 'spread_cost', 0.0005)*100:.2f}%)")
            print(f"   📈 Amplificador profit: {getattr(self, 'profit_amplifier', 2.0)}x")
            print(f"   📉 Amplificador loss: {getattr(self, 'loss_amplifier', 1.5)}x")
            print(f"   📊 Métricas mejoradas: {'✅' if getattr(self, 'use_enhanced_metrics', True) else '❌'}")
        
        # ✅ NUEVO: MOSTRAR CONFIGURACIÓN DE PERÍODOS DE FECHAS
        print(f"📅 Períodos de fechas: {'✅' if self.use_date_periods else '❌'}")
        if self.use_date_periods:
            print(f"   📊 Método: {self.date_periods_method}")
            if self.date_periods:
                print(f"   📅 Períodos configurados: {len(self.date_periods)}")
                for i, period in enumerate(self.date_periods[:3]):  # Mostrar solo los primeros 3
                    start = period.get('start_date', 'N/A')
                    end = period.get('end_date', 'N/A')
                    desc = period.get('description', 'Sin descripción')
                    print(f"      {i+1}. {start} a {end}: {desc}")
                if len(self.date_periods) > 3:
                    print(f"      ... y {len(self.date_periods) - 3} períodos más")
            elif self.date_periods_method == 'preset':
                print(f"   📅 Usando períodos predefinidos")
        print("=" * 50)


class TradingMetrics:
    """📊 Métricas específicas para trading con análisis detallado por clase"""

    def __init__(self):
        self.class_names = ['SELL', 'HOLD', 'BUY']
        self.class_colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']

    def calculate_trading_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                y_pred_proba: np.ndarray = None) -> Dict:
        """🎯 Calcular métricas específicas para trading"""

        # Métricas básicas
        accuracy = np.mean(y_true == y_pred)

        # Reporte de clasificación detallado
        report = classification_report(y_true, y_pred,
                                    target_names=self.class_names,
                                    output_dict=True)

        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)

        # Métricas por clase
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        # ✅ MÉTRICAS ESPECÍFICAS PARA TRADING
        trading_metrics = {
            'accuracy': accuracy,
            'precision_per_class': dict(zip(self.class_names, precision)),
            'recall_per_class': dict(zip(self.class_names, recall)),
            'f1_per_class': dict(zip(self.class_names, f1)),
            'support_per_class': dict(zip(self.class_names, support)),
            'confusion_matrix': cm,
            'classification_report': report,
            'total_samples': len(y_true)
        }

        # ✅ MÉTRICAS DE CONFIANZA (si hay probabilidades)
        if y_pred_proba is not None:
            confidence_metrics = self.calculate_confidence_metrics(y_true, y_pred, y_pred_proba)
            trading_metrics.update(confidence_metrics)

        return trading_metrics

    def calculate_confidence_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   y_pred_proba: np.ndarray) -> Dict:
        """🎯 Calcular métricas de confianza de las predicciones"""

        # Confianza promedio por predicción correcta/incorrecta
        correct_mask = y_true == y_pred
        incorrect_mask = ~correct_mask

        confidence_metrics = {
            'avg_confidence_correct': np.mean(np.max(y_pred_proba[correct_mask], axis=1)) if np.any(correct_mask) else 0,
            'avg_confidence_incorrect': np.mean(np.max(y_pred_proba[incorrect_mask], axis=1)) if np.any(incorrect_mask) else 0,
            'confidence_threshold_80': np.mean(np.max(y_pred_proba, axis=1) > 0.8),
            'confidence_threshold_90': np.mean(np.max(y_pred_proba, axis=1) > 0.9),
            'high_confidence_accuracy': self.calculate_high_confidence_accuracy(y_true, y_pred, y_pred_proba, threshold=0.8)
        }

        return confidence_metrics

    def calculate_high_confidence_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray,
                                         y_pred_proba: np.ndarray, threshold: float = 0.8) -> float:
        """🎯 Calcular accuracy solo para predicciones con alta confianza"""
        high_conf_mask = np.max(y_pred_proba, axis=1) > threshold
        if np.any(high_conf_mask):
            return np.mean(y_true[high_conf_mask] == y_pred[high_conf_mask])
        return 0.0

    def print_trading_report(self, metrics: Dict, symbol: str, timeframe: str):
        """📊 Imprimir reporte detallado de métricas de trading"""

        print(f"\n📊 REPORTE DE MÉTRICAS DE TRADING - {symbol} ({timeframe})")
        print("=" * 70)

        # Accuracy general
        print(f"🎯 ACCURACY GENERAL: {metrics['accuracy']:.3f}")

        # Métricas por clase
        print(f"\n📈 MÉTRICAS POR CLASE:")
        for i, class_name in enumerate(self.class_names):
            precision = metrics['precision_per_class'][class_name]
            recall = metrics['recall_per_class'][class_name]
            f1 = metrics['f1_per_class'][class_name]
            support = metrics['support_per_class'][class_name]

            print(f"   {class_name:>5}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, Support={support}")

        # Métricas de confianza
        if 'avg_confidence_correct' in metrics:
            print(f"\n🎯 MÉTRICAS DE CONFIANZA:")
            print(f"   Confianza promedio (correctas): {metrics['avg_confidence_correct']:.3f}")
            print(f"   Confianza promedio (incorrectas): {metrics['avg_confidence_incorrect']:.3f}")
            print(f"   Predicciones >80% confianza: {metrics['confidence_threshold_80']:.1%}")
            print(f"   Predicciones >90% confianza: {metrics['confidence_threshold_90']:.1%}")
            print(f"   Accuracy alta confianza (>80%): {metrics['high_confidence_accuracy']:.3f}")

        # Análisis de trading
        self.print_trading_analysis(metrics, symbol)

    def print_trading_analysis(self, metrics: Dict, symbol: str):
        """🎯 Análisis específico para trading"""

        print(f"\n🎯 ANÁLISIS DE TRADING - {symbol}:")

        # Análisis de señales de compra
        buy_precision = metrics['precision_per_class']['BUY']
        buy_recall = metrics['recall_per_class']['BUY']

        if buy_precision > 0.6 and buy_recall > 0.5:
            print(f"   ✅ BUY: Buena precisión ({buy_precision:.3f}) y recall ({buy_recall:.3f})")
        elif buy_precision < 0.4:
            print(f"   ⚠️  BUY: Baja precisión ({buy_precision:.3f}) - muchas falsas alarmas")
        elif buy_recall < 0.3:
            print(f"   ⚠️  BUY: Bajo recall ({buy_recall:.3f}) - se pierden oportunidades")

        # Análisis de señales de venta
        sell_precision = metrics['precision_per_class']['SELL']
        sell_recall = metrics['recall_per_class']['SELL']

        if sell_precision > 0.6 and sell_recall > 0.5:
            print(f"   ✅ SELL: Buena precisión ({sell_precision:.3f}) y recall ({sell_recall:.3f})")
        elif sell_precision < 0.4:
            print(f"   ⚠️  SELL: Baja precisión ({sell_precision:.3f}) - muchas falsas alarmas")
        elif sell_recall < 0.3:
            print(f"   ⚠️  SELL: Bajo recall ({sell_recall:.3f}) - se pierden oportunidades")

        # Análisis de HOLD
        hold_f1 = metrics['f1_per_class']['HOLD']
        if hold_f1 > 0.6:
            print(f"   ✅ HOLD: Buen balance ({hold_f1:.3f})")
        else:
            print(f"   ⚠️  HOLD: Balance pobre ({hold_f1:.3f})")

    def save_metrics_plot(self, metrics: Dict, symbol: str, timeframe: str, save_path: str):
        """📊 Guardar gráfico de métricas (opcional)"""

        # ✅ CORRECCIÓN: Verificar si se pueden generar gráficos
        if not PLOTTING_AVAILABLE:
            print(f"⚠️  Gráficos deshabilitados - matplotlib no disponible")
            print(f"   📊 Métricas disponibles en: {save_path.replace('.png', '.json')}")
            return

        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Métricas de Trading - {symbol} ({timeframe})', fontsize=16)

            # 1. Matriz de confusión
            cm = metrics['confusion_matrix']

            # ✅ CORRECCIÓN: Usar matplotlib si seaborn no está disponible
            if SEABORN_AVAILABLE:
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=self.class_names, yticklabels=self.class_names,
                           ax=axes[0,0])
            else:
                # Usar matplotlib básico para matriz de confusión
                im = axes[0,0].imshow(cm, cmap='Blues', interpolation='nearest')
                axes[0,0].set_xticks(range(len(self.class_names)))
                axes[0,0].set_yticks(range(len(self.class_names)))
                axes[0,0].set_xticklabels(self.class_names)
                axes[0,0].set_yticklabels(self.class_names)

                # Agregar texto en las celdas
                for i in range(len(self.class_names)):
                    for j in range(len(self.class_names)):
                        text = axes[0,0].text(j, i, str(cm[i, j]),
                                             ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")

                plt.colorbar(im, ax=axes[0,0])

            axes[0,0].set_title('Matriz de Confusión')
            axes[0,0].set_ylabel('Real')
            axes[0,0].set_xlabel('Predicción')

            # 2. Métricas por clase
            classes = list(metrics['precision_per_class'].keys())
            precision_values = list(metrics['precision_per_class'].values())
            recall_values = list(metrics['recall_per_class'].values())
            f1_values = list(metrics['f1_per_class'].values())

            x = np.arange(len(classes))
            width = 0.25

            axes[0,1].bar(x - width, precision_values, width, label='Precision', color='#ff6b6b')
            axes[0,1].bar(x, recall_values, width, label='Recall', color='#4ecdc4')
            axes[0,1].bar(x + width, f1_values, width, label='F1-Score', color='#45b7d1')

            axes[0,1].set_xlabel('Clases')
            axes[0,1].set_ylabel('Score')
            axes[0,1].set_title('Métricas por Clase')
            axes[0,1].set_xticks(x)
            axes[0,1].set_xticklabels(classes)
            axes[0,1].legend()

            # 3. Distribución de predicciones (simplificada)
            # Como no tenemos las predicciones reales, mostrar distribución de clases
            class_counts = [metrics['support_per_class'][name] for name in self.class_names]
            axes[1,0].pie(class_counts, labels=self.class_names, autopct='%1.1f%%',
                         colors=self.class_colors)
            axes[1,0].set_title('Distribución de Clases')

            # 4. Métricas de confianza (si están disponibles)
            if 'avg_confidence_correct' in metrics:
                conf_metrics = ['Correctas', 'Incorrectas']
                conf_values = [metrics['avg_confidence_correct'], metrics['avg_confidence_incorrect']]
                axes[1,1].bar(conf_metrics, conf_values, color=['#4ecdc4', '#ff6b6b'])
                axes[1,1].set_title('Confianza Promedio')
                axes[1,1].set_ylabel('Confianza')
            else:
                axes[1,1].text(0.5, 0.5, 'Métricas de confianza\nno disponibles',
                              ha='center', va='center', transform=axes[1,1].transAxes)
                axes[1,1].set_title('Métricas de Confianza')

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ Gráfico guardado: {save_path}")

        except Exception as e:
            print(f"⚠️  Error guardando gráfico: {e}")
            print(f"   📊 Error específico: {str(e)}")
            print(f"   💡 El entrenamiento continuará sin gráfico")
            print(f"   📊 Métricas disponibles en: {save_path.replace('.png', '.json')}")


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
            # ✅ PROTECCIÓN: Asegurar que np.mean esté disponible
            import numpy as np
            if not callable(np.mean):
                print(f"⚠️  np.mean no está disponible, usando fallback")
                return self._create_fallback_regimes(df, symbol)
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
        
        # ✅ PROTECCIÓN: Asegurar que numpy esté disponible
        import numpy as np
        
        regime_centers = {}
        for regime_id in range(3):
            regime_mask = labels == regime_id
            if np.any(regime_mask):
                regime_features = features[regime_mask]
                # Usar numpy.mean de forma explícita para evitar conflictos
                try:
                    regime_centers[regime_id] = np.array(regime_features).mean(axis=0)
                except Exception as e:
                    print(f"⚠️  Error calculando media para régimen {regime_id}: {e}")
                    # Usar método alternativo
                    regime_centers[regime_id] = np.sum(regime_features, axis=0) / len(regime_features)
        
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
                                  target_samples_per_regime: int = None,
                                  date_periods: List[Dict] = None) -> pd.DataFrame:
        """
        📅 Seleccionar datos de períodos de fechas específicos para entrenamiento equilibrado
        
        ✅ NUEVA FUNCIONALIDAD: SELECCIÓN POR PERÍODOS DE FECHAS
        - Permite especificar fechas de inicio y fin para cada período
        - Ideal para capturar mercados en diferentes regímenes
        - Más control que el muestreo automático por regímenes
        
        📋 PARÁMETROS:
        - df: DataFrame con datos históricos
        - symbol: Símbolo del par de trading
        - target_samples_per_regime: Mantenido por compatibilidad (no usado)
        - date_periods: Lista de diccionarios con 'start_date' y 'end_date'
        
        📅 FORMATO DE date_periods:
        [
            {'start_date': '2023-01-01', 'end_date': '2023-03-31'},  # Mercado alcista
            {'start_date': '2023-04-01', 'end_date': '2023-06-30'},  # Mercado bajista
            {'start_date': '2023-07-01', 'end_date': '2023-09-30'}   # Mercado lateral
        ]
        """
        
        print(f"📅 Seleccionando datos por períodos de fechas para {symbol}...")
        
        # ✅ DIAGNÓSTICO: Verificar estructura del DataFrame
        print(f"🔍 Diagnóstico del DataFrame para {symbol}:")
        print(f"   📊 Forma: {df.shape}")
        print(f"   📋 Columnas: {list(df.columns)}")
        print(f"   📅 Índice tipo: {type(df.index)}")
        
        if 'timestamp' in df.columns:
            print(f"   🕐 Timestamp tipo: {df['timestamp'].dtype}")
            print(f"   🕐 Timestamp rango: {df['timestamp'].min()} - {df['timestamp'].max()}")
        else:
            print(f"   ⚠️  No se encontró columna timestamp")
            if isinstance(df.index, pd.DatetimeIndex):
                print(f"   🕐 Índice temporal: {df.index.min()} - {df.index.max()}")
            else:
                print(f"   ⚠️  Índice no es temporal: {df.index.dtype}")
        
        # ✅ NUEVO: SELECCIÓN POR PERÍODOS DE FECHAS
        if date_periods and len(date_periods) > 0:
            return self._select_data_by_date_periods(df, symbol, date_periods)
        
        # ✅ MANTENER COMPATIBILIDAD: Usar método anterior si no hay períodos
        print(f"⚠️  No se especificaron períodos de fechas, usando método anterior...")
        return self._select_data_by_regimes_fallback(df, symbol, target_samples_per_regime)
    
    def _select_data_by_date_periods(self, df: pd.DataFrame, symbol: str, 
                                   date_periods: List[Dict]) -> pd.DataFrame:
        """📅 Seleccionar datos de períodos de fechas específicos"""
        
        print(f"📅 Aplicando selección por {len(date_periods)} períodos de fechas...")
        
        # ✅ PREPARAR DATAFRAME CON TIMESTAMP
        df_working = df.copy()
        
        # Asegurar que timestamp sea datetime
        if 'timestamp' in df_working.columns:
            if not pd.api.types.is_datetime64_any_dtype(df_working['timestamp']):
                df_working['timestamp'] = pd.to_datetime(df_working['timestamp'])
        else:
            # Si no hay timestamp, usar el índice
            if isinstance(df_working.index, pd.DatetimeIndex):
                df_working['timestamp'] = df_working.index
            else:
                # Intentar convertir el índice a datetime
                try:
                    df_working['timestamp'] = pd.to_datetime(df_working.index)
                except:
                    # Si falla, crear un timestamp basado en la posición
                    print(f"   ⚠️  No se pudo convertir índice a timestamp, creando timestamp sintético")
                    df_working['timestamp'] = pd.date_range(start='2020-01-01', periods=len(df_working), freq='5min')
        
        # ✅ SELECCIONAR DATOS DE CADA PERÍODO
        selected_data = []
        total_samples = 0
        
        for i, period in enumerate(date_periods):
            try:
                start_date = pd.to_datetime(period['start_date'])
                end_date = pd.to_datetime(period['end_date'])
                
                print(f"   📅 Período {i+1}: {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}")
                
                # Filtrar datos del período
                period_mask = (df_working['timestamp'] >= start_date) & (df_working['timestamp'] <= end_date)
                period_data = df_working[period_mask]
                
                if len(period_data) == 0:
                    print(f"      ⚠️  Sin datos en este período")
                    continue
                
                print(f"      ✅ {len(period_data)} muestras encontradas")
                selected_data.append(period_data)
                total_samples += len(period_data)
                
            except Exception as e:
                print(f"      ❌ Error procesando período {i+1}: {e}")
                continue
        
        # ✅ COMBINAR DATOS SELECCIONADOS
        if selected_data:
            final_df = pd.concat(selected_data, ignore_index=True)
            final_df = final_df.sort_values('timestamp').reset_index(drop=True)
            
            print(f"\n📊 RESUMEN DE SELECCIÓN POR PERÍODOS - {symbol}:")
            print("=" * 50)
            print(f"   📅 Períodos procesados: {len(selected_data)}")
            print(f"   📊 Total muestras: {total_samples}")
            print(f"   🕐 Rango temporal: {final_df['timestamp'].min().strftime('%Y-%m-%d')} a {final_df['timestamp'].max().strftime('%Y-%m-%d')}")
            
            # ✅ OPCIONAL: Detectar regímenes para análisis
            if len(final_df) > 100:  # Solo si hay suficientes datos
                try:
                    df_with_regimes = self.detect_market_regimes(final_df, symbol)
                    if not df_with_regimes.empty:
                        regime_counts = df_with_regimes['market_regime'].value_counts().sort_index()
                        print(f"\n📈 DISTRIBUCIÓN DE REGÍMENES EN PERÍODOS SELECCIONADOS:")
                        for regime_id in range(3):
                            count = regime_counts.get(regime_id, 0)
                            regime_name = self.regime_names[regime_id]
                            print(f"   {regime_name:>8}: {count:>4} muestras")
                except Exception as e:
                    print(f"   ⚠️  No se pudieron detectar regímenes: {e}")
            
            print(f"✅ Datos seleccionados por períodos para {symbol}")
            return final_df
        else:
            print(f"❌ ERROR: No se pudieron seleccionar datos de ningún período para {symbol}")
            return df
    
    def _select_data_by_regimes_fallback(self, df: pd.DataFrame, symbol: str, 
                                       target_samples_per_regime: int = None) -> pd.DataFrame:
        """🔄 Método de fallback usando selección por regímenes (método anterior)"""
        
        print(f"🔄 Usando método de regímenes como fallback...")
        
        # Detectar regímenes si no están presentes
        if 'market_regime' not in df.columns:
            df = self.detect_market_regimes(df, symbol)
        
        if df.empty:
            print(f"❌ ERROR: No se pudieron detectar regímenes para {symbol}")
            return df
        
        # Calcular muestras objetivo por régimen
        if target_samples_per_regime is None:
            regime_counts = df['market_regime'].value_counts()
            target_samples_per_regime = min(regime_counts.values)
            print(f"📊 Objetivo automático: {target_samples_per_regime} muestras por régimen")
        else:
            print(f"📊 Objetivo manual: {target_samples_per_regime} muestras por régimen")
        
        # Seleccionar muestras equilibradas
        balanced_data = []
        
        for regime_id in range(3):
            regime_mask = df['market_regime'] == regime_id
            regime_data = df[regime_mask]
            
            if len(regime_data) == 0:
                print(f"⚠️  Régimen {self.regime_names[regime_id]}: Sin datos")
                continue
            
            if len(regime_data) <= target_samples_per_regime:
                selected_data = regime_data
                print(f"   {self.regime_names[regime_id]:>8}: {len(selected_data)} muestras (todas disponibles)")
            else:
                selected_data = self._select_stratified_samples(regime_data, target_samples_per_regime)
                print(f"   {self.regime_names[regime_id]:>8}: {len(selected_data)} muestras (seleccionadas de {len(regime_data)})")
            
            balanced_data.append(selected_data)
        
        # Combinar datos equilibrados
        if balanced_data:
            final_df = pd.concat(balanced_data, ignore_index=True)
            
            if 'timestamp' in final_df.columns:
                try:
                    if not pd.api.types.is_datetime64_any_dtype(final_df['timestamp']):
                        final_df['timestamp'] = pd.to_datetime(final_df['timestamp'])
                    final_df = final_df.sort_values('timestamp').reset_index(drop=True)
                except Exception as e:
                    print(f"⚠️  Error procesando timestamp: {e}")
                    final_df = final_df.reset_index(drop=True)
            else:
                print("⚠️  No se encontró columna timestamp, usando índice numérico")
                final_df = final_df.reset_index(drop=True)
            
            # Verificar balance final
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
        
        # ✅ ORDENAR POR TIMESTAMP CON MANEJO DE ERRORES
        try:
            if 'timestamp' in regime_data.columns:
                # Verificar que timestamp sea datetime
                if not pd.api.types.is_datetime64_any_dtype(regime_data['timestamp']):
                    regime_data['timestamp'] = pd.to_datetime(regime_data['timestamp'])
                
                regime_data = regime_data.sort_values('timestamp').reset_index(drop=True)
            else:
                # Si no hay timestamp, usar índice
                print("⚠️  No se encontró columna timestamp, usando índice para estratificación")
                regime_data = regime_data.reset_index(drop=True)
        except Exception as e:
            print(f"⚠️  Error ordenando por timestamp: {e}, usando índice")
            regime_data = regime_data.reset_index(drop=True)
        
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
    
    def visualize_regime_distribution(self, df: pd.DataFrame, symbol: str, save_path: str = None):
        """📊 Visualizar distribución de regímenes de mercado"""
        
        if not PLOTTING_AVAILABLE:
            print(f"⚠️  Gráficos deshabilitados para visualización de regímenes")
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Análisis de Regímenes de Mercado - {symbol}', fontsize=16)
            
            # 1. Distribución de regímenes (pie chart)
            regime_counts = df['market_regime'].value_counts().sort_index()
            regime_labels = [self.regime_names[i] for i in regime_counts.index]
            regime_colors = [self.regime_colors[i] for i in regime_counts.index]
            
            axes[0,0].pie(regime_counts.values, labels=regime_labels, colors=regime_colors, autopct='%1.1f%%')
            axes[0,0].set_title('Distribución de Regímenes')
            
            # 2. Evolución temporal de regímenes
            df_sorted = df.sort_values('timestamp')
            axes[0,1].scatter(range(len(df_sorted)), df_sorted['market_regime'], 
                             c=[self.regime_colors[r] for r in df_sorted['market_regime']], 
                             alpha=0.6, s=20)
            axes[0,1].set_title('Evolución Temporal de Regímenes')
            axes[0,1].set_ylabel('Régimen')
            axes[0,1].set_xlabel('Tiempo')
            axes[0,1].set_yticks([0, 1, 2])
            axes[0,1].set_yticklabels(['Bajista', 'Lateral', 'Alcista'])
            
            # 3. Características por régimen (boxplot)
            if 'close' in df.columns:
                close_prices = df['close'].values.astype(float)
                regime_data = []
                regime_labels_plot = []
                
                for regime_id in range(3):
                    regime_mask = df['market_regime'] == regime_id
                    if np.any(regime_mask):
                        regime_prices = close_prices[regime_mask]
                        regime_data.append(regime_prices)
                        regime_labels_plot.append(self.regime_names[regime_id])
                
                if regime_data:
                    bp = axes[1,0].boxplot(regime_data, labels=regime_labels_plot, patch_artist=True)
                    for patch, color in zip(bp['boxes'], [self.regime_colors[i] for i in range(len(regime_data))]):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    axes[1,0].set_title('Distribución de Precios por Régimen')
                    axes[1,0].set_ylabel('Precio')
            
            # 4. Estadísticas por régimen
            regime_stats = []
            for regime_id in range(3):
                regime_mask = df['market_regime'] == regime_id
                if np.any(regime_mask):
                    regime_data = df[regime_mask]
                    stats = {
                        'Régimen': self.regime_names[regime_id],
                        'Muestras': len(regime_data),
                        'Precio Promedio': regime_data['close'].mean() if 'close' in regime_data.columns else 0,
                        'Volatilidad': regime_data['close'].std() if 'close' in regime_data.columns else 0
                    }
                    regime_stats.append(stats)
            
            if regime_stats:
                stats_df = pd.DataFrame(regime_stats)
                axes[1,1].axis('tight')
                axes[1,1].axis('off')
                table = axes[1,1].table(cellText=stats_df.values, colLabels=stats_df.columns, 
                                       cellLoc='center', loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.2, 1.5)
                axes[1,1].set_title('Estadísticas por Régimen')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ Gráfico de regímenes guardado: {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            print(f"⚠️  Error visualizando regímenes: {e}")

    def visualize_date_periods_distribution(self, df: pd.DataFrame, symbol: str, save_path: str = None):
        """
        📅 Visualizar distribución de datos por períodos de fechas seleccionados
        
        ✅ CARACTERÍSTICAS:
        - Gráfico de línea temporal mostrando distribución
        - Histograma de densidad temporal
        - Estadísticas por período
        - Análisis de gaps y continuidad
        """
        
        if not PLOTTING_AVAILABLE:
            print("⚠️  matplotlib no disponible para visualización")
            return
        
        try:
            # Preparar datos para visualización
            if 'timestamp' not in df.columns:
                print("⚠️  No se encontró columna timestamp para visualización")
                return
            
            # Convertir timestamp a datetime si es necesario
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Crear figura con subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Distribución de Datos por Períodos - {symbol}', fontsize=16, fontweight='bold')
            
            # 1. Gráfico de línea temporal
            df_sorted = df.sort_values('timestamp')
            axes[0,0].plot(df_sorted['timestamp'], df_sorted['close'], alpha=0.7, linewidth=0.8)
            axes[0,0].set_title('Precio a lo Largo del Tiempo')
            axes[0,0].set_xlabel('Fecha')
            axes[0,0].set_ylabel('Precio de Cierre')
            axes[0,0].tick_params(axis='x', rotation=45)
            axes[0,0].grid(True, alpha=0.3)
            
            # 2. Histograma de densidad temporal
            df_sorted['date'] = df_sorted['timestamp'].dt.date
            date_counts = df_sorted['date'].value_counts().sort_index()
            
            axes[0,1].bar(range(len(date_counts)), date_counts.values, alpha=0.7, color='skyblue')
            axes[0,1].set_title('Densidad de Datos por Fecha')
            axes[0,1].set_xlabel('Índice de Fecha')
            axes[0,1].set_ylabel('Número de Muestras')
            axes[0,1].grid(True, alpha=0.3)
            
            # 3. Análisis de gaps temporales
            df_sorted['time_diff'] = df_sorted['timestamp'].diff()
            gaps = df_sorted[df_sorted['time_diff'] > pd.Timedelta(minutes=5)]  # Gaps mayores a 5 minutos
            
            if not gaps.empty:
                gap_durations = gaps['time_diff'].dt.total_seconds() / 60  # En minutos
                axes[1,0].hist(gap_durations, bins=20, alpha=0.7, color='orange', edgecolor='black')
                axes[1,0].set_title('Distribución de Gaps Temporales')
                axes[1,0].set_xlabel('Duración del Gap (minutos)')
                axes[1,0].set_ylabel('Frecuencia')
                axes[1,0].grid(True, alpha=0.3)
            else:
                axes[1,0].text(0.5, 0.5, 'Sin gaps temporales\nsignificativos', 
                              ha='center', va='center', transform=axes[1,0].transAxes,
                              fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                axes[1,0].set_title('Análisis de Gaps Temporales')
            
            # 4. Estadísticas de distribución temporal
            total_days = (df_sorted['timestamp'].max() - df_sorted['timestamp'].min()).days
            samples_per_day = len(df_sorted) / max(1, total_days)
            
            stats_text = f"""
            📊 ESTADÍSTICAS TEMPORALES:
            
            📅 Rango total: {total_days} días
            📊 Muestras totales: {len(df_sorted)}
            📈 Muestras por día: {samples_per_day:.1f}
            🕐 Primera muestra: {df_sorted['timestamp'].min().strftime('%Y-%m-%d %H:%M')}
            🕐 Última muestra: {df_sorted['timestamp'].max().strftime('%Y-%m-%d %H:%M')}
            ⏱️  Timeframe: {self.timeframe if hasattr(self, 'timeframe') else 'N/A'}
            """
            
            axes[1,1].axis('off')
            axes[1,1].text(0.05, 0.95, stats_text, transform=axes[1,1].transAxes,
                          fontsize=10, verticalalignment='top',
                          bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ Gráfico de períodos guardado: {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            print(f"⚠️  Error visualizando distribución de períodos: {e}")

    def _visualize_date_periods_distribution(self, df: pd.DataFrame, symbol: str, save_path: str = None):
        """📅 Visualizar distribución de datos por períodos de fechas (método interno)"""
        try:
            # Usar la función del selector de regímenes si está disponible
            if hasattr(self.regime_selector, 'visualize_date_periods_distribution'):
                self.regime_selector.visualize_date_periods_distribution(df, symbol, save_path)
            else:
                print("⚠️  Función de visualización de períodos no disponible en el selector")
        except Exception as e:
            print(f"⚠️  Error en visualización de períodos: {e}")


class AdaptiveTCNTrainer:
    """🎯 Entrenador TCN con configuración totalmente personalizable"""

    def __init__(self, config: TrainingConfig = None):
        # ✅ CONFIGURACIÓN PERSONALIZABLE
        self.config = config if config else TrainingConfig()

        # Usar configuración para parámetros
        self.pairs = self.config.pairs
        self.lookback_window = self.config.lookback_window
        self.prediction_horizon = self.config.prediction_horizon
        self.timeframe = self.config.timeframe
        self.training_days = self.config.training_days
        self.start_date = self.config.start_date
        self.end_date = self.config.end_date
        self.use_adaptive_thresholds = self.config.use_adaptive_thresholds
        self.feature_set = self.config.feature_set  # AGREGADO: Usar feature_set de la configuración

        # Motor de features centralizado
        self.features_engine = CentralizedFeaturesEngine()

        # ✅ SISTEMA DE MÉTRICAS AVANZADAS
        self.trading_metrics = TradingMetrics()
        
        # 🎯 THRESHOLDS RENTABLES (Optimizados para rentabilidad real considerando fees)
        # Fee total: 0.1% trading + 0.05% spread = 0.15% mínimo para breakeven
        # Targets: weak 0.5% (3.3x fees), strong 1.0% (6.7x fees) para rentabilidad sólida
        self.fixed_thresholds = {
            'BTCUSDT': {
                'strong_sell': -0.010, 'weak_sell': -0.005,  # 1.0% / 0.5% (era -0.0025/-0.0015)
                'weak_buy': 0.005, 'strong_buy': 0.010
            },
            'ETHUSDT': {
                'strong_sell': -0.012, 'weak_sell': -0.006,  # 1.2% / 0.6% (ETH más volátil)
                'weak_buy': 0.006, 'strong_buy': 0.012
            },
            'BNBUSDT': {
                'strong_sell': -0.008, 'weak_sell': -0.004,  # 0.8% / 0.4% (BNB menos volátil)
                'weak_buy': 0.004, 'strong_buy': 0.008
            },
            'XRPUSDT': {
                'strong_sell': -0.015, 'weak_sell': -0.007,  # 1.5% / 0.7% (XRP muy volátil)
                'weak_buy': 0.007, 'strong_buy': 0.015
            },
            'ADAUSDT': {    
                'strong_sell': -0.012, 'weak_sell': -0.006,  # 1.2% / 0.6% (ADA volátil)
                'weak_buy': 0.006, 'strong_buy': 0.012
            },
            'DOTUSDT': {
                'strong_sell': -0.006, 'weak_sell': -0.003,  # 0.6% / 0.3% (DOT menos volátil de lo esperado)
                'weak_buy': 0.003, 'strong_buy': 0.006
            }
        }

        # ✅ NUEVO: SELECTOR DE REGÍMENES DE MERCADO
        self.regime_selector = MarketRegimeSelector()
        self.use_balanced_regimes = self.config.use_balanced_regimes
        self.regime_balance_method = self.config.regime_balance_method
        self.target_samples_per_regime = self.config.target_samples_per_regime
        
        # ✅ NUEVO: CONFIGURACIÓN DE PERÍODOS DE FECHAS
        self.use_date_periods = self.config.use_date_periods
        self.date_periods = self.config.date_periods
        self.date_periods_method = self.config.date_periods_method
        
        # 🚀 NUEVO: SISTEMA DE PERCENTILES DINÁMICOS
        self.use_dynamic_percentiles = getattr(self.config, 'use_dynamic_percentiles', True)
        self.percentile_window = getattr(self.config, 'percentile_window', 1000)
        self.buy_percentile = getattr(self.config, 'buy_percentile', 85)
        self.sell_percentile = getattr(self.config, 'sell_percentile', 15)
        self.min_samples_for_percentiles = getattr(self.config, 'min_samples_for_percentiles', 500)
        
        # 🚀 NUEVA ARQUITECTURA TCN MEJORADA
        self.use_enhanced_tcn = getattr(self.config, 'use_enhanced_tcn', True)
        self.tcn_architecture = getattr(self.config, 'tcn_architecture', 'enhanced')
        
        # 🚀 NUEVA ARQUITECTURA TCN V3 EFICIENTE
        self.use_efficient_tcn_v3 = getattr(self.config, 'use_efficient_tcn_v3', False)
        self.tcn_v3_filters = getattr(self.config, 'tcn_v3_filters', 64)
        self.tcn_v3_dilations = getattr(self.config, 'tcn_v3_dilations', [1, 2, 4, 8])
        
        # 🚀 SISTEMA DE AJUSTE DINÁMICO DE PERCENTILES
        self.distribution_tolerance = getattr(self.config, 'distribution_tolerance', 5.0)  # ±5% tolerancia
        self.max_adjustment_iterations = getattr(self.config, 'max_adjustment_iterations', 3)  # Máximo 3 iteraciones
        self.adjustment_convergence_threshold = getattr(self.config, 'adjustment_convergence_threshold', 2.0)  # ±2% convergencia

    def calculate_adaptive_thresholds(self, df: pd.DataFrame, symbol: str) -> dict:
        """
        ⚖️ Calcular thresholds adaptativos AGRESIVOS para maximizar señales de trading
        
        ✅ VERSIÓN AGRESIVA PARA GENERAR MÁS SEÑALES:
        - Factor ATR agresivo 1.5x-2.5x (optimizado para cantidad de señales)
        - Umbrales mínimos agresivos: Weak 0.25%, Strong 0.40%
        - Enfoque en generar suficientes señales BUY/SELL
        - Ajuste dinámico basado en volatilidad del mercado
        """
        if not self.use_adaptive_thresholds:
            return self.fixed_thresholds[symbol]
        
        try:
            # ✅ VALIDACIÓN CRÍTICA: Verificar que los datos son válidos
            if df.empty or len(df) < 14:
                print(f"⚠️ Datos insuficientes para {symbol}: {len(df)} registros")
                return self.fixed_thresholds[symbol]

            # Calcular ATR para volatilidad adaptativa
            high_prices = df['high'].values.astype(float)
            low_prices = df['low'].values.astype(float)
            close_prices = df['close'].values.astype(float)
            
            # ✅ VALIDACIÓN CRÍTICA: Verificar que los precios son válidos
            if np.any(np.isnan(close_prices)) or np.any(close_prices <= 0):
                print(f"⚠️ Precios inválidos detectados para {symbol}")
                print(f"   📊 Precios <= 0: {np.sum(close_prices <= 0)}")
                print(f"   📊 Precios NaN: {np.sum(np.isnan(close_prices))}")
                return self.fixed_thresholds[symbol]

            # ATR de 14 períodos
            atr_14 = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
            
            # ✅ VALIDACIÓN CRÍTICA: Verificar que ATR es válido
            if np.all(np.isnan(atr_14)) or len(atr_14) == 0:
                print(f"⚠️ ATR inválido para {symbol}")
                return self.fixed_thresholds[symbol]

            # ⚖️ BALANCEADO: Promedio de ATR intermedio (últimas 30 velas)
            recent_atr = atr_14[-30:] if len(atr_14) > 30 else atr_14
            recent_prices = close_prices[-30:] if len(close_prices) > 30 else close_prices
            
            # ✅ VALIDACIÓN CRÍTICA: Filtrar valores NaN del ATR
            valid_atr = recent_atr[~np.isnan(recent_atr)]
            if len(valid_atr) == 0:
                print(f"⚠️ No hay valores ATR válidos para {symbol}")
                return self.fixed_thresholds[symbol]
            
            avg_atr = np.mean(valid_atr)
            avg_price = np.mean(recent_prices)

            # ✅ CORRECCIÓN CRÍTICA: Validación robusta para división por cero
            if avg_price <= 0 or np.isnan(avg_price) or np.isnan(avg_atr):
                print(f"⚠️ Valores inválidos para {symbol}:")
                print(f"   📊 avg_price: {avg_price}")
                print(f"   📊 avg_atr: {avg_atr}")
                print(f"   🔄 Usando thresholds fijos como fallback")
                return self.fixed_thresholds[symbol]

            # ✅ CORRECCIÓN CRÍTICA: División segura
            atr_percent = avg_atr / avg_price

            # ⚖️ VALIDACIÓN BALANCEADA: Rango razonable para crypto
            if atr_percent <= 0 or atr_percent > 0.12:  # Máximo 12% (más conservador)
                print(f"⚠️ ATR percent inválido para {symbol}: {atr_percent:.4f}")
                print(f"   📊 avg_atr: {avg_atr:.6f}")
                print(f"   📊 avg_price: {avg_price:.6f}")
                print(f"   🔄 Usando thresholds fijos como fallback")
                return self.fixed_thresholds[symbol]

            # ⚖️ ENFOQUE AGRESIVO: Factor optimizado para generar más señales de trading
            volatility_factor = self.calculate_volatility_adjustment(atr_percent)
            base_threshold = max(atr_percent * volatility_factor, 0.0025)  # Mínimo 0.25%

            # ⚖️ LÍMITES AGRESIVOS: Optimizados para generar más señales de trading
            min_weak = 0.0025    # Mínimo 0.25% (más agresivo)
            min_strong = 0.0040  # Mínimo 0.40% (más agresivo)
            
            # 🎯 MULTIPLICADORES AGRESIVOS por timeframe para generar más señales
            timeframe_multipliers = {
                '1m': 1.5,   # 50% más activo en 1m
                '3m': 1.4,   # 40% más activo en 3m
                '5m': 1.3,   # 30% más activo en 5m
                '15m': 1.2,  # 20% más activo en 15m
                '1h': 1.1    # 10% más activo en 1h
            }
            
            # Detectar timeframe del modelo
            timeframe_multiplier = timeframe_multipliers.get(self.timeframe, 1.1)  # Default 1.1 para 5m
            
            # ⚖️ APLICAR MULTIPLICADOR BALANCEADO
            min_weak *= timeframe_multiplier
            min_strong *= timeframe_multiplier
            base_threshold *= timeframe_multiplier

            # 🎯 THRESHOLDS RENTABLES: Factores agresivos para crypto rentable
            adaptive_thresholds = {
                'strong_sell': -max(base_threshold * 2.5, min_strong),  # 2.5x factor (restaurado)
                'weak_sell': -max(base_threshold * 1.5, min_weak),     # 1.5x factor (era 1.1x)
                'weak_buy': max(base_threshold * 1.5, min_weak),       # 1.5x factor (era 1.1x)
                'strong_buy': max(base_threshold * 2.5, min_strong)    # 2.5x factor (restaurado)
            }

            # 🎯 VALIDACIÓN ADAPTADA A CRYPTO
            max_reasonable = 0.08  # Máximo 8% (era 3.5%) - Crypto es volátil
            if (abs(adaptive_thresholds['strong_buy']) > max_reasonable or 
                abs(adaptive_thresholds['strong_sell']) > max_reasonable):
                print(f"⚠️ Thresholds demasiado extremos para {symbol}:")
                print(f"   📊 strong_buy: {adaptive_thresholds['strong_buy']:.4f}")
                print(f"   📊 strong_sell: {adaptive_thresholds['strong_sell']:.4f}")
                print(f"   🔄 Usando thresholds fijos como fallback")
                return self.fixed_thresholds[symbol]

            print(f"🎯 {symbol}: ATR RENTABLE {atr_percent:.4f} ({atr_percent*100:.2f}%) | TF: {self.timeframe}")
            print(f"   💰 Thresholds rentables (factor {timeframe_multiplier:.1f}x, volatility {volatility_factor:.1f}x):")
            print(f"      Strong Buy: {adaptive_thresholds['strong_buy']:.4f} ({adaptive_thresholds['strong_buy']*100:.2f}%)")
            print(f"      Weak Buy: {adaptive_thresholds['weak_buy']:.4f} ({adaptive_thresholds['weak_buy']*100:.2f}%)")
            print(f"      Weak Sell: {adaptive_thresholds['weak_sell']:.4f} ({adaptive_thresholds['weak_sell']*100:.2f}%)")
            print(f"      Strong Sell: {adaptive_thresholds['strong_sell']:.4f} ({adaptive_thresholds['strong_sell']*100:.2f}%)")
            
            return adaptive_thresholds
            
        except Exception as e:
            print(f"⚠️ Error calculando thresholds balanceados para {symbol}: {e}")
            print(f"   🔄 Usando thresholds fijos como fallback")
            return self.fixed_thresholds[symbol]

    def calculate_volatility_adjustment(self, atr_percent: float) -> float:
        """⚖️ Ajustar factor ATR basado en volatilidad actual del mercado"""
        
        # ⚖️ AJUSTE AGRESIVO: Factor más alto para generar más señales
        if atr_percent > 0.06:      # Alta volatilidad >6%
            return 1.8  # Factor agresivo (era 1.5)
        elif atr_percent > 0.03:    # Volatilidad media 3-6%
            return 2.0  # Factor muy agresivo (era 1.7)
        else:                       # Baja volatilidad <3%
            return 2.2  # Factor extremadamente agresivo (era 1.9)
            
        # Resultado: Factor entre 1.8x y 2.2x (optimizado para generar más señales)

    def calculate_adjusted_percentiles(self, distribution_analysis: dict) -> tuple:
        """
        🚀 CALCULAR PERCENTILES AJUSTADOS PARA DISTRIBUCIÓN EQUILIBRADA
        
        ✅ LÓGICA CORREGIDA (2024):
        - NO usar distribución actual para calcular nuevos percentiles
        - Usar targets fijos y ajustar gradualmente
        - Evitar bucles sin convergencia
        
        Args:
            distribution_analysis: Análisis de distribución actual
            
        Returns:
            Tuple de (percentil_buy_ajustado, percentil_sell_ajustado)
        """
        actual_buy_pct = distribution_analysis['buy_percentage']
        actual_sell_pct = distribution_analysis['sell_percentage']
        
        # ✅ LÓGICA CORREGIDA: Usar targets fijos como base
        target_buy_pct = 15.0  # Target fijo: 15% BUY
        target_sell_pct = 15.0  # Target fijo: 15% SELL
        
        # ✅ CÁLCULO CORRECTO: Ajustar percentiles para acercar distribución al target
        # Si tenemos más BUY del target, AUMENTAR el percentil BUY (hacer más restrictivo)
        if actual_buy_pct > target_buy_pct:
            # Aumentar percentil BUY para reducir cantidad de BUY
            adjustment_factor = (actual_buy_pct / target_buy_pct) - 1.0
            adjusted_buy_percentile = min(99.0, self.buy_percentile + (adjustment_factor * 5.0))
        else:
            # Si tenemos menos BUY del target, REDUCIR el percentil BUY (hacer menos restrictivo)
            adjustment_factor = 1.0 - (actual_buy_pct / target_buy_pct)
            adjusted_buy_percentile = max(70.0, self.buy_percentile - (adjustment_factor * 5.0))
        
        # Si tenemos más SELL del target, REDUCIR el percentil SELL (hacer más restrictivo)
        if actual_sell_pct > target_sell_pct:
            # Reducir percentil SELL para reducir cantidad de SELL
            adjustment_factor = (actual_sell_pct / target_sell_pct) - 1.0
            adjusted_sell_percentile = max(1.0, self.sell_percentile - (adjustment_factor * 5.0))
        else:
            # Si tenemos menos SELL del target, AUMENTAR el percentil SELL (hacer menos restrictivo)
            adjustment_factor = 1.0 - (actual_sell_pct / target_sell_pct)
            adjusted_sell_percentile = min(30.0, self.sell_percentile + (adjustment_factor * 5.0))
        
        # ✅ VALIDACIÓN: Asegurar que los percentiles estén en rangos válidos
        adjusted_buy_percentile = np.clip(adjusted_buy_percentile, 70.0, 99.0)
        adjusted_sell_percentile = np.clip(adjusted_sell_percentile, 1.0, 30.0)
        
        print(f"🔄 Percentiles ajustados calculados:")
        print(f"   📊 BUY: {adjusted_buy_percentile:.1f}% (antes: {self.buy_percentile:.1f}%)")
        print(f"   📊 SELL: {adjusted_sell_percentile:.1f}% (antes: {self.sell_percentile:.1f}%)")
        print(f"   📊 Distribución actual: BUY {actual_buy_pct:.1f}%, SELL {actual_sell_pct:.1f}%")
        print(f"   🎯 Target: BUY {target_buy_pct:.1f}%, SELL {target_sell_pct:.1f}%")
        
        return adjusted_buy_percentile, adjusted_sell_percentile

    def analyze_label_distribution(self, labels: np.ndarray) -> dict:
        """
        🔍 ANALIZAR DISTRIBUCIÓN DE ETIQUETAS GENERADAS
        
        Returns:
            Diccionario con estadísticas de distribución y flags de ajuste
        """
        unique, counts = np.unique(labels, return_counts=True)
        label_dist = dict(zip(unique, counts))
        
        total_labels = len(labels)
        class_names = ['SELL', 'HOLD', 'BUY']
        
        # Calcular porcentajes
        buy_count = label_dist.get(2, 0)
        sell_count = label_dist.get(0, 0)
        hold_count = label_dist.get(1, 0)
        
        actual_buy_pct = buy_count / total_labels * 100 if total_labels > 0 else 0
        actual_sell_pct = sell_count / total_labels * 100 if total_labels > 0 else 0
        actual_hold_pct = hold_count / total_labels * 100 if total_labels > 0 else 0
        
        # Calcular desviaciones del target
        target_buy_pct = 15.0
        target_sell_pct = 15.0
        buy_deviation = abs(actual_buy_pct - target_buy_pct)
        sell_deviation = abs(actual_sell_pct - target_sell_pct)
        
        distribution_analysis = {
            'total_labels': total_labels,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'hold_count': hold_count,
            'buy_percentage': actual_buy_pct,
            'sell_percentage': actual_sell_pct,
            'hold_percentage': actual_hold_pct,
            'buy_deviation': buy_deviation,
            'sell_deviation': sell_deviation,
            'max_deviation': max(buy_deviation, sell_deviation),
            'needs_adjustment': max(buy_deviation, sell_deviation) > self.distribution_tolerance,
            'class_names': class_names,
            'label_distribution': label_dist,
            'target_buy_percentage': target_buy_pct,
            'target_sell_percentage': target_sell_pct
        }
        
        # Logging de análisis
        print(f"📊 Distribución de etiquetas:")
        for i, name in enumerate(class_names):
            count = label_dist.get(i, 0)
            pct = (count / total_labels * 100) if total_labels > 0 else 0
            print(f"   - {name}: {count} ({pct:.1f}%)")
        
        print(f"📊 Target: BUY {target_buy_pct}%, SELL {target_sell_pct}%")
        print(f"📊 Desviaciones: BUY ±{buy_deviation:.1f}%, SELL ±{sell_deviation:.1f}%")
        
        if distribution_analysis['needs_adjustment']:
            print(f"⚠️ Distribución desviada del target. Se requiere ajuste automático.")
        
        return distribution_analysis

    def validate_percentile_configuration(self) -> bool:
        """
        ✅ VALIDAR CONFIGURACIÓN DE PERCENTILES PARA EVITAR ERRORES
        
        Returns:
            True si la configuración es válida, False en caso contrario
        """
        print(f"🔍 Validando configuración de percentiles...")
        
        # ✅ VALIDACIONES CRÍTICAS
        if self.buy_percentile <= self.sell_percentile:
            print(f"❌ ERROR: buy_percentile ({self.buy_percentile}) debe ser mayor que sell_percentile ({self.sell_percentile})")
            return False
        
        if self.buy_percentile >= 100 or self.sell_percentile <= 0:
            print(f"❌ ERROR: Percentiles fuera de rango válido [1, 99]")
            print(f"   📊 buy_percentile: {self.buy_percentile}")
            print(f"   📊 sell_percentile: {self.sell_percentile}")
            return False
        
        # ✅ VALIDAR DISTRIBUCIÓN OBJETIVO
        target_buy_pct = 100 - self.buy_percentile
        target_sell_pct = self.sell_percentile
        target_hold_pct = self.buy_percentile - self.sell_percentile
        
        if target_buy_pct < 5 or target_sell_pct < 5:
            print(f"⚠️ ADVERTENCIA: Distribución muy desbalanceada")
            print(f"   📊 BUY: {target_buy_pct}% (mínimo recomendado: 5%)")
            print(f"   📊 SELL: {target_sell_pct}% (mínimo recomendado: 5%)")
            print(f"   📊 HOLD: {target_hold_pct}%")
        
        # ✅ VALIDAR VENTANA ROLLING
        if self.percentile_window < 50:
            print(f"⚠️ ADVERTENCIA: Ventana rolling muy pequeña ({self.percentile_window})")
            print(f"   📊 Mínimo recomendado: 100 para estabilidad")
        elif self.percentile_window > 5000:
            print(f"⚠️ ADVERTENCIA: Ventana rolling muy grande ({self.percentile_window})")
            print(f"   📊 Máximo recomendado: 2000 para eficiencia")
        
        # ✅ VALIDAR MUESTRAS MÍNIMAS
        if self.min_samples_for_percentiles < 100:
            print(f"⚠️ ADVERTENCIA: Mínimo de muestras muy bajo ({self.min_samples_for_percentiles})")
            print(f"   📊 Mínimo recomendado: 200 para confiabilidad")
        
        print(f"✅ Configuración de percentiles validada")
        print(f"   📊 BUY: {target_buy_pct}% (percentil {self.buy_percentile}+)")
        print(f"   📊 SELL: {target_sell_pct}% (percentil 0-{self.sell_percentile})")
        print(f"   📊 HOLD: {target_hold_pct}% (percentil {self.sell_percentile}-{self.buy_percentile})")
        print(f"   📊 Ventana rolling: {self.percentile_window}")
        print(f"   📊 Muestras mínimas: {self.min_samples_for_percentiles}")
        
        return True

    def optimize_thresholds_for_symbol(self, df: pd.DataFrame, symbol: str) -> dict:
        """🎯 Optimización automática de thresholds para maximizar cantidad de señales de trading"""
        
        print(f"🎯 Optimizando thresholds para {symbol}...")
        
        # Candidatos de thresholds RENTABLES (todos cubren fees + ganancia)
        threshold_candidates = [
            # Conservador (pero rentable)
            {'factor': 2.0, 'min_weak': 0.008, 'min_strong': 0.015, 
             'weak_factor': 2.0, 'strong_factor': 3.0, 'name': 'Conservador'},
            # Balanceado rentable
            {'factor': 2.5, 'min_weak': 0.006, 'min_strong': 0.012,
             'weak_factor': 2.5, 'strong_factor': 3.5, 'name': 'Balanceado'},
            # Agresivo rentable
            {'factor': 3.0, 'min_weak': 0.005, 'min_strong': 0.010,
             'weak_factor': 3.0, 'strong_factor': 4.0, 'name': 'Activo'},
            # Ultra agresivo para crypto volátil
            {'factor': 3.5, 'min_weak': 0.004, 'min_strong': 0.008,
             'weak_factor': 3.5, 'strong_factor': 4.5, 'name': 'Ultra-Activo'},
        ]
        
        best_score = -1
        best_thresholds = None
        best_config = None
        
        # Usar últimos 500 registros para evaluación rápida
        eval_df = df.tail(min(500, len(df))) if len(df) > 100 else df
        
        for config in threshold_candidates:
            try:
                # Generar thresholds de prueba
                test_thresholds = self.generate_test_thresholds(eval_df, symbol, config)
                
                # Evaluación rápida de calidad
                score = self.evaluate_threshold_quality(eval_df, test_thresholds, symbol)
                
                print(f"   📊 {config['name']}: Score {score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_thresholds = test_thresholds
                    best_config = config
                    
            except Exception as e:
                print(f"   ⚠️ Error evaluando {config['name']}: {e}")
                continue
        
        if best_thresholds:
            print(f"   ✅ Mejor configuración: {best_config['name']} (score: {best_score:.3f})")
            return best_thresholds
        else:
            print(f"   ⚠️ Optimización falló, usando thresholds fijos")
            return self.fixed_thresholds.get(symbol, self.get_default_thresholds(symbol))

    def generate_test_thresholds(self, df: pd.DataFrame, symbol: str, config: dict) -> dict:
        """🔧 Generar thresholds de prueba basados en configuración"""
        
        # Calcular ATR básico
        high_prices = df['high'].values.astype(float)
        low_prices = df['low'].values.astype(float)
        close_prices = df['close'].values.astype(float)
        
        atr_14 = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
        valid_atr = atr_14[~np.isnan(atr_14)]
        
        if len(valid_atr) == 0:
            return self.fixed_thresholds.get(symbol, self.get_default_thresholds(symbol))
        
        avg_atr = np.mean(valid_atr[-20:])  # Últimos 20 valores
        avg_price = np.mean(close_prices[-20:])
        
        if avg_price <= 0:
            return self.fixed_thresholds.get(symbol, self.get_default_thresholds(symbol))
        
        atr_percent = avg_atr / avg_price
        base_threshold = max(atr_percent * config['factor'], config['min_weak'])
        
        # 🎯 USAR LOS FACTORES DE LA CONFIGURACIÓN (NO FIJOS)
        strong_factor = config.get('strong_factor', 2.5)  # Factor para strong (por defecto 2.5x)
        weak_factor = config.get('weak_factor', 1.5)      # Factor para weak (por defecto 1.5x)
        
        return {
            'strong_sell': -max(base_threshold * strong_factor, config['min_strong']),
            'weak_sell': -max(base_threshold * weak_factor, config['min_weak']),
            'weak_buy': max(base_threshold * weak_factor, config['min_weak']),
            'strong_buy': max(base_threshold * strong_factor, config['min_strong'])
        }

    def evaluate_threshold_quality(self, df: pd.DataFrame, thresholds: dict, symbol: str) -> float:
        """📊 Evaluación rápida de calidad de thresholds basada en retornos futuros"""
        
        try:
            close_prices = df['close'].values.astype(float)
            
            if len(close_prices) < 50:
                return 0.0
            
            # Simular señales con estos thresholds
            returns_1h = []  # Retornos 1 hora después de señal
            
            for i in range(len(close_prices) - 12):  # 12 períodos = ~1 hora en 5m
                current_price = close_prices[i]
                future_price = close_prices[i + 12] if i + 12 < len(close_prices) else close_prices[-1]
                
                # Calcular retorno
                price_change = (future_price - current_price) / current_price
                
                # Determinar señal con thresholds de prueba
                if price_change >= thresholds['strong_buy']:
                    signal = 'STRONG_BUY'
                elif price_change >= thresholds['weak_buy']:
                    signal = 'WEAK_BUY'
                elif price_change <= thresholds['strong_sell']:
                    signal = 'STRONG_SELL'
                elif price_change <= thresholds['weak_sell']:
                    signal = 'WEAK_SELL'
                else:
                    signal = 'HOLD'
                
                # Solo evaluar señales de trading (no HOLD)
                if signal in ['STRONG_BUY', 'WEAK_BUY', 'STRONG_SELL', 'WEAK_SELL']:
                    # Simular el trade correcto
                    if signal in ['STRONG_BUY', 'WEAK_BUY']:
                        trade_return = price_change  # Long trade
                    else:
                        trade_return = -price_change  # Short trade (pero solo hacemos long)
                        trade_return = 0  # En spot solo long, así que short = 0
                    
                    if signal in ['STRONG_BUY', 'WEAK_BUY']:  # Solo evaluar longs
                        returns_1h.append(trade_return)
            
            if len(returns_1h) < 5:  # Muy pocas señales
                return 0.0
            
            # Calcular métricas de calidad
            win_rate = len([r for r in returns_1h if r > 0]) / len(returns_1h)
            avg_return = np.mean(returns_1h)
            
            # 💰 BONIFICAR señales de trading activas (no penalizar)
            signal_frequency = len(returns_1h) / len(close_prices)
            # Bonificar entre 5-25% de señales, penalizar solo si es extremo (<2% o >50%)
            if signal_frequency < 0.02:  # Muy pocas señales
                frequency_bonus = 0.5
            elif signal_frequency < 0.05:  # Pocas señales
                frequency_bonus = 0.8  
            elif signal_frequency <= 0.25:  # Rango óptimo de trading
                frequency_bonus = 1.0 + (signal_frequency * 2)  # Bonificar más señales
            elif signal_frequency <= 0.40:  # Activo pero aceptable
                frequency_bonus = 1.2
            else:  # Overtrading extremo (>40%)
                frequency_bonus = 0.8
            
            # Score combinado: win rate * avg return * frequency bonus
            quality_score = win_rate * max(avg_return, 0) * frequency_bonus * 100
            
            return quality_score
            
        except Exception as e:
            return 0.0

    def get_default_thresholds(self, symbol: str) -> dict:
        """💰 Obtener thresholds por defecto RENTABLES para cualquier símbolo"""

        # 🎯 THRESHOLDS POR DEFECTO AGRESIVOS (CUBREN FEES + GANANCIA)
        default_thresholds = {
            'strong_sell': -0.012,  # -1.2% (era -0.3%)
            'weak_sell': -0.006,    # -0.6% (era -0.15%)  
            'weak_buy': 0.006,      # 0.6% (era 0.15%)
            'strong_buy': 0.012     # 1.2% (era 0.3%)
        }

        # Si el símbolo tiene thresholds específicos, usarlos
        if symbol in self.fixed_thresholds:
            return self.fixed_thresholds[symbol]

        print(f"⚠️ Usando thresholds por defecto para {symbol}")
        return default_thresholds

    def validate_thresholds(self, thresholds: dict, symbol: str) -> bool:
        """🎯 Validar que los thresholds son razonables"""

        try:
            # Verificar que todos los campos están presentes
            required_fields = ['strong_sell', 'weak_sell', 'weak_buy', 'strong_buy']
            for field in required_fields:
                if field not in thresholds:
                    print(f"❌ Campo faltante en thresholds: {field}")
                    return False

            # Verificar que los valores son números válidos
            for field, value in thresholds.items():
                if not isinstance(value, (int, float)) or np.isnan(value):
                    print(f"❌ Valor inválido en {field}: {value}")
                    return False

            # Verificar orden lógico: strong_sell < weak_sell < weak_buy < strong_buy
            if not (thresholds['strong_sell'] < thresholds['weak_sell'] <
                   thresholds['weak_buy'] < thresholds['strong_buy']):
                print(f"❌ Orden lógico incorrecto en thresholds para {symbol}")
                return False

            # Verificar que los valores no son extremos
            max_threshold = 0.1  # Máximo 10%
            for field, value in thresholds.items():
                if abs(value) > max_threshold:
                    print(f"❌ Threshold demasiado extremo en {field}: {value:.4f}")
                    return False

            return True

        except Exception as e:
            print(f"❌ Error validando thresholds para {symbol}: {e}")
            return False

    def create_balanced_labels(self, df: pd.DataFrame, features: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """🎯 Crear etiquetas - SISTEMA INTELIGENTE CON PERCENTILES DINÁMICOS"""

        # 🚀 PRIORIDAD 1: SISTEMA DE PERCENTILES DINÁMICOS (PROFESIONAL)
        if hasattr(self, 'use_dynamic_percentiles') and self.use_dynamic_percentiles:
            print(f"🚀 ACTIVANDO SISTEMA DE PERCENTILES DINÁMICOS para {symbol}...")
            return self.create_dynamic_percentile_labels(df, features, symbol)

        # 🔄 FALLBACK: Sistema tradicional de thresholds
        print(f"🎯 Usando sistema tradicional de thresholds para {symbol}...")

        close_prices = df['close'].values
        
        # 🎯 RESPETAR CONFIGURACIÓN DE THRESHOLDS
        try:
            if self.use_adaptive_thresholds and len(df) > 200:
                # Optimización automática solo si está habilitada
                print(f"🎯 Iniciando optimización automática de thresholds para {symbol}")
                thresholds = self.optimize_thresholds_for_symbol(df, symbol)
            elif self.use_adaptive_thresholds:
                # Thresholds adaptativos si hay pocos datos
                print(f"⚖️ Datos insuficientes para optimización, usando thresholds adaptativos")
                thresholds = self.calculate_adaptive_thresholds(df, symbol)
            else:
                # 💰 FORZAR USO DE THRESHOLDS FIJOS RENTABLES
                print(f"💰 Usando thresholds fijos rentables (configuración forzada)")
                thresholds = self.fixed_thresholds.get(symbol, self.get_default_thresholds(symbol))
            
            # 🔍 DEBUG: Mostrar thresholds exactos que se van a usar
            print(f"   📊 THRESHOLDS FINALES PARA {symbol}:")
            print(f"      Strong Sell: {thresholds['strong_sell']:.4f} ({thresholds['strong_sell']*100:.2f}%)")
            print(f"      Weak Sell:   {thresholds['weak_sell']:.4f} ({thresholds['weak_sell']*100:.2f}%)")
            print(f"      Weak Buy:    {thresholds['weak_buy']:.4f} ({thresholds['weak_buy']*100:.2f}%)")
            print(f"      Strong Buy:  {thresholds['strong_buy']:.4f} ({thresholds['strong_buy']*100:.2f}%)")
            
            # ✅ VALIDACIÓN CRÍTICA: Verificar que los thresholds son válidos
            if not self.validate_thresholds(thresholds, symbol):
                print(f"⚠️ Thresholds inválidos para {symbol}, usando fijos balanceados")
                thresholds = self.fixed_thresholds.get(symbol, self.get_default_thresholds(symbol))
                print(f"   📊 THRESHOLDS FALLBACK:")
                print(f"      Strong Sell: {thresholds['strong_sell']:.4f} ({thresholds['strong_sell']*100:.2f}%)")
                print(f"      Weak Sell:   {thresholds['weak_sell']:.4f} ({thresholds['weak_sell']*100:.2f}%)")
                print(f"      Weak Buy:    {thresholds['weak_buy']:.4f} ({thresholds['weak_buy']*100:.2f}%)")
                print(f"      Strong Buy:  {thresholds['strong_buy']:.4f} ({thresholds['strong_buy']*100:.2f}%)")

        except Exception as e:
            print(f"⚠️ Error obteniendo thresholds para {symbol}: {e}")
            print(f"   🔄 Usando thresholds fijos balanceados")
            thresholds = self.fixed_thresholds.get(symbol, self.get_default_thresholds(symbol))

        labels = []

        # ✅ VALIDACIÓN CRÍTICA: Verificar que tenemos suficientes datos
        if len(close_prices) <= self.prediction_horizon:
            print(f"❌ ERROR: Datos insuficientes para {symbol}")
            print(f"   📊 Datos disponibles: {len(close_prices)}")
            print(f"   📊 Horizonte requerido: {self.prediction_horizon}")
            return pd.DataFrame()  # Retornar DataFrame vacío

        # 🔄 RESTO DE LA LÓGICA: Con validaciones adicionales
        for i in range(len(close_prices) - self.prediction_horizon):
            current_price = close_prices[i]
            future_price = close_prices[i + self.prediction_horizon]

            # 🎯 VALIDACIÓN OPTIMIZADA: Menos fallbacks a HOLD
            if current_price <= 0 or future_price <= 0:
                # En lugar de HOLD, usar precio anterior válido si es posible
                if i > 0 and close_prices[i-1] > 0:
                    current_price = close_prices[i-1]
                    print(f"⚠️ Precio corregido en posición {i}: usando precio anterior {current_price}")
                else:
                    print(f"⚠️ Precios inválidos sin corrección posible en posición {i}")
                    label = 1  # HOLD solo como último recurso
                    labels.append(label)
                    continue

            # Calcular retorno futuro con manejo robusto
            try:
                future_return = (future_price - current_price) / current_price
                
                # Si el retorno es muy extremo pero válido, no usar HOLD automáticamente
                if np.isnan(future_return) or np.isinf(future_return):
                    # Intentar usando precio promedio de ventana móvil
                    if i >= 3:
                        avg_price = np.mean(close_prices[max(0, i-3):i+1])
                        if avg_price > 0:
                            future_return = (future_price - avg_price) / avg_price
                            if not (np.isnan(future_return) or np.isinf(future_return)):
                                print(f"⚠️ Retorno corregido usando promedio móvil en posición {i}")
                            else:
                                label = 1  # HOLD como último recurso
                                labels.append(label)
                                continue
                        else:
                            label = 1  # HOLD como último recurso
                            labels.append(label)
                            continue
                    else:
                        label = 1  # HOLD para datos insuficientes
                        labels.append(label)
                        continue
                        
            except ZeroDivisionError:
                # Intentar corrección usando datos históricos
                if i >= 5:
                    historical_avg = np.mean(close_prices[max(0, i-5):i])
                    if historical_avg > 0:
                        future_return = (future_price - historical_avg) / historical_avg
                        print(f"⚠️ Retorno calculado con promedio histórico en posición {i}")
                    else:
                        label = 1  # HOLD como último recurso
                        labels.append(label)
                        continue
                else:
                    label = 1  # HOLD para datos insuficientes
                    labels.append(label)
                    continue

            # 🎯 LÓGICA RENTABLE (CON INDICADORES TÉCNICOS CORREGIDOS)
            if future_return <= thresholds['strong_sell']:
                label = 0  # SELL FUERTE
            elif future_return <= thresholds['weak_sell']:
                # Zona débil negativa: confirmar con indicadores
                try:
                    current_rsi = features['rsi_14'].iloc[i] if i < len(features) else 50
                    current_macd = features['macd_histogram'].iloc[i] if i < len(features) else 0
                except:
                    current_rsi = 50
                    current_macd = 0

                # CORREGIDO: RSI < 30 = sobreventa (SELL), MACD < 0 = tendencia bajista
                if current_rsi < 35 or current_macd < -0.1:
                    label = 0  # SELL (confirmación bajista)
                else:
                    label = 1  # HOLD (señal mixta)
            elif future_return >= thresholds['strong_buy']:
                label = 2  # BUY FUERTE  
            elif future_return >= thresholds['weak_buy']:
                # Zona débil positiva: confirmar con indicadores
                try:
                    current_rsi = features['rsi_14'].iloc[i] if i < len(features) else 50
                    current_macd = features['macd_histogram'].iloc[i] if i < len(features) else 0
                except:
                    current_rsi = 50
                    current_macd = 0

                # CORREGIDO: RSI > 70 = sobrecompra pero aún alcista, MACD > 0 = tendencia alcista
                if current_rsi > 65 or current_macd > 0.1:
                    label = 2  # BUY (confirmación alcista)
                else:
                    label = 1  # HOLD (señal mixta)
            else:
                # 💰 ZONA NEUTRAL: Momentum agresivo para crypto
                if i >= 10:  # Usar ventana más larga (10 períodos)
                    try:
                        # Momentum de corto plazo (5 períodos) y mediano plazo (10 períodos)
                        short_momentum = (close_prices[i] - close_prices[i-5]) / close_prices[i-5]
                        medium_momentum = (close_prices[i] - close_prices[i-10]) / close_prices[i-10]

                        # ✅ VALIDACIÓN: Verificar que el momentum es válido
                        if (not np.isnan(short_momentum) and not np.isinf(short_momentum) and
                            not np.isnan(medium_momentum) and not np.isinf(medium_momentum)):
                            
                            # 🎯 Umbrales agresivos para crypto (era 0.25%)
                            strong_momentum_threshold = 0.015  # 1.5% para señales fuertes
                            weak_momentum_threshold = 0.008   # 0.8% para señales débiles
                            
                            # Combinar momentum corto y mediano plazo
                            if (short_momentum > strong_momentum_threshold or 
                                (short_momentum > weak_momentum_threshold and medium_momentum > 0.005)):
                                label = 2  # BUY (momentum alcista fuerte)
                            elif (short_momentum < -strong_momentum_threshold or 
                                  (short_momentum < -weak_momentum_threshold and medium_momentum < -0.005)):
                                label = 0  # SELL (momentum bajista fuerte)
                            else:
                                label = 1  # HOLD (momentum neutral)
                        else:
                            label = 1  # HOLD como fallback
                    except (ZeroDivisionError, IndexError):
                        label = 1  # HOLD como fallback
                elif i >= 5:
                    # Fallback para datos insuficientes
                    try:
                        short_momentum = (close_prices[i] - close_prices[i-5]) / close_prices[i-5]
                        if not np.isnan(short_momentum) and not np.isinf(short_momentum):
                            if short_momentum > 0.012:  # 1.2%
                                label = 2  # BUY
                            elif short_momentum < -0.012:  # -1.2%
                                label = 0  # SELL
                            else:
                                label = 1  # HOLD
                        else:
                            label = 1  # HOLD
                    except (ZeroDivisionError, IndexError):
                        label = 1  # HOLD
                else:
                    label = 1  # HOLD (datos insuficientes)

            labels.append(label)

        # Agregar labels al DataFrame
        df_labeled = df.iloc[:-self.prediction_horizon].copy()
        df_labeled['label'] = labels

        # ✅ VALIDACIÓN FINAL: Verificar que tenemos suficientes etiquetas
        if len(labels) == 0:
            print(f"❌ ERROR: No se pudieron generar etiquetas para {symbol}")
            return pd.DataFrame()

        # Verificar distribución
        label_counts = pd.Series(labels).value_counts().sort_index()
        total = len(labels)

        print("📊 Distribución de etiquetas:")
        class_names = ['SELL', 'HOLD', 'BUY']
        for i, name in enumerate(class_names):
            count = label_counts.get(i, 0) or 0
            pct = (count / total * 100) if total > 0 and count is not None else 0
            print(f"   - {name}: {count} ({pct:.1f}%)")

        # 💰 VALIDACIÓN RENTABLE: Distribución optimizada para trading activo
        # Para crypto, 1-2% por clase trading es aceptable (no necesitamos 5%)
        min_samples_per_class = max(50, total * 0.01)  # Mínimo 1% por clase o 50 muestras
        for i, name in enumerate(class_names):
            count = label_counts.get(i, 0) or 0
            if count < min_samples_per_class:
                print(f"⚠️ ADVERTENCIA: Pocas muestras de clase {name}: {count} (mínimo {min_samples_per_class})")
                
        # 🎯 VALIDACIÓN ADICIONAL: Advertir si hay demasiados HOLD
        hold_count = label_counts.get(1, 0) or 0
        hold_percentage = (hold_count / total * 100) if total > 0 else 0
        if hold_percentage > 85:
            print(f"⚠️ CRÍTICO: Demasiados HOLD ({hold_percentage:.1f}%) - Modelo será pasivo")
            print(f"💡 Sugerencia: Usar thresholds más agresivos para aumentar señales de trading")

        return df_labeled

    def create_dynamic_percentile_labels(self, df: pd.DataFrame, features: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        🚀 SISTEMA DE ETIQUETADO POR PERCENTILES DINÁMICOS (MÉTODO PROFESIONAL)
        
        ✅ LÓGICA CORREGIDA (2024):
        - buy_percentile = 85 → BUY = percentil 85+ (top 15% de retornos)
        - sell_percentile = 15 → SELL = percentil 0-15 (bottom 15% de retornos)  
        - HOLD = percentil 15-85 (70% restante)
        
        Usado por sistemas de trading algorítmico exitosos como:
        - Renaissance Technologies
        - Two Sigma  
        - AQR Capital
        
        Ventajas:
        - Garantiza distribución balanceada (15% BUY, 15% SELL, 70% HOLD)
        - Se adapta automáticamente a la volatilidad del mercado
        - No requiere ajustar thresholds manualmente por símbolo
        - Siempre rentable porque se basa en distribución real de retornos
        """
        
        print(f"🚀 Creando etiquetas con PERCENTILES DINÁMICOS para {symbol}...")
        print(f"   📊 Target: {100-self.buy_percentile}% BUY, {self.sell_percentile}% SELL, {self.buy_percentile-self.sell_percentile}% HOLD")
        print(f"   🎯 BUY: percentil {self.buy_percentile}+ (top {100-self.buy_percentile}%)")
        print(f"   🎯 SELL: percentil 0-{self.sell_percentile} (bottom {self.sell_percentile}%)")
        
        # ✅ VALIDAR CONFIGURACIÓN ANTES DE PROCEDER
        if not self.validate_percentile_configuration():
            print(f"⚠️ Configuración de percentiles inválida para {symbol}")
            print(f"🔄 Fallback a sistema de thresholds...")
            return self.create_balanced_labels(df, features, symbol)
        
        close_prices = df['close'].values.astype(float)
        
        # ✅ VALIDACIÓN INICIAL CRÍTICA
        if len(close_prices) <= self.prediction_horizon:
            print(f"❌ Insuficientes datos para {symbol}: {len(close_prices)} <= {self.prediction_horizon}")
            return pd.DataFrame()
            
        if len(close_prices) < self.min_samples_for_percentiles:
            print(f"⚠️ Pocos datos para percentiles ({len(close_prices)} < {self.min_samples_for_percentiles})")
            print(f"🔄 Fallback a sistema de thresholds...")
            return self.create_balanced_labels(df, features, symbol)
        
        # 🎯 CALCULAR RETORNOS FUTUROS ROBUSTAMENTE
        print(f"📊 Calculando retornos futuros (horizonte: {self.prediction_horizon})...")
        
        future_returns = []
        valid_indices = []
        
        for i in range(len(close_prices) - self.prediction_horizon):
            current_price = close_prices[i]
            future_price = close_prices[i + self.prediction_horizon]
            
            # ✅ VALIDACIONES ROBUSTAS
            if current_price <= 0 or future_price <= 0:
                continue
                
            try:
                future_return = (future_price - current_price) / current_price
                
                # Verificar que el retorno es válido
                if np.isnan(future_return) or np.isinf(future_return):
                    continue
                    
                # Filtrar retornos extremos (outliers >±50%)
                if abs(future_return) > 0.5:
                    continue
                    
                future_returns.append(future_return)
                valid_indices.append(i)
                
            except (ZeroDivisionError, OverflowError):
                continue
        
        if len(future_returns) < self.min_samples_for_percentiles * 0.8:
            print(f"❌ Muy pocos retornos válidos: {len(future_returns)}")
            print(f"🔄 Fallback a sistema de thresholds...")
            return self.create_balanced_labels(df, features, symbol)
        
        future_returns = np.array(future_returns)
        print(f"✅ Retornos válidos calculados: {len(future_returns)}")
        print(f"   📊 Min: {future_returns.min():.4f} ({future_returns.min()*100:.2f}%)")
        print(f"   📊 Max: {future_returns.max():.4f} ({future_returns.max()*100:.2f}%)")
        print(f"   📊 Std: {future_returns.std():.4f} ({future_returns.std()*100:.2f}%)")
        
        # 🎯 ETIQUETADO POR PERCENTILES DINÁMICOS CON VENTANA ROLLING CORREGIDA
        labels = []
        window_size = min(self.percentile_window, len(future_returns))
        
        print(f"🔄 Procesando etiquetas con ventana rolling de {window_size} samples...")
        
        for i in range(len(future_returns)):
            # ✅ VENTANA ROLLING CORREGIDA: Ventana de tamaño fijo centrada en el punto actual
            # Si estamos cerca de los bordes, ajustar la ventana para mantener tamaño consistente
            if i < window_size // 2:
                # Al inicio: ventana desde 0 hasta window_size
                start_idx = 0
                end_idx = window_size
            elif i >= len(future_returns) - window_size // 2:
                # Al final: ventana desde len - window_size hasta len
                start_idx = len(future_returns) - window_size
                end_idx = len(future_returns)
            else:
                # En el medio: ventana centrada en i
                start_idx = i - window_size // 2
                end_idx = i + window_size // 2
            
            # ✅ VALIDACIÓN ADICIONAL: Asegurar que la ventana tenga el tamaño correcto
            if end_idx - start_idx != window_size:
                # Ajustar para mantener tamaño consistente
                if end_idx - start_idx < window_size:
                    if start_idx > 0:
                        start_idx = max(0, end_idx - window_size)
                    else:
                        end_idx = min(len(future_returns), start_idx + window_size)
            
            window_returns = future_returns[start_idx:end_idx]
            current_return = future_returns[i]
            
            # ✅ VALIDACIÓN: Asegurar que la ventana tenga suficientes datos
            if len(window_returns) < window_size * 0.8:
                # Si la ventana es muy pequeña, usar todos los datos disponibles hasta el momento
                window_returns = future_returns[:i+1]
                if len(window_returns) < 10:  # Mínimo de datos para percentiles
                    label = 1  # HOLD si no hay suficientes datos
                    labels.append(label)
                    continue
            
            # Calcular percentiles dinámicos
            buy_threshold = np.percentile(window_returns, self.buy_percentile)  # 85 = top 15%
            sell_threshold = np.percentile(window_returns, self.sell_percentile)  # 15 = bottom 15%
            
            # ✅ LÓGICA CORREGIDA: BUY = percentil 85+ (top 15%), SELL = percentil 0-15 (bottom 15%)
            if current_return >= buy_threshold:
                label = 2  # BUY - Solo el top 15% de retornos
            elif current_return <= sell_threshold:
                label = 0  # SELL - Solo el bottom 15% de retornos
            else:
                label = 1  # HOLD - El 70% restante (percentiles 15-85)
                
            labels.append(label)
        
        # 🔍 VALIDACIÓN DE DISTRIBUCIÓN RESULTANTE CON AJUSTE DINÁMICO
        labels = np.array(labels)
        
        # ✅ ANALIZAR DISTRIBUCIÓN Y APLICAR AJUSTE DINÁMICO SI ES NECESARIO
        distribution_analysis = self.analyze_label_distribution(labels)
        
        # 🔄 APLICAR AJUSTE DINÁMICO ITERATIVO SI LA DISTRIBUCIÓN SE DESVÍA
        if distribution_analysis['needs_adjustment']:
            print(f"🔄 Aplicando ajuste dinámico de percentiles...")
            
            for iteration in range(self.max_adjustment_iterations):
                print(f"   📊 Iteración {iteration + 1}/{self.max_adjustment_iterations}")
                
                # Calcular percentiles ajustados
                adjusted_buy_percentile, adjusted_sell_percentile = self.calculate_adjusted_percentiles(distribution_analysis)
                
                # Recalcular etiquetas con percentiles ajustados
                labels_adjusted = []
                for i in range(len(future_returns)):
                    # Usar la misma lógica de ventana rolling corregida
                    if i < window_size // 2:
                        start_idx = 0
                        end_idx = window_size
                    elif i >= len(future_returns) - window_size // 2:
                        start_idx = len(future_returns) - window_size
                        end_idx = len(future_returns)
                    else:
                        start_idx = i - window_size // 2
                        end_idx = i + window_size // 2
                    
                    if end_idx - start_idx != window_size:
                        if end_idx - start_idx < window_size:
                            if start_idx > 0:
                                start_idx = max(0, end_idx - window_size)
                            else:
                                end_idx = min(len(future_returns), start_idx + window_size)
                    
                    window_returns = future_returns[start_idx:end_idx]
                    current_return = future_returns[i]
                    
                    if len(window_returns) < window_size * 0.8:
                        window_returns = future_returns[:i+1]
                        if len(window_returns) < 10:
                            label = 1
                            labels_adjusted.append(label)
                            continue
                    
                    # Usar percentiles ajustados
                    buy_threshold = np.percentile(window_returns, adjusted_buy_percentile)
                    sell_threshold = np.percentile(window_returns, adjusted_sell_percentile)
                    
                    if current_return >= buy_threshold:
                        label = 2
                    elif current_return <= sell_threshold:
                        label = 0
                    else:
                        label = 1
                    
                    labels_adjusted.append(label)
                
                # Analizar nueva distribución
                labels = np.array(labels_adjusted)
                distribution_analysis = self.analyze_label_distribution(labels)
                
                # Verificar si ya convergió
                if distribution_analysis['max_deviation'] <= self.adjustment_convergence_threshold:
                    print(f"   ✅ Convergencia alcanzada en iteración {iteration + 1}")
                    print(f"   📊 Desviación final: {distribution_analysis['max_deviation']:.1f}%")
                    break
                else:
                    print(f"   📊 Desviación actual: {distribution_analysis['max_deviation']:.1f}%")
            
            print(f"✅ Ajuste dinámico completado")
        
        # ✅ VERIFICACIÓN FINAL DE LA DISTRIBUCIÓN
        total_labels = len(labels)
        class_names = ['SELL', 'HOLD', 'BUY']
        
        print(f"📊 Distribución final de etiquetas:")
        for i, name in enumerate(class_names):
            count = distribution_analysis['label_distribution'].get(i, 0)
            pct = (count / total_labels * 100) if total_labels > 0 else 0
            print(f"   - {name}: {count} ({pct:.1f}%)")
        
        # ✅ VERIFICACIÓN DE LA LÓGICA CORREGIDA
        expected_buy_pct = 100 - self.buy_percentile  # Si buy_percentile = 85, expected = 15%
        expected_sell_pct = self.sell_percentile       # Si sell_percentile = 15, expected = 15%
        expected_hold_pct = self.buy_percentile - self.sell_percentile  # 85 - 15 = 70%
        
        print(f"🎯 Distribución esperada:")
        print(f"   - BUY: {expected_buy_pct}% (percentil {self.buy_percentile}+)")
        print(f"   - SELL: {expected_sell_pct}% (percentil 0-{self.sell_percentile})")
        print(f"   - HOLD: {expected_hold_pct}% (percentil {self.sell_percentile}-{self.buy_percentile})")
        
        # ✅ CREAR DATAFRAME FINAL
        df_labeled = df.iloc[valid_indices].copy()
        
        if len(df_labeled) != len(labels):
            print(f"⚠️ Ajustando longitudes: df={len(df_labeled)}, labels={len(labels)}")
            min_len = min(len(df_labeled), len(labels))
            df_labeled = df_labeled.iloc[:min_len].copy()
            labels = labels[:min_len]
        
        df_labeled['label'] = labels
        
        # 🎯 VALIDACIÓN FINAL
        if len(df_labeled) == 0:
            print(f"❌ ERROR: No se pudieron generar etiquetas para {symbol}")
            return pd.DataFrame()
            
        print(f"✅ Etiquetas por percentiles dinámicos completadas para {symbol}")
        print(f"   📊 Dataset final: {len(df_labeled)} muestras")
        
        return df_labeled

    def handle_missing_values_intelligently(self, df: pd.DataFrame, method='adaptive') -> pd.DataFrame:
        """
        🧠 Manejo inteligente de valores faltantes

        ✅ MÉTODOS DISPONIBLES:
        - 'adaptive': Método adaptativo basado en el tipo de dato
        - 'interpolate': Interpolación lineal
        - 'median': Mediana de la columna
        - 'forward_backward': Forward fill + backward fill
        """

        print(f"🧠 Aplicando manejo inteligente de valores faltantes (método: {method})...")

        if method == 'adaptive':
            return self._handle_missing_values_adaptive(df)
        elif method == 'interpolate':
            return self._handle_missing_values_interpolate(df)
        elif method == 'median':
            return self._handle_missing_values_median(df)
        elif method == 'forward_backward':
            return self._handle_missing_values_forward_backward(df)
        else:
            print(f"⚠️ Método '{method}' no reconocido, usando 'adaptive'")
            return self._handle_missing_values_adaptive(df)

    def _handle_missing_values_adaptive(self, df: pd.DataFrame) -> pd.DataFrame:
        """🎯 Manejo adaptativo basado en el tipo de dato"""

        df_clean = df.copy()

        # ✅ CLASIFICACIÓN DE COLUMNAS POR TIPO
        price_columns = ['open', 'high', 'low', 'close', 'volume']
        technical_indicators = ['rsi', 'macd', 'bbands', 'stoch', 'cci', 'adx', 'atr']
        momentum_indicators = ['momentum', 'roc', 'williams_r', 'mfi']
        trend_indicators = ['sma', 'ema', 'macd_signal', 'macd_histogram']

        print(f"📊 Analizando {len(df.columns)} columnas...")

        for col in df.columns:
            if col in df_clean.columns and df_clean[col].isna().any():
                nan_count = df_clean[col].isna().sum()
                nan_percent = (nan_count / len(df_clean)) * 100

                print(f"   🔧 {col}: {nan_count} NaN ({nan_percent:.1f}%)")

                # ✅ ESTRATEGIA ADAPTATIVA POR TIPO DE DATO
                if any(price_col in col.lower() for price_col in price_columns):
                    # Para precios: interpolación lineal
                    df_clean[col] = df_clean[col].interpolate(method='linear', limit_direction='both')
                    print(f"      📈 Precio: interpolación lineal")

                elif any(tech in col.lower() for tech in technical_indicators):
                    # Para indicadores técnicos: forward fill + backward fill
                    df_clean[col] = df_clean[col].ffill().bfill()
                    print(f"      📊 Técnico: forward + backward fill")

                elif any(mom in col.lower() for mom in momentum_indicators):
                    # Para momentum: mediana de ventana móvil
                    window_size = min(20, len(df_clean) // 4)
                    df_clean[col] = df_clean[col].fillna(df_clean[col].rolling(window=window_size, min_periods=1).median())
                    print(f"      ⚡ Momentum: mediana móvil (ventana={window_size})")

                elif any(trend in col.lower() for trend in trend_indicators):
                    # Para tendencias: interpolación cúbica
                    df_clean[col] = df_clean[col].interpolate(method='cubic', limit_direction='both')
                    print(f"      📈 Tendencia: interpolación cúbica")

                else:
                    # Para otros: mediana de la columna
                    median_val = df_clean[col].median()
                    if pd.isna(median_val):
                        median_val = 0  # Fallback
                    df_clean[col] = df_clean[col].fillna(median_val)
                    print(f"      📊 Otro: mediana ({median_val:.4f})")

        return df_clean

    def _handle_missing_values_interpolate(self, df: pd.DataFrame) -> pd.DataFrame:
        """📈 Interpolación lineal para todas las columnas"""

        df_clean = df.copy()

        for col in df_clean.columns:
            if df_clean[col].isna().any():
                df_clean[col] = df_clean[col].interpolate(method='linear', limit_direction='both')

        return df_clean

    def _handle_missing_values_median(self, df: pd.DataFrame) -> pd.DataFrame:
        """📊 Mediana de cada columna"""

        df_clean = df.copy()

        for col in df_clean.columns:
            if df_clean[col].isna().any():
                median_val = df_clean[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df_clean[col] = df_clean[col].fillna(median_val)

        return df_clean

    def _handle_missing_values_forward_backward(self, df: pd.DataFrame) -> pd.DataFrame:
        """🔄 Forward fill + backward fill"""

        df_clean = df.copy()

        for col in df_clean.columns:
            if df_clean[col].isna().any():
                df_clean[col] = df_clean[col].ffill().bfill()

        return df_clean

    def diagnose_missing_values(self, df: pd.DataFrame, symbol: str) -> Dict:
        """🔍 Diagnóstico detallado de valores faltantes"""

        print(f"🔍 DIAGNÓSTICO DE VALORES FALTANTES - {symbol}")
        print("=" * 60)

        diagnosis = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns_with_nan': [],
            'nan_summary': {},
            'inf_summary': {},
            'recommendations': []
        }

        # Analizar cada columna
        for col in df.columns:
            nan_count = df[col].isna().sum()
            
            # ✅ CORRECCIÓN: Verificar si la columna es numérica antes de usar np.isinf
            if pd.api.types.is_numeric_dtype(df[col]):
                inf_count = np.isinf(df[col]).sum()
            else:
                inf_count = 0  # Columnas no numéricas no pueden tener infinitos
                
            nan_percent = (nan_count / len(df)) * 100
            inf_percent = (inf_count / len(df)) * 100

            if nan_count > 0 or inf_count > 0:
                diagnosis['columns_with_nan'].append(col)
                diagnosis['nan_summary'][col] = {
                    'count': nan_count,
                    'percent': nan_percent
                }
                diagnosis['inf_summary'][col] = {
                    'count': inf_count,
                    'percent': inf_percent
                }

                print(f"📊 {col}:")
                if nan_count > 0:
                    print(f"   ❌ NaN: {nan_count} ({nan_percent:.1f}%)")
                if inf_count > 0:
                    print(f"   ⚠️  Inf: {inf_count} ({inf_percent:.1f}%)")

                # Generar recomendaciones
                if nan_percent > 50:
                    diagnosis['recommendations'].append(f"⚠️  {col}: >50% NaN - considerar eliminar columna")
                elif nan_percent > 20:
                    diagnosis['recommendations'].append(f"🔧 {col}: 20-50% NaN - usar interpolación")
                elif nan_percent > 5:
                    diagnosis['recommendations'].append(f"📊 {col}: 5-20% NaN - usar forward/backward fill")
                else:
                    diagnosis['recommendations'].append(f"✅ {col}: <5% NaN - usar mediana")

        # Resumen general
        total_nan = df.isna().sum().sum()
        
        # ✅ CORRECCIÓN: Calcular infinitos solo en columnas numéricas
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            total_inf = np.isinf(numeric_df).sum().sum()
        else:
            total_inf = 0

        print(f"\n📊 RESUMEN GENERAL:")
        print(f"   📊 Total NaN: {total_nan}")
        print(f"   📊 Total Inf: {total_inf}")
        print(f"   📊 Columnas con problemas: {len(diagnosis['columns_with_nan'])}")

        if diagnosis['recommendations']:
            print(f"\n💡 RECOMENDACIONES:")
            for rec in diagnosis['recommendations']:
                print(f"   {rec}")

        return diagnosis

    # ✅ MÉTODOS CONFIGURABLES
    async def get_real_market_data(self, symbol: str, days: int = None) -> pd.DataFrame:
        """📊 Obtener datos reales de mercado con cache"""

        # ✅ CACHE: Verificar si ya tenemos datos guardados
        days = days or self.training_days
        cache_file = f"cache/{symbol}_{self.timeframe}_{days}d.pkl"
        os.makedirs("cache", exist_ok=True)

        if os.path.exists(cache_file):
            # Verificar si el cache es reciente (menos de 1 hora)
            cache_time = os.path.getmtime(cache_file)
            current_time = time.time()
            if current_time - cache_time < 3600:  # 1 hora
                print(f"📋 Usando datos cacheados para {symbol} ({self.timeframe})")
                try:
                    import pickle
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    print(f"⚠️ Error leyendo cache: {e}, descargando de nuevo...")

        # Usar configuración para determinar período
        if self.start_date and self.end_date:
            start_time = int(self.start_date.timestamp() * 1000)
            end_time = int(self.end_date.timestamp() * 1000)
            period_desc = f"desde {self.start_date.strftime('%Y-%m-%d')} hasta {self.end_date.strftime('%Y-%m-%d')}"
        else:
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            period_desc = f"{days} días"

        print(f"📊 Obteniendo datos {period_desc} para {symbol} ({self.timeframe})...")

        base_url = "https://api.binance.com"

        async with aiohttp.ClientSession() as session:
            url = f"{base_url}/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': self.timeframe,  # ✅ TIMEFRAME CONFIGURABLE
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

        print(f"✅ Obtenidos {len(df)} registros de {symbol}")

        # ✅ CACHE: Guardar datos descargados
        try:
            import pickle
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)
            print(f"💾 Datos guardados en cache: {cache_file}")
        except Exception as e:
            print(f"⚠️ Error guardando cache: {e}")

        return df

    def prepare_training_data(self, df: pd.DataFrame, features: pd.DataFrame) -> tuple:
        """🔧 Preparar datos para entrenamiento - CON MANEJO INTELIGENTE DE NaN"""
        print("🔧 Preparando datos para entrenamiento...")

        features_aligned = features.iloc[:-self.prediction_horizon]
        
        # ✅ CORRECCIÓN: Permitir más tipos de datos numéricos
        # Incluir float32, float64, int32, int64, y otros tipos numéricos
        feature_columns = []
        for col in features_aligned.columns:
            dtype = features_aligned[col].dtype
            # Verificar si es un tipo numérico
            if pd.api.types.is_numeric_dtype(dtype):
                feature_columns.append(col)
            else:
                print(f"⚠️  Columna no numérica excluida: {col} (tipo: {dtype})")
        
        # ✅ CONVERSIÓN AUTOMÁTICA: Convertir todas las features a float64 para consistencia
        print(f"🔧 Convirtiendo {len(feature_columns)} features a float64 para consistencia...")
        for col in feature_columns:
            try:
                features_aligned[col] = features_aligned[col].astype('float64')
            except Exception as e:
                print(f"❌ Error convirtiendo {col} a float64: {e}")
                # Si falla la conversión, excluir la columna
                feature_columns.remove(col)
                print(f"   🚫 Columna {col} excluida del entrenamiento")

        # ✅ VALIDACIÓN COMPREHENSIVA DE DATOS
        print(f"🔍 Verificando calidad de datos...")
        nan_count = features_aligned[feature_columns].isna().sum().sum()
        inf_count = np.isinf(features_aligned[feature_columns]).sum().sum()

        print(f"📊 Estado inicial:")
        print(f"   📊 Valores NaN: {nan_count}")
        print(f"   📊 Valores Inf: {inf_count}")
        print(f"   📊 Columnas: {len(feature_columns)}")
        print(f"   📊 Filas: {len(features_aligned)}")

        # ✅ NUEVO: MANEJO INTELIGENTE DE VALORES FALTANTES
        if nan_count > 0:
            print(f"🧠 Aplicando manejo inteligente de {nan_count} valores NaN...")

            # Usar manejo adaptativo por defecto
            features_aligned = self.handle_missing_values_intelligently(features_aligned, method='adaptive')

            # Verificar resultado
            final_nan = features_aligned[feature_columns].isna().sum().sum()
            if final_nan > 0:
                print(f"⚠️  Aún quedan {final_nan} valores NaN, aplicando fallback...")
                # Fallback: mediana por columna
                for col in feature_columns:
                    if features_aligned[col].isna().any():
                        median_val = features_aligned[col].median()
                        if pd.isna(median_val):
                            median_val = 0
                        features_aligned[col] = features_aligned[col].fillna(median_val)

        # ✅ MANEJO DE VALORES INFINITOS
        if inf_count > 0:
            print(f"⚠️  Encontrados {inf_count} valores infinitos, reemplazando...")

            # Reemplazar infinitos con valores límite
            for col in feature_columns:
                if np.isinf(features_aligned[col]).any():
                    # Calcular límites basados en percentiles
                    col_data = features_aligned[col].replace([np.inf, -np.inf], np.nan)
                    p99 = col_data.quantile(0.99)
                    p01 = col_data.quantile(0.01)

                    # Reemplazar infinitos con límites
                    features_aligned[col] = features_aligned[col].replace([np.inf, -np.inf], [p99, p01])
                    print(f"      🔧 {col}: límites [{p01:.4f}, {p99:.4f}]")

        # ✅ VERIFICACIÓN FINAL
        final_nan = features_aligned[feature_columns].isna().sum().sum()
        final_inf = np.isinf(features_aligned[feature_columns]).sum().sum()
        print(f"✅ Datos limpiados: NaN={final_nan}, Inf={final_inf}")

        # ✅ VALIDACIÓN CRÍTICA: Verificar que no hay valores inválidos
        if final_nan > 0 or final_inf > 0:
            print(f"❌ ERROR: Aún hay valores inválidos después de la limpieza")
            return None, None, None, None, None

        # ✅ ESCALADO ROBUSTO
        print(f"📊 Aplicando escalado robusto...")
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features_aligned[feature_columns])

        # ✅ VERIFICACIÓN POST-ESCALADO
        if np.isnan(features_scaled).any():
            print("❌ ERROR: RobustScaler produjo valores NaN")
            return None, None, None, None, None

        if np.isinf(features_scaled).any():
            print("❌ ERROR: RobustScaler produjo valores infinitos")
            return None, None, None, None, None

        # ✅ PREPARACIÓN DE SECUENCIAS
        print(f"📊 Preparando secuencias de entrenamiento...")
        X = []
        y = []

        for i in range(self.lookback_window, len(features_scaled)):
            sequence = features_scaled[i-self.lookback_window:i]
            X.append(sequence)
            y.append(df['label'].iloc[i])

        X = np.array(X)
        y = np.array(y)

        # ✅ VALIDACIÓN FINAL DE DATOS
        if len(X) == 0 or len(y) == 0:
            print("❌ ERROR: No se pudieron crear secuencias de entrenamiento")
            return None, None, None, None, None

        print(f"✅ Datos preparados exitosamente:")
        print(f"   📊 X shape: {X.shape}")
        print(f"   📊 y shape: {y.shape}")
        print(f"   📊 Feature columns: {len(feature_columns)}")
        print(f"   📊 Lookback window: {self.lookback_window}")
        print(f"   📊 Prediction horizon: {self.prediction_horizon}")

        # ✅ CÁLCULO DE PESOS DE CLASE
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

        print(f"📊 Pesos de clase calculados:")
        for class_idx, weight in class_weight_dict.items():
            class_name = ['SELL', 'HOLD', 'BUY'][class_idx]
            class_count = np.sum(y == class_idx)
            print(f"   📊 {class_name}: {class_count} muestras, peso={weight:.3f}")

        # 🎯 NUEVA FUNCIONALIDAD: REBALANCEO DE MUESTRAS PARA CLASES MINORITARIAS
        X, y = self.rebalance_training_samples(X, y)
        
        return X, y, scaler, feature_columns, class_weight_dict

    def rebalance_training_samples(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        🎯 Rebalancear muestras de entrenamiento para reducir sesgo hacia HOLD
        
        Estrategia:
        1. Mantener todas las muestras BUY/SELL (minoritarias)
        2. Reducir muestras HOLD para equilibrar mejor
        3. Aplicar oversampling inteligente para BUY/SELL si es necesario
        """
        from sklearn.utils import resample
        
        # Contar clases
        unique_classes, class_counts = np.unique(y, return_counts=True)
        print(f"\n🔄 REBALANCEANDO MUESTRAS DE ENTRENAMIENTO:")
        
        for i, count in enumerate(class_counts):
            class_name = ['SELL', 'HOLD', 'BUY'][unique_classes[i]]
            print(f"   📊 {class_name}: {count} muestras")
        
        # Identificar clase mayoritaria (HOLD = 1)
        hold_class = 1
        hold_count = class_counts[unique_classes == hold_class][0] if hold_class in unique_classes else 0
        
        # Calcular target para cada clase
        minority_classes = [0, 2]  # SELL, BUY
        minority_counts = [class_counts[unique_classes == c][0] for c in minority_classes if c in unique_classes]
        
        if minority_counts:
            max_minority = max(minority_counts)
            # Target: 2.5x la clase minoritaria más grande (en lugar de mantener todo HOLD)
            target_hold = min(max_minority * 3, hold_count)  # Máximo 3x, pero no más de lo que hay
            
            print(f"🎯 Target balanceado:")
            print(f"   📊 HOLD: {target_hold} (reducido de {hold_count})")
            print(f"   📊 SELL/BUY: {max_minority} cada una")
            
            # Separar datos por clase
            X_rebalanced = []
            y_rebalanced = []
            
            for class_idx in unique_classes:
                class_mask = y == class_idx
                X_class = X[class_mask]
                y_class = y[class_mask]
                
                if class_idx == hold_class and len(y_class) > target_hold:
                    # Reducir HOLD por submuestreo
                    X_class_resampled, y_class_resampled = resample(
                        X_class, y_class, 
                        n_samples=target_hold, 
                        random_state=42,
                        stratify=None
                    )
                else:
                    # Mantener clases minoritarias como están
                    X_class_resampled = X_class
                    y_class_resampled = y_class
                
                X_rebalanced.append(X_class_resampled)
                y_rebalanced.append(y_class_resampled)
            
            # Combinar datos rebalanceados
            X_final = np.vstack(X_rebalanced)
            y_final = np.hstack(y_rebalanced)
            
            # Mezclar datos
            shuffle_idx = np.random.permutation(len(X_final))
            X_final = X_final[shuffle_idx]
            y_final = y_final[shuffle_idx]
            
            # Mostrar resultado final
            final_unique, final_counts = np.unique(y_final, return_counts=True)
            print(f"\n✅ REBALANCEO COMPLETADO:")
            for i, count in enumerate(final_counts):
                class_name = ['SELL', 'HOLD', 'BUY'][final_unique[i]]
                original_count = class_counts[unique_classes == final_unique[i]][0] if final_unique[i] in unique_classes else 0
                change = count - original_count
                print(f"   📊 {class_name}: {count} muestras ({change:+d})")
            
            return X_final, y_final
        
        return X, y

    def apply_confidence_filtering(self, y_pred_proba: np.ndarray, 
                                   confidence_threshold: float = None) -> np.ndarray:
        # Usar configuración del archivo .env si no se especifica
        if confidence_threshold is None:
            confidence_threshold = TRAINING_CONFIG.get('MIN_CONFIDENCE_THRESHOLD', 0.6)
        """
        🎯 Aplicar filtrado por confianza a las predicciones
        
        Si la confianza es baja, convertir a HOLD para evitar trades arriesgados
        
        Args:
            y_pred_proba: Probabilidades de predicción [n_samples, 3]
            confidence_threshold: Umbral mínimo de confianza (0.6 = 60%)
            
        Returns:
            y_pred_filtered: Predicciones filtradas por confianza
        """
        # Calcular predicción original
        y_pred_original = np.argmax(y_pred_proba, axis=1)
        
        # Calcular confianza máxima para cada predicción
        max_confidence = np.max(y_pred_proba, axis=1)
        
        # Aplicar filtro de confianza
        y_pred_filtered = y_pred_original.copy()
        low_confidence_mask = max_confidence < confidence_threshold
        
        # Convertir predicciones de baja confianza a HOLD (clase 1)
        y_pred_filtered[low_confidence_mask] = 1  # HOLD
        
        # Estadísticas del filtrado
        total_predictions = len(y_pred_original)
        filtered_count = np.sum(low_confidence_mask)
        high_conf_buy_sell = np.sum((y_pred_filtered != 1) & (max_confidence >= confidence_threshold))
        
        print(f"\n🎯 FILTRADO POR CONFIANZA (umbral: {confidence_threshold:.1%}):")
        print(f"   📊 Total predicciones: {total_predictions}")
        print(f"   📊 Filtradas a HOLD: {filtered_count} ({filtered_count/total_predictions:.1%})")
        print(f"   📊 BUY/SELL alta confianza: {high_conf_buy_sell} ({high_conf_buy_sell/total_predictions:.1%})")
        print(f"   📊 Confianza promedio: {np.mean(max_confidence):.3f}")
        
        return y_pred_filtered

    
    def create_crypto_optimized_model(self, input_shape: tuple):
        """
        🚀 ARQUITECTURA MODERNA PARA CRYPTO TRADING (2024)
        
        Características especializadas:
        - Multi-Head Temporal Attention eficiente
        - Multi-timeframe processing
        - Volatility-aware components
        - Market regime detection
        - ~150K parámetros (balance perfecto eficiencia/capacidad)
        
        Inspirada en:
        - Temporal Fusion Transformers
        - TSMixer (Google Research 2023)
        - VolatilityNet (Finance-specific)
        """

        def efficient_multi_head_attention(x, num_heads=4, key_dim=16, dropout=0.1):
            """
            🧠 Multi-Head Attention eficiente para series temporales financieras
            Inspirado en Temporal Fusion Transformers pero más liviano
            """
            # Usar MultiHeadAttention nativo de Keras (más eficiente)
            attention_layer = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=key_dim,
                dropout=dropout,
                use_bias=False  # Reducir parámetros
            )
            
            # Self-attention temporal
            attended = attention_layer(x, x)
            
            # Residual connection + Layer norm (estilo Transformer)
            x = tf.keras.layers.Add()([x, attended])
            x = tf.keras.layers.LayerNormalization()(x)
            
            return x

        def multi_timeframe_block(x, base_filters=32):
            """
            📊 Procesamiento multi-timeframe especializado para crypto
            Simula análisis de 1min, 3min, 15min simultáneamente
            """
            # Diferentes escalas temporales para crypto
            filters_per_branch = base_filters // 3  # Dividir entre 3 ramas
            
            short_term = tf.keras.layers.Conv1D(
                filters_per_branch, 3, dilation_rate=1, 
                padding='causal', activation='swish'
            )(x)  # ~3 min
            
            medium_term = tf.keras.layers.Conv1D(
                filters_per_branch, 5, dilation_rate=3,
                padding='causal', activation='swish'  
            )(x)  # ~15 min
            
            long_term = tf.keras.layers.Conv1D(
                filters_per_branch, 7, dilation_rate=5,
                padding='causal', activation='swish'
            )(x)  # ~1 hour
            
            # Feature fusion inteligente
            timeframes = tf.keras.layers.Concatenate()([short_term, medium_term, long_term])
            
            # Reducir dimensiones con 1x1 conv (más eficiente que Dense)
            fused = tf.keras.layers.Conv1D(base_filters, 1, activation='swish')(timeframes)
            fused = tf.keras.layers.LayerNormalization()(fused)
            
            # Skip connection con proyección automática
            input_filters = x.shape[-1]
            if input_filters != base_filters:
                x_proj = tf.keras.layers.Conv1D(base_filters, 1)(x)
            else:
                x_proj = x
                
            return tf.keras.layers.Add()([fused, x_proj])

        def volatility_aware_block(x, filters=32):
            """
            📈 Bloque especializado en volatilidad para crypto
            Detecta y adapta a cambios de régimen de mercado
            """
            # Detectar volatilidad local (ventana móvil)
            vol_detector = tf.keras.layers.Conv1D(
                8, 5, padding='causal', activation='sigmoid'
            )(x)  # Pequeño detector de volatilidad
            
            # Procesamiento adaptativo basado en volatilidad
            low_vol_path = tf.keras.layers.Conv1D(
                filters // 2, 3, padding='causal', activation='relu'
            )(x)
            
            high_vol_path = tf.keras.layers.Conv1D(
                filters // 2, 3, padding='causal', activation='tanh'
            )(x)
            
            # Gating mechanism (como LSTM pero más simple)
            combined = tf.keras.layers.Concatenate()([low_vol_path, high_vol_path])
            
            # Reducir y normalizar
            output = tf.keras.layers.Conv1D(filters, 1, activation='swish')(combined)
            output = tf.keras.layers.LayerNormalization()(output)
            
            return output

        def crypto_feature_extractor(x, filters=48):
            """
            💰 Extractor de features específicas de crypto trading
            Patrones comunes: pump/dump, breakouts, support/resistance
            """
            # Detector de momentum (cambios rápidos)
            momentum = tf.keras.layers.Conv1D(
                filters // 3, 3, dilation_rate=1, 
                padding='causal', activation='relu'
            )(x)
            
            # Detector de reversión (patrones de vuelta)
            reversal = tf.keras.layers.Conv1D(
                filters // 3, 5, dilation_rate=2,
                padding='causal', activation='tanh'
            )(x)
            
            # Detector de breakout (rupturas de rangos)
            breakout = tf.keras.layers.Conv1D(
                filters // 3, 7, dilation_rate=3,
                padding='causal', activation='swish'
            )(x)
            
            # Fusionar patrones
            crypto_features = tf.keras.layers.Concatenate()([momentum, reversal, breakout])
            crypto_features = tf.keras.layers.LayerNormalization()(crypto_features)
            
            return crypto_features
        
        # 🏗️ ARQUITECTURA PRINCIPAL - MODERNA Y EFICIENTE
        inputs = tf.keras.layers.Input(shape=input_shape, name='crypto_sequence_input')
        
        # 1️⃣ ENTRADA Y NORMALIZACIÓN
        x = tf.keras.layers.LayerNormalization()(inputs)
        
        # Embedding inicial - proyectar directamente a 48 para compatibilidad
        x = tf.keras.layers.Conv1D(48, 1, activation='swish')(x)
        
        # 2️⃣ PRIMER BLOQUE: MULTI-TIMEFRAME PROCESSING
        x = multi_timeframe_block(x, base_filters=48)
        x = tf.keras.layers.Dropout(0.1)(x)
        
        # 3️⃣ SEGUNDO BLOQUE: VOLATILITY-AWARE PROCESSING (ajustar a 48 filtros)
        vol_features = volatility_aware_block(x, filters=48)
        x = tf.keras.layers.Add()([x, vol_features])  # Residual connection
        x = tf.keras.layers.Dropout(0.15)(x)
        
        # 4️⃣ TERCER BLOQUE: CRYPTO-SPECIFIC FEATURES (reducido para evitar concatenación masiva)
        crypto_features = crypto_feature_extractor(x, filters=32)  # Reducido de 48 a 32
        x = tf.keras.layers.Concatenate()([x, crypto_features])  # 48 + 32 = 80
        
        # Reducir dimensiones después de concatenación
        x = tf.keras.layers.Conv1D(64, 1, activation='swish')(x)  # 80 -> 64
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # 5️⃣ ATTENTION BLOCK (el más importante)
        x = efficient_multi_head_attention(x, num_heads=4, key_dim=16, dropout=0.1)
        
        # 6️⃣ SEGUNDO ATTENTION (para patrones complejos)
        x = efficient_multi_head_attention(x, num_heads=2, key_dim=32, dropout=0.15)
        
        # 7️⃣ AGREGACIÓN TEMPORAL INTELIGENTE
        # Múltiples formas de agregar información temporal
        avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
        max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
        
        # Atención sobre las últimas posiciones (más relevantes para trading)
        last_timesteps = tf.keras.layers.Lambda(lambda x: x[:, -3:, :])(x)  # Últimos 3 timesteps
        last_avg = tf.keras.layers.GlobalAveragePooling1D()(last_timesteps)
        
        # Combinar todas las agregaciones
        pooled = tf.keras.layers.Concatenate()([avg_pool, max_pool, last_avg])
        
        # 8️⃣ CAPAS DE DECISIÓN EFICIENTES
        # Reducir dimensiones primero
        x = tf.keras.layers.Dense(96, activation='swish', 
                                kernel_initializer='he_normal',
                                kernel_regularizer=tf.keras.regularizers.l2(0.001))(pooled)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Capa intermedia para patrones complejos
        x = tf.keras.layers.Dense(32, activation='swish',
                                kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = tf.keras.layers.LayerNormalization()(x)  
        x = tf.keras.layers.Dropout(0.4)(x)
        
        # 9️⃣ CAPA DE SALIDA
        outputs = tf.keras.layers.Dense(3, activation='softmax', 
                                      kernel_initializer='glorot_uniform',
                                      name='trading_predictions')(x)
        
        # 🏆 CREAR MODELO FINAL
        model = tf.keras.Model(
            inputs=inputs, 
            outputs=outputs, 
            name='CryptoOptimizedTransformer_v2024'
        )
        
        # 📊 OPTIMIZADOR ESPECIALIZADO
        timeframe = getattr(self, 'timeframe', '3m')
        if timeframe == '1m':
            learning_rate = 5e-5  # Más conservador para timeframes rápidos
        elif timeframe == '3m':
            learning_rate = 1e-4  # Balance óptimo
        elif timeframe == '5m':
            learning_rate = 1.5e-4  
        else:
            learning_rate = 2e-4
        
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=0.01,  # L2 regularization dentro del optimizador
            clipnorm=1.0
        )
        
        # 🎯 COMPILACIÓN
        loss_function = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
        
        model.compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=metrics
        )
        
        return model

    def create_enhanced_crypto_tcn(self, input_shape: tuple):
        """
        🚀 ARQUITECTURA TCN OPTIMIZADA PARA CRYPTO (VERSIÓN EFICIENTE 2024)
        
        ✅ OPTIMIZACIONES IMPLEMENTADAS:
        - Reducción inteligente de parámetros sin perder capacidad
        - Stack TCN más eficiente (4 bloques vs 6)
        - Atención simplificada pero efectiva
        - Pooling más directo
        - ~80K parámetros (vs 200K original)
        
        🎯 TARGET: ~1 minuto por época
        """
        
        def efficient_tcn_block(x, filters, dilation_rate, dropout=0.1):
            """
            🔄 Bloque TCN eficiente con menos operaciones pero igual efectividad
            """
            # Una sola convolución causal más efectiva
            conv = tf.keras.layers.Conv1D(
                filters, 3, dilation_rate=dilation_rate, 
                padding='causal', activation='swish',
                kernel_initializer='he_normal'
            )(x)
            conv = tf.keras.layers.LayerNormalization()(conv)
            conv = tf.keras.layers.Dropout(dropout)(conv)
            
            # Conexión residual simplificada
            if x.shape[-1] != filters:
                x = tf.keras.layers.Conv1D(
                    filters, 1, kernel_initializer='he_normal'
                )(x)
            
            # Suma residual
            residual = tf.keras.layers.Add()([x, conv])
            
            return residual
        
        def lite_crypto_attention(x):
            """
            🧠 Atención temporal simplificada pero efectiva para crypto
            """
            # Atención más ligera con menos heads
            attention = tf.keras.layers.MultiHeadAttention(
                num_heads=2, key_dim=16, dropout=0.1  # Reducido de 4 a 2 heads
            )(x, x)
            
            # Conexión residual directa
            output = tf.keras.layers.Add()([x, attention])
            output = tf.keras.layers.LayerNormalization()(output)
            
            return output
        
        def efficient_pooling_block(x):
            """
            📊 Pooling eficiente manteniendo información clave
            """
            # Solo las estrategias más efectivas
            avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
            max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
            
            # Últimos timesteps (más relevantes para trading)
            last_timesteps = tf.keras.layers.Lambda(lambda x: x[:, -3:, :])(x)  # Reducido de 5 a 3
            last_avg = tf.keras.layers.GlobalAveragePooling1D()(last_timesteps)
            
            # Combinar agregaciones
            pooled = tf.keras.layers.Concatenate()([avg_pool, max_pool, last_avg])
            
            return pooled
        
        # 🏗️ ARQUITECTURA PRINCIPAL OPTIMIZADA
        inputs = tf.keras.layers.Input(shape=input_shape, name='efficient_tcn_input')
        
        # 1️⃣ NORMALIZACIÓN INICIAL
        x = tf.keras.layers.LayerNormalization()(inputs)
        
        # 2️⃣ PROYECCIÓN INICIAL MÁS PEQUEÑA
        x = tf.keras.layers.Conv1D(
            48, 1, activation='swish',  # Reducido de 64 a 48
            kernel_initializer='he_normal'
        )(x)
        
        # 3️⃣ STACK TCN OPTIMIZADO (4 bloques vs 6 original)
        dilations = [1, 2, 4, 8]  # Reducido pero manteniendo cobertura temporal
        filters = [48, 48, 40, 40]  # Filtros más pequeños pero suficientes
        dropouts = [0.1, 0.15, 0.2, 0.2]
        
        print(f"🔧 Construyendo stack TCN eficiente con {len(dilations)} bloques...")
        for i, (dilation, num_filters, dropout) in enumerate(zip(dilations, filters, dropouts)):
            print(f"   📊 Bloque {i+1}: filters={num_filters}, dilation={dilation}, dropout={dropout}")
            x = efficient_tcn_block(x, num_filters, dilation, dropout=dropout)
        
        # 4️⃣ ATENCIÓN TEMPORAL LIGERA
        print(f"🧠 Aplicando atención temporal eficiente...")
        x = lite_crypto_attention(x)
        
        # 5️⃣ POOLING EFICIENTE
        print(f"📊 Aplicando pooling optimizado...")
        x = efficient_pooling_block(x)
        
        # 6️⃣ CAPAS DE DECISIÓN MÁS LIGERAS PERO EFECTIVAS
        print(f"🎯 Construyendo capas de decisión optimizadas...")
        
        # Primera capa densa (reducida)
        x = tf.keras.layers.Dense(
            64, activation='swish',  # Reducido de 128 a 64
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Segunda capa densa (reducida)
        x = tf.keras.layers.Dense(
            32, activation='swish',  # Reducido de 64 a 32
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        
        # 7️⃣ CAPA DE SALIDA
        outputs = tf.keras.layers.Dense(
            3, activation='softmax',
            kernel_initializer='glorot_uniform',
            name='efficient_tcn_predictions'
        )(x)
        
        # 🏆 CREAR MODELO FINAL
        model = tf.keras.Model(
            inputs=inputs,
            outputs=outputs,
            name='EfficientCryptoTCN_v2024'
        )
        
        # 📊 OPTIMIZADOR AJUSTADO PARA CONVERGENCIA RÁPIDA
        timeframe = getattr(self, 'timeframe', '3m')
        if timeframe == '1m':
            learning_rate = 5e-4  # Más agresivo para convergencia rápida
        elif timeframe == '3m':
            learning_rate = 8e-4  # Balance óptimo
        elif timeframe == '5m':
            learning_rate = 1e-3
        else:
            learning_rate = 1.2e-3
        
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=0.01,  # Regularización ajustada
            clipnorm=1.0
        )
        
        # 🎯 COMPILACIÓN
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 📈 MOSTRAR ESTADÍSTICAS DEL MODELO
        print(f"\n🎯 MODELO OPTIMIZADO CREADO:")
        print(f"   📊 Parámetros estimados: ~80K (vs 200K original)")
        print(f"   ⚡ Reducción esperada de tiempo: ~85%")
        print(f"   🎯 Target tiempo por época: ~1 minuto")
        
        return model

    def compare_tcn_architectures(self, input_shape: tuple):
        """
        📊 COMPARAR DIFERENTES ARQUITECTURAS TCN PARA AYUDA EN LA SELECCIÓN
        
        Args:
            input_shape: Forma de entrada para el modelo
            
        Returns:
            Diccionario con comparación de arquitecturas
        """
        print(f"📊 COMPARANDO ARQUITECTURAS TCN DISPONIBLES...")
        
        architectures = {}
        
        # 1️⃣ ARQUITECTURA TCN V3 EFICIENTE
        try:
            model_v3 = self.create_efficient_tcn_v3(input_shape)
            architectures['efficient_v3'] = {
                'name': 'TCN V3 Eficiente',
                'params': model_v3.count_params(),
                'speed': '2-3x más rápido',
                'memory': 'Bajo',
                'complexity': 'Baja',
                'best_for': 'Prototipado rápido, datasets pequeños',
                'model': model_v3
            }
            print(f"✅ TCN V3 Eficiente: {model_v3.count_params():,} parámetros")
        except Exception as e:
            print(f"❌ Error creando TCN V3: {e}")
        
        # 2️⃣ ARQUITECTURA TCN ENHANCED
        try:
            model_enhanced = self.create_enhanced_crypto_tcn(input_shape)
            architectures['enhanced'] = {
                'name': 'TCN Enhanced',
                'params': model_enhanced.count_params(),
                'speed': 'Estándar',
                'memory': 'Medio',
                'complexity': 'Alta',
                'best_for': 'Producción, datasets grandes, máxima precisión',
                'model': model_enhanced
            }
            print(f"✅ TCN Enhanced: {model_enhanced.count_params():,} parámetros")
        except Exception as e:
            print(f"❌ Error creando TCN Enhanced: {e}")
        
        # 3️⃣ ARQUITECTURA TCN ORIGINAL
        try:
            model_original = self.create_crypto_optimized_model(input_shape)
            architectures['original'] = {
                'name': 'TCN Original',
                'params': model_original.count_params(),
                'speed': 'Estándar',
                'memory': 'Medio',
                'complexity': 'Media',
                'best_for': 'Balance entre velocidad y precisión',
                'model': model_original
            }
            print(f"✅ TCN Original: {model_original.count_params():,} parámetros")
        except Exception as e:
            print(f"❌ Error creando TCN Original: {e}")
        
        # 📊 RESUMEN DE COMPARACIÓN
        print(f"\n📊 RESUMEN DE ARQUITECTURAS TCN:")
        print(f"{'='*80}")
        for key, arch in architectures.items():
            print(f"🏗️ {arch['name']}:")
            print(f"   📊 Parámetros: {arch['params']:,}")
            print(f"   ⚡ Velocidad: {arch['speed']}")
            print(f"   💾 Memoria: {arch['memory']}")
            print(f"   🔧 Complejidad: {arch['complexity']}")
            print(f"   🎯 Mejor para: {arch['best_for']}")
            print(f"   {'-'*40}")
        
        # 🎯 RECOMENDACIÓN AUTOMÁTICA
        if 'efficient_v3' in architectures:
            print(f"🚀 RECOMENDACIÓN: TCN V3 Eficiente para entrenamiento rápido")
            print(f"   💡 Ideal para: Experimentación, prototipado, datasets pequeños")
        elif 'enhanced' in architectures:
            print(f"🚀 RECOMENDACIÓN: TCN Enhanced para máxima precisión")
            print(f"   💡 Ideal para: Producción, datasets grandes, máxima calidad")
        else:
            print(f"🚀 RECOMENDACIÓN: TCN Original como fallback")
            print(f"   💡 Ideal para: Casos generales, balance velocidad/precisión")
        
        return architectures

    # 🔧 FUNCIONES AUXILIARES PARA OPTIMIZACIÓN ADICIONAL

    def get_efficient_training_config(self):
        """
        ⚡ Configuración de entrenamiento optimizada para velocidad
        """
        config = {
            'batch_size': 128,  # Incrementar batch size si tienes GPU decente
            'mixed_precision': True,  # Usar mixed precision para acelerar
            'use_multiprocessing': True,
            'workers': 4,
            'max_queue_size': 20
        }
        
        # Ajustar batch size según timeframe
        timeframe = getattr(self, 'timeframe', '3m')
        if timeframe == '1m':
            config['batch_size'] = 256  # Más muestras, puede permitir batch más grande
        elif timeframe in ['3m', '5m']:
            config['batch_size'] = 128
        else:
            config['batch_size'] = 64
        
        return config

    def setup_mixed_precision(self):
        """
        🚀 Configurar mixed precision para acelerar entrenamiento
        """
        try:
            import tensorflow as tf
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("✅ Mixed precision habilitado - aceleración esperada: 30-50%")
            return True
        except Exception as e:
            print(f"⚠️ No se pudo habilitar mixed precision: {e}")
            return False

    def create_efficient_tcn_v3(self, input_shape: tuple):
        """
        ⚡ ARQUITECTURA TCN V3 EFICIENTE Y RÁPIDA (2024)
        
        ✅ CARACTERÍSTICAS PRINCIPALES:
        - Arquitectura limpia y directa para entrenamiento rápido
        - Bloques residuales TCN simples con BatchNormalization
        - Menos parámetros (~100K) para evitar overfitting
        - Convergencia más estable y predecible
        
        🎯 BENEFICIOS:
        - Entrenamiento 2-3x más rápido que arquitectura enhanced
        - Menor uso de memoria GPU/RAM
        - Mejor generalización en datasets pequeños
        - Ideal para prototipado y experimentación rápida
        
        🏗️ ARQUITECTURA:
        - 4 bloques TCN residuales con dilataciones [1, 2, 4, 8]
        - Filtros consistentes (64) para estabilidad
        - Activación ReLU estándar para convergencia rápida
        - Dropout progresivo para regularización
        """
        
        def residual_block(x, filters, dilation_rate, dropout_rate=0.2):
            """
            🔄 Bloque residual TCN simple y eficiente
            
            Características:
            - Convoluciones causales con dilatación
            - BatchNormalization para estabilidad
            - Activación ReLU estándar
            - Dropout progresivo
            - Conexiones residuales con proyección automática
            """
            prev_x = x
            
            # Primera convolución
            x = tf.keras.layers.Conv1D(
                filters, 3, padding='causal', dilation_rate=dilation_rate,
                kernel_initializer='he_normal'
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)
            
            # Segunda convolución
            x = tf.keras.layers.Conv1D(
                filters, 3, padding='causal', dilation_rate=dilation_rate,
                kernel_initializer='he_normal'
            )(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)
            
            # ✅ CONEXIÓN RESIDUAL CON PROYECCIÓN AUTOMÁTICA
            if prev_x.shape[-1] != filters:
                prev_x = tf.keras.layers.Conv1D(
                    filters, 1, padding='same', kernel_initializer='he_normal'
                )(prev_x)
            
            # Suma residual
            return tf.keras.layers.Add()([prev_x, x])
        
        # 🏗️ ARQUITECTURA PRINCIPAL EFICIENTE
        inputs = tf.keras.layers.Input(shape=input_shape, name='efficient_tcn_v3_input')
        x = inputs
        
        # ✅ NORMALIZACIÓN INICIAL
        x = tf.keras.layers.BatchNormalization()(x)
        
        # ✅ PROYECCIÓN INICIAL A FILTROS OBJETIVO
        x = tf.keras.layers.Conv1D(
            self.tcn_v3_filters, 1, activation='relu',
            kernel_initializer='he_normal'
        )(x)
        
        # 🔄 STACK DE BLOQUES TCN EFICIENTES
        print(f"⚡ Construyendo stack TCN V3 eficiente con {len(self.tcn_v3_dilations)} bloques...")
        for i, dilation in enumerate(self.tcn_v3_dilations):
            dropout_rate = 0.1 + (i * 0.05)  # Dropout progresivo: 0.1, 0.15, 0.2, 0.25
            print(f"   📊 Bloque {i+1}: filters={self.tcn_v3_filters}, dilation={dilation}, dropout={dropout_rate:.2f}")
            x = residual_block(x, self.tcn_v3_filters, dilation, dropout_rate=dropout_rate)
        
        # 📊 POOLING GLOBAL SIMPLE
        print(f"📊 Aplicando pooling global simple...")
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # 🎯 CAPAS DE DECISIÓN OPTIMIZADAS
        print(f"🎯 Construyendo capas de decisión optimizadas...")
        
        # Primera capa densa
        x = tf.keras.layers.Dense(
            128, activation='relu',
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Segunda capa densa
        x = tf.keras.layers.Dense(
            64, activation='relu',
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # 🎯 CAPA DE SALIDA
        outputs = tf.keras.layers.Dense(
            3, activation='softmax',
            kernel_initializer='glorot_uniform',
            name='efficient_tcn_v3_predictions'
        )(x)
        
        # 🏆 CREAR MODELO FINAL
        model = tf.keras.Model(
            inputs=inputs,
            outputs=outputs,
            name='EfficientTCN_V3_2024'
        )
        
        # 📊 OPTIMIZADOR OPTIMIZADO PARA ARQUITECTURA EFICIENTE
        timeframe = getattr(self, 'timeframe', '3m')
        if timeframe == '1m':
            learning_rate = 5e-5  # Más agresivo para arquitectura eficiente
        elif timeframe == '3m':
            learning_rate = 1e-4  # Balance óptimo para V3
        elif timeframe == '5m':
            learning_rate = 1.5e-4
        else:
            learning_rate = 2e-4
        
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=0.003,  # L2 regularization más agresivo para V3
            clipnorm=1.0  # Gradient clipping menos restrictivo
        )
        
        # 🎯 COMPILACIÓN
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 📈 RESUMEN DEL MODELO EFICIENTE
        print(f"\n⚡ MODELO TCN V3 EFICIENTE CREADO:")
        print(f"   📊 Parámetros totales: {model.count_params():,}")
        print(f"   🧠 Arquitectura: {len(self.tcn_v3_dilations)} bloques TCN residuales")
        print(f"   🎯 Dilataciones: {self.tcn_v3_dilations}")
        print(f"   🔧 Filtros por bloque: {self.tcn_v3_filters}")
        print(f"   💰 Learning rate: {learning_rate:.2e}")
        print(f"   ⚡ Optimizador: AdamW optimizado para V3")
        print(f"   🚀 Entrenamiento estimado: 2-3x más rápido que enhanced")
        
        return model

    def create_definitive_tcn_model(self, input_shape: tuple):
        """
        🚀 MÉTODO PRINCIPAL: Crea el modelo TCN según la configuración seleccionada
        
        ✅ ARQUITECTURAS DISPONIBLES:
        - 'enhanced': Nueva arquitectura TCN mejorada (RECOMENDADO)
        - 'original': Arquitectura crypto optimizada original
        - 'hybrid': Combinación de ambas arquitecturas
        - 'efficient_v3': Arquitectura TCN V3 eficiente y rápida (NUEVA)
        """
        print(f"🏗️ Seleccionando arquitectura TCN: {self.tcn_architecture}")
        
        # 🚀 PRIORIDAD: ARQUITECTURA TCN V3 EFICIENTE SI ESTÁ HABILITADA
        if self.use_efficient_tcn_v3:
            print(f"⚡ Usando ARQUITECTURA TCN V3 EFICIENTE...")
            return self.create_efficient_tcn_v3(input_shape)
        
        # 🔄 ARQUITECTURAS EXISTENTES
        if self.tcn_architecture == 'enhanced':
            print(f"🚀 Usando ARQUITECTURA TCN MEJORADA...")
            return self.create_enhanced_crypto_tcn(input_shape)
        elif self.tcn_architecture == 'hybrid':
            print(f"🔀 Usando ARQUITECTURA HÍBRIDA...")
            # Implementar lógica híbrida si se desea
            return self.create_enhanced_crypto_tcn(input_shape)
        else:
            print(f"📊 Usando ARQUITECTURA ORIGINAL...")
            return self.create_crypto_optimized_model(input_shape)

    # MÉTODO ANTIGUO MANTENIDO PARA REFERENCIA
    def create_old_tcn_model(self, input_shape: tuple):
        """🔧 Método TCN original (mantenido para comparaciones)"""
        
        def attention_layer(x):
            """🎯 Mecanismo de atención robusto para dimensiones dinámicas y estáticas"""

            # ✅ CORRECCIÓN: Obtener dimensiones de forma segura
            shape = tf.shape(x)
            batch_size = shape[0]
            seq_len = shape[1]
            features = shape[2]

            # ✅ CORRECCIÓN: Usar Dense layers que manejan dimensiones dinámicas
            # Generar pesos de atención usando Dense layers
            attention_weights = tf.keras.layers.Dense(1, activation='tanh')(x)
            attention_weights = tf.keras.layers.Softmax(axis=1)(attention_weights)

            # ✅ CORRECCIÓN DIMENSIONAL: Asegurar que attention_weights tenga las dimensiones correctas
            # attention_weights tiene shape (batch, seq_len, 1) después de Dense(1)
            # No necesitamos expand_dims adicional

            # Aplicar atención usando Multiply directamente
            context = tf.keras.layers.Multiply()([x, attention_weights])

            # ✅ CORRECCIÓN: Conexión residual segura
            return tf.keras.layers.Add()([x, context])

        def volatility_adaptation(x):
            """🎯 Adaptación a volatilidad del mercado con dimensiones dinámicas"""

            # ✅ CORRECCIÓN: Obtener número de features de la forma estática del tensor
            # Usar `x.shape[-1]` para obtener un entero, no un tensor simbólico
            features = x.shape[-1]
            if features is None:
                raise ValueError("La dimensión de canales (features) debe estar definida.")

            # Detector de volatilidad
            vol_detector = tf.keras.layers.Conv1D(1, 3, padding='same', activation='sigmoid')(x)

            # Gate de volatilidad. Usa el número de features estático.
            vol_gate = tf.keras.layers.Conv1D(features, 1, activation='sigmoid')(vol_detector)

            # Aplicar gate de volatilidad
            gated = tf.keras.layers.Multiply()([x, vol_gate])

            # Conexión residual
            return tf.keras.layers.Add()([x, gated])

        print(f"🚀 Creando TCN optimizado para crypto ({self.timeframe})...")

        # ✅ CORRECCIÓN: Usar (None, num_features) para aceptar secuencias de cualquier longitud
        # Esto hace el modelo mucho más flexible y robusto.
        num_features = input_shape[1]
        inputs = tf.keras.layers.Input(shape=(None, num_features))
        x = tf.keras.layers.LayerNormalization()(inputs)

        # Feature enhancement inicial REDUCIDO
        x = tf.keras.layers.Conv1D(32, 1, padding='same', activation='relu')(x)

        # Configuración REDUCIDA para evitar overfitting con pocos datos
        if hasattr(self, 'timeframe'):
            # ✅ CORRECCIÓN: ELIMINAR RESTRICCIONES FORZADAS PARA 1M
            # El usuario puede elegir libremente la arquitectura para cualquier timeframe
            if self.timeframe == '1m':
                # 🎯 CONFIGURACIÓN OPTIMIZADA PERO NO RESTRICTIVA para 1m
                # El usuario puede elegir entre configuraciones optimizadas o completas
                if hasattr(self, 'use_optimized_1m') and self.use_optimized_1m:
                    print("⚡ Usando configuración optimizada para 1m (recomendado para alta frecuencia)")
                    dilation_groups = [[1, 2], [4, 6]]  # Optimizado para velocidad
                    filters_progression = [32, 48]      # Menos filtros para rapidez
                else:
                    print("🚀 Usando configuración completa para 1m (máxima precisión)")
                    dilation_groups = [[1, 2], [4, 8], [16, 32]]  # Configuración completa
                    filters_progression = [32, 48, 64]            # Filtros completos
            elif self.timeframe == '5m':
                dilation_groups = [[1, 2], [4, 8]]
                filters_progression = [24, 36]
            else:
                dilation_groups = [[1, 3], [6, 12]]
                filters_progression = [24, 32]
        else:
            # Configuración por defecto REDUCIDA
            dilation_groups = [[1, 2], [4, 8]]
            filters_progression = [24, 36]

        # Bloques multi-escala con DROPOUT AUMENTADO
        for i, (dilations, filters) in enumerate(zip(dilation_groups, filters_progression)):
            x = multi_scale_block(x, filters, dilations, dropout_rate=0.2 + i * 0.1)  # MÁS DROPOUT

        # ELIMINAR atención intermedia para reducir parámetros
        # Solo una atención al final

        # ELIMINAR volatility adaptation (muchos parámetros)
        # x = volatility_adaptation(x)

        # SIMPLIFICADO: Solo 2 extractores de tendencias en lugar de 3
        short_trend = tf.keras.layers.Conv1D(16, 3, dilation_rate=1, padding='causal', activation='tanh')(x)  # REDUCIDO: 16 filtros
        medium_trend = tf.keras.layers.Conv1D(16, 5, dilation_rate=3, padding='causal', activation='tanh')(x)  # REDUCIDO: 16 filtros
        
        # Concatenar y normalizar
        trend_features = tf.keras.layers.Concatenate()([short_trend, medium_trend])  # Solo 2 en lugar de 3
        trend_features = tf.keras.layers.LayerNormalization()(trend_features)

        # Normalizar entradas y combinar - REDUCIDO
        x_normalized = tf.keras.layers.Conv1D(48, 1, padding='same')(x)  # REDUCIDO: 48 en lugar de 96
        combined = tf.keras.layers.Concatenate()([x_normalized, trend_features])

        # Procesar combinación - MUY REDUCIDO
        x = tf.keras.layers.Conv1D(64, 1, padding='same', activation='relu')(combined)  # REDUCIDO: 64 en lugar de 256

        # Atención final
        x = attention_layer(x)

        # Agregación temporal dual
        avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
        max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
        pooled = tf.keras.layers.Concatenate()([avg_pool, max_pool])

        # Capas de decisión REDUCIDAS con mayor regularización
        x = tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal',
                                kernel_regularizer=tf.keras.regularizers.l2(0.01))(pooled)  # REDUCIDO: 64 en lugar de 256, L2 más fuerte
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)  # AUMENTADO: más dropout

        x = tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_normal',
                                kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)  # REDUCIDO: 32 en lugar de 128
        x = tf.keras.layers.Dropout(0.4)(x)  # AUMENTADO: más dropout

        # Output
        outputs = tf.keras.layers.Dense(3, activation='softmax',
                                      kernel_initializer='glorot_uniform',
                                      bias_initializer='zeros')(x)

        # Crear modelo
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Optimizador mejorado
        timeframe = getattr(self, 'timeframe', '5m')
        if timeframe == '1m':
            learning_rate = 1e-4  # REDUCIDO: 3e-4 → 1e-4
        elif timeframe == '5m':
            learning_rate = 1.5e-4  # REDUCIDO: 5e-4 → 1.5e-4
        else:
            learning_rate = 2e-4  # REDUCIDO: 7e-4 → 2e-4

        # ✅ CORRECCIÓN: Learning rate fijo para máxima estabilidad
        # Usar learning rate fijo que ya está probado y funciona
        optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=learning_rate,  # LR fijo sin schedule
            clipnorm=1.0
        )

        # 🎯 NUEVA LÓGICA: Usar profit-aware loss si está habilitado
        if (hasattr(self, 'use_profit_aware_loss') and 
            self.use_profit_aware_loss and 
            PROFIT_AWARE_AVAILABLE):
            print(f"🎯 Usando Profit-Aware Loss: {getattr(self, 'loss_type', 'combined')}")
            loss_function = 'sparse_categorical_crossentropy'  # Se recompilará después
            metrics = ['accuracy']
        else:
            # Método tradicional
            loss_function = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        model.compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=metrics
        )

        param_count = model.count_params()
        print(f"✅ TCN Optimizado creado: {param_count:,} parámetros")
        print(f"   🎯 Arquitectura: Multi-scale + Attention + Volatility-adaptive")
        print(f"   📊 LR: {learning_rate}")

        return model


    def evaluate_model_with_trading_metrics(self, model: tf.keras.Model, X_test: np.ndarray,
                                          y_test: np.ndarray, symbol: str) -> Dict:
        """🎯 Evaluar modelo con métricas específicas de trading"""

        print(f"📊 Evaluando modelo con métricas de trading para {symbol}...")

        # Predicciones
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred_raw = np.argmax(y_pred_proba, axis=1)
        
        # 🎯 APLICAR FILTRADO POR CONFIANZA
        confidence_threshold = TRAINING_CONFIG.get('MIN_CONFIDENCE_THRESHOLD', 0.65)
        y_pred = self.apply_confidence_filtering(y_pred_proba, confidence_threshold=confidence_threshold)

        # Calcular métricas de trading
        trading_metrics = self.trading_metrics.calculate_trading_metrics(
            y_test, y_pred, y_pred_proba
        )

        # Imprimir reporte detallado
        self.trading_metrics.print_trading_report(trading_metrics, symbol, self.timeframe)

        # Guardar gráfico de métricas
        model_name = f"{symbol.lower()}_{self.timeframe}_{self.prediction_horizon}h_{self.lookback_window}w"
        plot_path = f'models/adaptive_{model_name}/trading_metrics.png'

        try:
            self.trading_metrics.save_metrics_plot(trading_metrics, symbol, self.timeframe, plot_path)
        except Exception as e:
            print(f"⚠️  Error guardando gráfico: {e}")

        # Guardar métricas en archivo
        metrics_path = f'models/adaptive_{model_name}/trading_metrics.json'
        try:
            import json

            def convert_numpy_types(obj):
                """🔄 Convertir tipos numpy a tipos nativos de Python para JSON"""
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj

            # Convertir todas las métricas a tipos compatibles con JSON
            metrics_for_json = convert_numpy_types(trading_metrics)

            with open(metrics_path, 'w') as f:
                json.dump(metrics_for_json, f, indent=2)
            print(f"✅ Métricas guardadas: {metrics_path}")

        except Exception as e:
            print(f"❌ ERROR guardando archivos: {e}")

        return trading_metrics

    def validate_dynamic_dimensions(self, model: tf.keras.Model) -> bool:
        """🎯 Validar que el modelo maneja dimensiones dinámicas correctamente"""

        print(f"🔍 Validando manejo de dimensiones dinámicas...")

        try:
            # ✅ TEST 1: Verificar que el modelo puede compilarse
            print(f"   📊 Test 1: Compilación del modelo...")

            # ✅ TEST 2: Verificar que puede procesar datos con diferentes tamaños
            print(f"   📊 Test 2: Procesamiento con diferentes tamaños...")

            # Generar datos de prueba con diferentes tamaños
            test_sizes = [(32, 24, 88), (64, 48, 88), (16, 32, 88)]

            for batch_size, seq_len, features in test_sizes:
                test_data = np.random.randn(batch_size, seq_len, features).astype(np.float32)

                try:
                    # Intentar hacer predicción
                    predictions = model.predict(test_data, verbose=0)

                    # Verificar que las predicciones tienen la forma correcta
                    expected_shape = (batch_size, 3)  # 3 clases
                    if predictions.shape != expected_shape:
                        print(f"      ❌ Error: predicciones con forma incorrecta {predictions.shape} != {expected_shape}")
                        return False

                    print(f"      ✅ Tamaño {batch_size}x{seq_len}x{features}: OK")

                except Exception as e:
                    print(f"      ❌ Error con tamaño {batch_size}x{seq_len}x{features}: {e}")
                    return False

            # ✅ TEST 3: Verificar que las capas de atención funcionan
            print(f"   📊 Test 3: Capas de atención...")

            # Verificar que el modelo tiene capas de atención
            attention_layers = [layer for layer in model.layers if 'attention' in layer.name.lower()]
            if not attention_layers:
                print(f"      ⚠️  No se encontraron capas de atención explícitas")
            else:
                print(f"      ✅ Encontradas {len(attention_layers)} capas de atención")

            # ✅ TEST 4: Verificar que las dimensiones se propagan correctamente
            print(f"   📊 Test 4: Propagación de dimensiones...")

            # Usar un tamaño de prueba estándar
            test_data = np.random.randn(16, 24, 88).astype(np.float32)

            # Verificar que no hay errores de dimensiones
            try:
                predictions = model.predict(test_data, verbose=0)
                print(f"      ✅ Propagación de dimensiones: OK")
            except Exception as e:
                print(f"      ❌ Error en propagación de dimensiones: {e}")
                return False

            print(f"✅ Validación de dimensiones dinámicas: PASADO")
            return True

        except Exception as e:
            print(f"❌ Error en validación de dimensiones dinámicas: {e}")
            return False

    def create_callbacks(self, model_dir: str) -> List[tf.keras.callbacks.Callback]:
        """🎯 Crear callbacks con manejo de memory leak"""

        print(f"🧠 Creando callbacks con gestión de memoria...")

        # ✅ CORRECCIÓN: Limpiar backend de Keras antes de crear callbacks
        tf.keras.backend.clear_session()

        # ✅ CORRECCIÓN: Callback personalizado para liberar memoria
        class MemoryCleanupCallback(tf.keras.callbacks.Callback):
            def __init__(self, cleanup_frequency=10):
                super().__init__()
                self.cleanup_frequency = cleanup_frequency
                self.epoch_count = 0

            def on_epoch_end(self, epoch, logs=None):
                self.epoch_count += 1
                if self.epoch_count % self.cleanup_frequency == 0:
                    print(f"🧹 Limpiando memoria en época {self.epoch_count}...")
                    tf.keras.backend.clear_session()
                    # Forzar garbage collection
                    import gc
                    gc.collect()

            def on_train_end(self, logs=None):
                print(f"🧹 Limpieza final de memoria...")
                tf.keras.backend.clear_session()
                import gc
                gc.collect()

        # ✅ CORRECCIÓN: Callback para monitorear uso de memoria
        class MemoryMonitorCallback(tf.keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.memory_usage = []

            def on_epoch_begin(self, epoch, logs=None):
                try:
                    import psutil
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    self.memory_usage.append(memory_mb)
                    if epoch % 5 == 0:  # Reportar cada 5 épocas
                        print(f"📊 Memoria en época {epoch}: {memory_mb:.1f} MB")
                except ImportError:
                    pass  # psutil no disponible

            def on_train_end(self, logs=None):
                if self.memory_usage:
                    max_memory = max(self.memory_usage)
                    print(f"📊 Uso máximo de memoria: {max_memory:.1f} MB")

        # ✅ CORRECCIÓN: Callbacks con configuración optimizada
        callbacks = [
            # Callback para terminar en NaN
            tf.keras.callbacks.TerminateOnNaN(),

            # Early stopping optimizado
            tf.keras.callbacks.EarlyStopping(
                patience=8,
                restore_best_weights=True,
                monitor='val_loss',
                min_delta=0.001,
                verbose=1
            ),

            # Reduce learning rate optimizado
            tf.keras.callbacks.ReduceLROnPlateau(
                patience=5,
                factor=0.5,
                min_lr=1e-6,
                monitor='val_loss',
                verbose=1
            ),

            # Model checkpoint optimizado
            tf.keras.callbacks.ModelCheckpoint(
                f'{model_dir}/best_model.h5',
                save_best_only=True,
                monitor='val_loss',
                save_weights_only=False,
                verbose=1
            ),

            # ✅ NUEVO: Callback para limpiar memoria
            MemoryCleanupCallback(cleanup_frequency=10),

            # ✅ NUEVO: Callback para monitorear memoria
            MemoryMonitorCallback(),

            # ✅ NUEVO: Callback para logging detallado
            tf.keras.callbacks.CSVLogger(
                f'{model_dir}/training_log.csv',
                separator=',',
                append=False
            )
        ]

        print(f"✅ Callbacks creados con gestión de memoria")
        return callbacks

    def cleanup_memory(self):
        """🧹 Limpiar memoria después del entrenamiento"""

        print(f"🧹 Limpiando memoria...")

        try:
            # Limpiar backend de Keras
            tf.keras.backend.clear_session()

            # Forzar garbage collection
            import gc
            gc.collect()

            # Reportar uso de memoria si psutil está disponible
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                print(f"📊 Memoria después de limpieza: {memory_mb:.1f} MB")
            except ImportError:
                print(f"📊 Limpieza de memoria completada")

        except Exception as e:
            print(f"⚠️  Error durante limpieza de memoria: {e}")

    def monitor_memory_usage(self) -> Dict:
        """📊 Monitorear uso de memoria del sistema"""

        try:
            import psutil

            # Información del sistema
            memory_info = {
                'total_memory_mb': psutil.virtual_memory().total / 1024 / 1024,
                'available_memory_mb': psutil.virtual_memory().available / 1024 / 1024,
                'used_memory_mb': psutil.virtual_memory().used / 1024 / 1024,
                'memory_percent': psutil.virtual_memory().percent,
                'process_memory_mb': psutil.Process().memory_info().rss / 1024 / 1024
            }

            print(f"📊 MONITOREO DE MEMORIA:")
            print(f"   📊 Memoria total: {memory_info['total_memory_mb']:.1f} MB")
            print(f"   📊 Memoria disponible: {memory_info['available_memory_mb']:.1f} MB")
            print(f"   📊 Memoria usada: {memory_info['used_memory_mb']:.1f} MB")
            print(f"   📊 Porcentaje usado: {memory_info['memory_percent']:.1f}%")
            print(f"   📊 Memoria del proceso: {memory_info['process_memory_mb']:.1f} MB")

            # ✅ ALERTAS DE MEMORIA
            if memory_info['memory_percent'] > 90:
                print(f"⚠️  ADVERTENCIA: Uso de memoria crítico ({memory_info['memory_percent']:.1f}%)")
            elif memory_info['memory_percent'] > 80:
                print(f"⚠️  ADVERTENCIA: Uso de memoria alto ({memory_info['memory_percent']:.1f}%)")

            return memory_info

        except ImportError:
            print(f"📊 psutil no disponible para monitoreo de memoria")
            return {}
        except Exception as e:
            print(f"⚠️  Error monitoreando memoria: {e}")
            return {}

    def validate_configuration_consistency(self):
        """🎯 Validación inteligente de configuración"""

        print(f"🔍 Validando consistencia de configuración...")

        # ✅ RELACIÓN ENTRE TIMEFRAME Y HORIZONTE
        timeframe_to_minutes = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15,
            '30m': 30, '1h': 60, '4h': 240, '1d': 1440
        }

        tf_minutes = timeframe_to_minutes.get(self.timeframe, 5)

        # ✅ VALIDACIÓN DE HORIZONTE
        # El horizonte debe ser al menos 1 período del timeframe
        min_horizon = tf_minutes
        # Pero no más de 100 períodos
        max_horizon = tf_minutes * 100

        original_horizon = self.prediction_horizon

        if self.prediction_horizon < min_horizon:
            print(f"⚠️  Horizonte muy corto para {self.timeframe}: {self.prediction_horizon} < {min_horizon}")
            print(f"   🔧 Ajustando horizonte a {min_horizon} minutos")
            self.prediction_horizon = min_horizon

        if self.prediction_horizon > max_horizon:
            print(f"⚠️  Horizonte muy largo para {self.timeframe}: {self.prediction_horizon} > {max_horizon}")
            print(f"   🔧 Ajustando horizonte a {max_horizon} minutos")
            self.prediction_horizon = max_horizon

        if original_horizon != self.prediction_horizon:
            print(f"✅ Horizonte ajustado: {original_horizon} → {self.prediction_horizon}")

        # ✅ VALIDACIÓN DE LOOKBACK
        # Lookback debe ser suficiente para calcular indicadores
        min_lookback = max(24, self.prediction_horizon * 2)
        original_lookback = self.lookback_window

        if self.lookback_window < min_lookback:
            print(f"⚠️  Lookback insuficiente: {self.lookback_window} < {min_lookback}")
            print(f"   🔧 Ajustando lookback a {min_lookback} períodos")
            self.lookback_window = min_lookback

        if original_lookback != self.lookback_window:
            print(f"✅ Lookback ajustado: {original_lookback} → {self.lookback_window}")

        # ✅ VALIDACIÓN DE DÍAS DE ENTRENAMIENTO
        # Calcular días mínimos basados en lookback y horizonte
        min_days = max(7, (self.lookback_window + self.prediction_horizon) // 1440 + 1)
        original_days = self.training_days

        if self.training_days < min_days:
            print(f"⚠️  Días de entrenamiento insuficientes: {self.training_days} < {min_days}")
            print(f"   🔧 Ajustando días a {min_days}")
            self.training_days = min_days

        if original_days != self.training_days:
            print(f"✅ Días ajustados: {original_days} → {self.training_days}")

        # ✅ VALIDACIÓN DE BATCH SIZE
        # Batch size debe ser apropiado para el tamaño de datos
        if self.config.batch_size not in [32, 64, 128]:
            print(f"⚠️  Batch size no estándar: {self.config.batch_size}")
            print(f"   🔧 Ajustando batch size a 64")
            self.config.batch_size = 64

        # ✅ VALIDACIÓN DE ÉPOCAS
        if self.config.epochs < 10:
            print(f"⚠️  Épocas muy pocas: {self.config.epochs} < 10")
            print(f"   🔧 Ajustando épocas a 50")
            self.config.epochs = 50
        elif self.config.epochs > 200:
            print(f"⚠️  Épocas muy altas: {self.config.epochs} > 200")
            print(f"   🔧 Ajustando épocas a 100")
            self.config.epochs = 100

        # ✅ VALIDACIÓN ESPECÍFICA PARA TIMEFRAMES
        if self.timeframe == '1m':
            # Para 1m, validaciones especiales
            if self.prediction_horizon > 30:
                print(f"⚠️  Para 1m, horizonte máximo recomendado es 30 minutos")
                print(f"   🔧 Ajustando horizonte a 30")
                self.prediction_horizon = 30

            if self.lookback_window < 48:
                print(f"⚠️  Para 1m, lookback mínimo recomendado es 48 períodos")
                print(f"   🔧 Ajustando lookback a 48")
                self.lookback_window = 48

        elif self.timeframe == '5m':
            # Para 5m, validaciones especiales
            if self.prediction_horizon > 60:
                print(f"⚠️  Para 5m, horizonte máximo recomendado es 60 minutos")
                print(f"   🔧 Ajustando horizonte a 60")
                self.prediction_horizon = 60

        # ✅ VALIDACIÓN DE MEMORIA
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / 1024 / 1024 / 1024

            # Estimar uso de memoria basado en configuración
            estimated_memory_gb = (self.lookback_window * len(self.pairs) * self.training_days) / 1000000

            if estimated_memory_gb > available_memory_gb * 0.8:
                print(f"⚠️  ADVERTENCIA: Uso estimado de memoria alto")
                print(f"   📊 Memoria disponible: {available_memory_gb:.1f} GB")
                print(f"   📊 Uso estimado: {estimated_memory_gb:.1f} GB")
                print(f"   💡 Considera reducir lookback_window o training_days")
        except ImportError:
            pass  # psutil no disponible

        print(f"✅ Validación de configuración completada")
        return True

    async def train_adaptive_model(self, symbol: str) -> bool:
        """🎯 Entrenar modelo con thresholds adaptativos - CON VALIDACIONES ESPECÍFICAS PARA 1M"""

        print(f"\n🎯 ENTRENANDO MODELO ADAPTATIVO PARA {symbol}")
        print(f"⏰ TIMEFRAME: {self.timeframe}")
        print(f"🔮 HORIZONTE: {self.prediction_horizon} minutos")
        print(f"📊 VENTANA: {self.lookback_window} períodos")
        print(f"🎯 FEATURE SET: {self.feature_set}")
        print("=" * 70)

        # ✅ NUEVO: MONITOREO DE MEMORIA INICIAL
        print(f"📊 Monitoreando memoria inicial...")
        initial_memory = self.monitor_memory_usage()

        # ✅ NUEVO: Diagnóstico de métricas comprehensivas
        self._run_metrics_diagnostics()

        # ✅ NUEVO: VALIDACIÓN DE CONFIGURACIÓN
        print(f"🔍 Validando configuración antes del entrenamiento...")
        self.validate_configuration_consistency()

        try:
            # ✅ VALIDACIÓN ESPECÍFICA PARA 1M
            if self.timeframe == '1m':
                print("⚠️  VALIDACIÓN ESPECIAL PARA TIMEFRAME 1M:")
                print("   - Verificando cantidad mínima de datos...")
                print("   - Validando calidad de features...")
                print("   - Comprobando compatibilidad del modelo...")

            # 1. Obtener datos - CONFIGURABLEABLES
            df = await self.get_real_market_data(symbol)

            # ✅ VALIDACIÓN CRÍTICA DE DATOS PARA 1M
            if self.timeframe == '1m':
                if len(df) < 1000:  # Mínimo 1000 velas para 1m
                    print(f"❌ ERROR: Datos insuficientes para 1m. Solo {len(df)} velas (mínimo 1000)")
                    return False
                print(f"✅ Datos 1m válidos: {len(df)} velas")

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
                            
                            # ✅ NUEVO: Visualizar distribución de regímenes
                            if PLOTTING_AVAILABLE:
                                model_name = f"{symbol.lower()}_{self.timeframe}_{self.prediction_horizon}h_{self.lookback_window}w_{self.feature_set}"
                                regime_plot_path = f'models/adaptive_{model_name}/regime_distribution.png'
                                try:
                                    os.makedirs(os.path.dirname(regime_plot_path), exist_ok=True)
                                    self.regime_selector.visualize_regime_distribution(df, symbol, regime_plot_path)
                                except Exception as e:
                                    print(f"⚠️  Error guardando gráfico de regímenes: {e}")
                        else:
                            print(f"⚠️  No se pudieron seleccionar datos equilibrados, usando datos originales")
                    else:
                        print(f"⚠️  No se pudieron detectar regímenes, usando datos originales")
                        
                except Exception as e:
                    print(f"⚠️  Error en selección de regímenes: {e}")
                    print(f"   🔄 Continuando con datos originales...")

            # ✅ NUEVO: SELECCIÓN POR PERÍODOS DE FECHAS ESPECÍFICOS
            if self.use_date_periods and self.date_periods:
                print(f"📅 Aplicando selección por períodos de fechas específicos...")
                
                try:
                    # Usar la nueva función de selección por períodos
                    df_periods = self.regime_selector.select_balanced_regime_data(
                        df, symbol, date_periods=self.date_periods
                    )
                    
                    if not df_periods.empty and len(df_periods) > 100:  # Mínimo 100 muestras
                        df = df_periods
                        print(f"✅ Datos seleccionados por períodos: {len(df)} muestras")
                        
                        # ✅ OPCIONAL: Visualizar distribución temporal
                        if PLOTTING_AVAILABLE:
                            try:
                                model_name = f"{symbol.lower()}_{self.timeframe}_{self.prediction_horizon}h_{self.lookback_window}w_{self.feature_set}"
                                periods_plot_path = f'models/adaptive_{model_name}/date_periods_distribution.png'
                                os.makedirs(os.path.dirname(periods_plot_path), exist_ok=True)
                                self._visualize_date_periods_distribution(df, symbol, periods_plot_path)
                            except Exception as e:
                                print(f"⚠️  Error guardando gráfico de períodos: {e}")
                    else:
                        print(f"⚠️  No se pudieron seleccionar datos por períodos, usando datos originales")
                        
                except Exception as e:
                    print(f"⚠️  Error en selección por períodos: {e}")
                    print(f"   🔄 Continuando con datos originales...")

            # 2. Calcular features
            print(f"🔄 Calculando features...")
            
            # ✅ NUEVO: Manejo especial para features 3M especializadas
            if self.feature_set == 'features_3m_specialized' and FEATURES_3M_AVAILABLE:
                print(f"🎯 Usando Features 3M especializadas...")
                try:
                    # Usar el motor de features 3M especializado
                    features = AdvancedFeaturesEngine3m.create_complete_feature_set(df, symbol)
                    if features is None or features.empty:
                        print(f"❌ Error calculando features 3M especializadas")
                        return False
                    print(f"✅ Features 3M especializadas calculadas: {features.shape}")
                except Exception as e:
                    print(f"❌ Error con features 3M especializadas: {e}")
                    print(f"🔄 Fallback a motor de features estándar...")
                    features = self.features_engine.calculate_features(df, feature_set='tcn_definitivo_v3')
            else:
                # Usar motor de features estándar
                features = self.features_engine.calculate_features(df, feature_set=self.feature_set)

            if features.empty:
                print(f"❌ Error calculando features")
                return False

            # ✅ NUEVO: DIAGNÓSTICO DE VALORES FALTANTES
            print(f"🔍 Ejecutando diagnóstico de valores faltantes...")
            missing_diagnosis = self.diagnose_missing_values(features, symbol)

            # ✅ VALIDACIÓN: Verificar si hay demasiados valores faltantes
            total_nan = features.isna().sum().sum()
            total_inf = np.isinf(features.select_dtypes(include=[np.number])).sum().sum()

            if total_nan > len(features) * len(features.columns) * 0.3:  # Más del 30% de valores faltantes
                print(f"⚠️  ADVERTENCIA: Muchos valores faltantes ({total_nan}) para {symbol}")
                print(f"   📊 Considerando usar método de manejo más agresivo...")

            if total_inf > 0:
                print(f"⚠️  ADVERTENCIA: Valores infinitos detectados ({total_inf}) para {symbol}")

            # ✅ VALIDACIÓN ESPECÍFICA DE FEATURES SEGÚN FEATURE SET
            feature_set_expected = {
                'tcn_definitivo_v3': 54,  # V3 optimizado corregido
                'tcn_definitivo_v3_enhanced': 62,   # 🚀 NUEVO: Enhanced (54 base + 8 bajistas)
                'tcn_definitivo_v3_volume_enhanced': 64,   # 🆕 NUEVO: Volume Enhanced (62 base + 2 volume)
                'tcn_definitivo': 88,      # V2 original
                'ultra_momentum_combined': 79,  # TCN V3 + Ultra momentum core
                'ultra_momentum': 25,      # Ultra momentum optimizado
                'optimized_crypto': 25,
                'ultra_optimized': 15,
                'features_3m_specialized': 50  # 🎯 NUEVO: Features 3M especializadas (~50 features)
            }
            
            # Validación especial para enhanced, volume_enhanced, ultra_momentum y features 3M
            if self.feature_set == 'tcn_definitivo_v3_enhanced':
                expected_features = 62  # 54 base + 8 bajistas
                print(f"🚀 Usando conjunto ENHANCED: {expected_features} features con detección bajista optimizada")
            elif self.feature_set == 'tcn_definitivo_v3_volume_enhanced':
                expected_features = 64  # 62 enhanced + 2 volume
                print(f"🆕 Usando conjunto VOLUME ENHANCED: {expected_features} features con análisis de volumen avanzado")
            elif self.feature_set == 'features_3m_specialized':
                expected_features = 50  # ~50 features especializadas para 3M
                print(f"🎯 Usando conjunto FEATURES 3M ESPECIALIZADAS: {expected_features} features optimizadas para timeframe 3M")
            elif 'ultra_momentum' in self.feature_set.lower():
                if self.feature_set == 'ultra_momentum':
                    expected_features = 25
                elif self.feature_set == 'ultra_momentum_combined':
                    expected_features = 79
                else:
                    expected_features = feature_set_expected.get(self.feature_set, 25)  # Default 25 para variants de ultra_momentum
            else:
                expected_features = feature_set_expected.get(self.feature_set, 88)
            
            actual_features = len(features.columns)
            
            if actual_features < expected_features * 0.8:  # 80% mínimo
                print(f"❌ ERROR: Features insuficientes para {self.feature_set}. {actual_features}/{expected_features}")
                return False
            print(f"✅ Features {self.feature_set} válidas: {actual_features}/{expected_features}")

            # 3. Crear etiquetas con thresholds adaptativos
            df_labeled = self.create_balanced_labels(df, features, symbol)
            
            # 🎯 NUEVO: Preparar retornos futuros para profit-aware loss (MANEJO ROBUSTO)
            future_returns_tensor = None
            future_returns_array = None
            if (hasattr(self, 'use_profit_aware_loss') and 
                self.use_profit_aware_loss and 
                PROFIT_AWARE_AVAILABLE):
                print(f"🎯 Preparando retornos futuros para Profit-Aware Loss...")
                try:
                    future_returns_tensor = prepare_future_returns_tensor(df, self.prediction_horizon)
                    print(f"✅ Retornos futuros preparados exitosamente")
                except Exception as e:
                    print(f"⚠️ Error preparando retornos futuros: {e}")
                    print(f"🔄 Intentando método alternativo de preparación...")
                    try:
                        # Método alternativo manual
                        close_prices = df['close'].values
                        alt_returns = []
                        for i in range(len(close_prices) - self.prediction_horizon):
                            current_price = close_prices[i]
                            future_price = close_prices[i + self.prediction_horizon]
                            if current_price > 0:
                                alt_return = (future_price - current_price) / current_price
                                alt_returns.append(alt_return)
                            else:
                                alt_returns.append(0.0)
                        
                        import tensorflow as tf
                        future_returns_tensor = tf.constant(alt_returns[:len(df_labeled)], dtype=tf.float32)
                        print(f"✅ Método alternativo exitoso: {len(alt_returns)} retornos calculados")
                    except Exception as e2:
                        print(f"⚠️ Método alternativo también falló: {e2}")
                        print(f"⚠️ ADVERTENCIA: Profit-aware loss no se usará para este símbolo")
                        print(f"🔄 El modelo se entrenará con loss estándar (menos optimizado para rentabilidad)")
                        # NO desactivar globalmente: self.use_profit_aware_loss = False
                        future_returns_tensor = None
                
                # También como array para evaluación posterior
                close_prices = df['close'].values
                future_returns_list = []
                for i in range(len(close_prices) - self.prediction_horizon):
                    current_price = close_prices[i]
                    future_price = close_prices[i + self.prediction_horizon]
                    future_return = (future_price - current_price) / current_price
                    future_returns_list.append(future_return)
                
                # Ajustar longitud para coincidir con las etiquetas
                label_length = len(df_labeled)
                if len(future_returns_list) > label_length:
                    future_returns_array = np.array(future_returns_list[:label_length])
                else:
                    # Padding si es necesario
                    padding = [0.0] * (label_length - len(future_returns_list))
                    future_returns_array = np.array(future_returns_list + padding)
                
                print(f"✅ Retornos futuros preparados: {len(future_returns_array)} muestras")

            # 4. Preparar datos
            X, y, scaler, feature_columns, class_weights = self.prepare_training_data(df_labeled, features)

            # ✅ VALIDACIÓN CRÍTICA: Verificar que los datos se prepararon correctamente
            if X is None or y is None or scaler is None:
                print(f"❌ ERROR: No se pudieron preparar los datos para {symbol}")
                return False

            if len(X) == 0 or len(y) == 0:
                print(f"❌ ERROR: Datos vacíos para {symbol}")
                return False

            # ✅ VALIDACIÓN ESPECÍFICA PARA 1M: Verificar suficientes muestras
            if self.timeframe == '1m':
                min_samples = 500  # Mínimo 500 muestras para 1m
                if len(X) < min_samples:
                    print(f"❌ ERROR: Muestras insuficientes para 1m. Solo {len(X)} (mínimo {min_samples})")
                    return False
                print(f"✅ Muestras 1m válidas: {len(X)}")

            # 5. Split (con future_returns si disponible)
            if future_returns_array is not None:
                # Ajustar future_returns a la longitud de X e y
                if len(future_returns_array) > len(X):
                    future_returns_array = future_returns_array[:len(X)]
                elif len(future_returns_array) < len(X):
                    padding = np.zeros(len(X) - len(future_returns_array))
                    future_returns_array = np.concatenate([future_returns_array, padding])
                
                X_train, X_test, y_train, y_test, returns_train, returns_test = train_test_split(
                    X, y, future_returns_array, test_size=0.2, random_state=42, stratify=y
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                returns_train = returns_test = None

            # 6. Crear modelo
            model = self.create_definitive_tcn_model((X.shape[1], X.shape[2]))
            
            # 🎯 NUEVO: Recompilar con profit-aware loss si está habilitado (MANEJO ROBUSTO)
            profit_loss_enabled = False
            if (hasattr(self, 'use_profit_aware_loss') and 
                self.use_profit_aware_loss and 
                PROFIT_AWARE_AVAILABLE and 
                returns_train is not None):
                print(f"🎯 Recompilando modelo con Profit-Aware Loss: {getattr(self, 'loss_type', 'combined')}")
                
                try:
                    import tensorflow as tf
                    # Validar que returns_train tiene datos válidos
                    valid_returns = returns_train[~np.isnan(returns_train)]
                    if len(valid_returns) < len(returns_train) * 0.8:
                        print(f"⚠️ Demasiados valores NaN en returns_train ({len(valid_returns)}/{len(returns_train)})")
                        raise ValueError("Datos de retornos insuficientes")
                    
                    # Crear loss function específica
                    profit_loss_fn = create_profit_aware_loss(
                        tf.constant(returns_train, dtype=tf.float32),
                        loss_type=getattr(self, 'loss_type', 'combined'),
                        base_fee=getattr(self, 'base_fee', 0.001),
                        spread_cost=getattr(self, 'spread_cost', 0.0005),
                        profit_amplifier=getattr(self, 'profit_amplifier', 2.0),
                        loss_amplifier=getattr(self, 'loss_amplifier', 1.5)
                    )
                    
                    # Recompilar modelo con nueva loss
                    model.compile(
                        optimizer=model.optimizer,
                        loss=profit_loss_fn,
                        metrics=['accuracy']
                    )
                    profit_loss_enabled = True
                    print("✅ Modelo recompilado con profit-aware loss exitosamente")
                    
                except Exception as e:
                    print(f"⚠️ Error recompilando con profit-aware loss: {e}")
                    print("🔄 Continuando con loss estándar para este símbolo")
                    print("💡 Sugerencia: Verifica que profit_aware_loss.py esté disponible y actualizado")
                    # Mantener modelo con loss estándar
                    model.compile(
                        optimizer=model.optimizer,
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']
                    )

                    # ✅ ELIMINADO: La función de validación de dimensiones dinámicas es innecesaria
        # ya que el modelo ahora acepta secuencias de cualquier longitud.
        # print(f"🔍 Validando arquitectura del modelo...")
        # if not self.validate_dynamic_dimensions(model):
        #     print(f"❌ ERROR: Validación de dimensiones dinámicas falló para {symbol}")
        #     return False
        # print(f"✅ Arquitectura del modelo validada correctamente")

            # ✅ NOMBRE DEL MODELO CON TIMEFRAME Y CONFIGURACIÓN
            model_name = f"{symbol.lower()}_{self.timeframe}_{self.prediction_horizon}h_{self.lookback_window}w_{self.feature_set}"
            model_dir = f'models/adaptive_{model_name}'

            # ✅ VALIDACIÓN DE DIRECTORIO ANTES DE ENTRENAR
            try:
                os.makedirs(model_dir, exist_ok=True)
                print(f"✅ Directorio creado: {model_dir}")
            except Exception as dir_error:
                print(f"❌ ERROR creando directorio: {dir_error}")
                return False

            # ✅ CALLBACKS ANTI-OVERFITTING
            callbacks = self.create_callbacks(model_dir)

            print(f"🚀 Entrenando modelo: {model_name}")
            print(f"📊 Datos: {len(X_train)} train, {len(X_test)} test")

            # ✅ ENTRENAMIENTO CON MANEJO DE ERRORES MEJORADO
            try:
                # ✅ ENTRENAMIENTO CONFIGURABLE
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=self.config.epochs,  # ✅ CONFIGURABLE
                    batch_size=self.config.batch_size,  # ✅ CONFIGURABLE
                    callbacks=callbacks,
                    class_weight=class_weights,
                    verbose=1
                )

                # 7. Evaluar con métricas de trading avanzadas
                print(f"📊 Evaluando modelo con métricas de trading...")

                # Evaluación básica de Keras
                evaluation_results = model.evaluate(X_test, y_test, verbose=0)

                # ✅ CORRECCIÓN: Manejar múltiples métricas devueltas
                if isinstance(evaluation_results, list):
                    test_loss = evaluation_results[0]
                    test_acc = evaluation_results[1]  # accuracy principal
                else:
                    test_loss = evaluation_results
                    test_acc = 0.5  # fallback

                # 🎯 NUEVO: Usar métricas mejoradas si están habilitadas
                if (hasattr(self, 'use_enhanced_metrics') and 
                    self.use_enhanced_metrics and 
                    ENHANCED_METRICS_AVAILABLE and 
                    returns_test is not None):
                    print(f"📊 Evaluando con métricas mejoradas profit-aware...")
                    try:
                        enhanced_metrics = evaluate_model_with_enhanced_metrics(
                            model, X_test, y_test, returns_test, symbol=symbol
                        )
                        # Mantener compatibilidad con el código existente
                        trading_metrics = self.evaluate_model_with_trading_metrics(model, X_test, y_test, symbol)
                        trading_metrics['enhanced_metrics'] = enhanced_metrics
                    except Exception as e:
                        print(f"⚠️ Error en métricas mejoradas: {e}")
                        trading_metrics = self.evaluate_model_with_trading_metrics(model, X_test, y_test, symbol)
                else:
                    trading_metrics = self.evaluate_model_with_trading_metrics(model, X_test, y_test, symbol)

                # ✅ VALIDACIÓN ESPECÍFICA PARA 1M: Verificar calidad del entrenamiento
                if self.timeframe == '1m':
                    if test_acc < 0.4:  # Mínimo 40% accuracy para 1m
                        print(f"⚠️  WARNING: Accuracy baja para 1m ({test_acc:.3f} < 0.4)")
                    else:
                        print(f"✅ Accuracy 1m aceptable: {test_acc:.3f}")

                    # ✅ NUEVO: Validación de métricas de trading para 1m
                    buy_precision = trading_metrics['precision_per_class']['BUY']
                    sell_precision = trading_metrics['precision_per_class']['SELL']

                    if buy_precision < 0.35 or sell_precision < 0.35:
                        print(f"⚠️  WARNING: Precisión de señales baja para 1m (BUY:{buy_precision:.3f}, SELL:{sell_precision:.3f})")

                    # Validar confianza si está disponible
                    if 'avg_confidence_correct' in trading_metrics:
                        conf_correct = trading_metrics['avg_confidence_correct']
                        if conf_correct < 0.6:
                            print(f"⚠️  WARNING: Confianza baja para predicciones correctas ({conf_correct:.3f})")

                # ✅ NUEVO: Verificar que el entrenamiento fue exitoso con métricas de trading
                if (np.isnan(test_loss) or test_acc < 0.3 or
                    trading_metrics['f1_per_class']['BUY'] < 0.25 or
                    trading_metrics['f1_per_class']['SELL'] < 0.25):
                    print(f"⚠️  WARNING: Entrenamiento de {symbol} posiblemente problemático")
                    print(f"   📊 Métricas: Loss={test_loss:.4f}, Acc={test_acc:.3f}")
                    print(f"   📊 Trading: BUY-F1={trading_metrics['f1_per_class']['BUY']:.3f}, SELL-F1={trading_metrics['f1_per_class']['SELL']:.3f}")

            except Exception as train_error:
                print(f"❌ ERROR durante entrenamiento de {symbol}: {train_error}")
                # ✅ CORRECCIÓN: Limpiar memoria en caso de error
                self.cleanup_memory()
                return False

            # ✅ CORRECCIÓN: Limpiar memoria después del entrenamiento exitoso
            print(f"🧹 Limpiando memoria después del entrenamiento...")
            self.cleanup_memory()

            # ✅ VALIDACIÓN ANTES DE GUARDAR ARCHIVOS
            print(f"💾 Guardando archivos del modelo...")

            # 8. Guardar componentes CON VALIDACIÓN
            try:
                model.save(f'{model_dir}/model.h5')
                print(f"✅ Modelo guardado: {model_dir}/model.h5")

                with open(f'{model_dir}/scaler.pkl', 'wb') as f:
                    pickle.dump(scaler, f)
                print(f"✅ Scaler guardado: {model_dir}/scaler.pkl")
                
                with open(f'{model_dir}/feature_columns.pkl', 'wb') as f:
                    pickle.dump(feature_columns, f)
                print(f"✅ Feature columns guardado: {model_dir}/feature_columns.pkl")

                # ✅ NUEVO: Guardar configuración del modelo con métricas de trading
                config_info = {
                    'symbol': symbol,
                    'timeframe': self.timeframe,
                    'prediction_horizon': self.prediction_horizon,
                    'lookback_window': self.lookback_window,
                    'training_days': self.training_days,
                    'epochs': self.config.epochs,
                    'batch_size': self.config.batch_size,
                    'feature_set': self.config.feature_set,
                    'basic_metrics': {
                        'accuracy': float(test_acc),
                        'loss': float(test_loss)
                    },
                    'trading_metrics': trading_metrics,
                    'created_at': datetime.now().isoformat()
                }

                # ✅ CORRECCIÓN: Convertir tipos numpy antes de guardar JSON
                def convert_numpy_types_for_config(obj):
                    """🔄 Convertir tipos numpy a tipos nativos de Python para config.json"""
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, dict):
                        return {k: convert_numpy_types_for_config(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy_types_for_config(item) for item in obj]
                    else:
                        return obj

                # Convertir config_info para JSON
                config_info_json = convert_numpy_types_for_config(config_info)

                with open(f'{model_dir}/config.json', 'w') as f:
                    import json
                    json.dump(config_info_json, f, indent=2)
                print(f"✅ Config guardado: {model_dir}/config.json")

                # ✅ VALIDACIÓN FINAL DE ARCHIVOS
                required_files = ['model.h5', 'scaler.pkl', 'feature_columns.pkl', 'config.json']
                missing_files = []
                for file in required_files:
                    if not os.path.exists(f'{model_dir}/{file}'):
                        missing_files.append(file)

                if missing_files:
                    print(f"❌ ERROR: Archivos faltantes: {missing_files}")
                    return False
                else:
                    print(f"✅ Todos los archivos guardados correctamente")

                # ✅ NUEVO: Evaluación adicional con métricas de trading
                print(f"📊 Evaluando modelo con métricas de trading...")
                try:
                    trading_metrics = self.evaluate_model_with_trading_metrics(model, X_test, y_test, symbol)
                    print(f"✅ Evaluación de trading completada")
                except Exception as e:
                    print(f"⚠️  Error en evaluación de trading: {e}")

            except Exception as save_error:
                print(f"❌ ERROR guardando archivos: {save_error}")
                return False

            print(f"✅ Modelo guardado: {model_dir}")
            print(f"   📊 Configuración: {symbol} | {self.timeframe} | {self.prediction_horizon}h | {self.lookback_window}w")
            print(f"   🎯 Accuracy: {test_acc:.3f}")

            # ✅ RESUMEN FINAL ESPECÍFICO PARA 1M
            if self.timeframe == '1m':
                print(f"🎯 RESUMEN MODELO 1M:")
                print(f"   ✅ Datos: {len(df)} velas")
                print(f"   ✅ Features: {len(feature_columns)} columnas")
                print(f"   ✅ Muestras: {len(X)} total")
                print(f"   ✅ Accuracy: {test_acc:.3f}")
                print(f"   ✅ Archivos: {len(required_files)} guardados")

            return True

        except Exception as e:
            print(f"❌ Error: {e}")
            return False

    def _test_comprehensive_metrics(self) -> bool:
        """🧪 Probar que las métricas comprehensivas funcionan correctamente"""

        print("🧪 Probando métricas comprehensivas...")

        try:
            # Crear modelo de prueba
            input_shape = (48, 88)  # Formato estándar
            test_model = self.create_definitive_tcn_model(input_shape)

            # Generar datos de prueba
            X_test = np.random.randn(100, 48, 88)
            y_test = np.random.randint(0, 3, 100)

            # Evaluar modelo
            evaluation_results = test_model.evaluate(X_test, y_test, verbose=0)

            # Verificar que devuelve todas las métricas esperadas
            expected_metrics = 8  # loss + 7 métricas
            if len(evaluation_results) == expected_metrics:
                print(f"✅ Métricas comprehensivas: PASADO")
                print(f"   📊 Métricas devueltas: {len(evaluation_results)}")
                print(f"   📊 Loss: {evaluation_results[0]:.4f}")
                print(f"   📊 Accuracy: {evaluation_results[1]:.3f}")
                print(f"   📊 Sparse Categorical Accuracy: {evaluation_results[2]:.3f}")
                print(f"   📊 Top-2 Accuracy: {evaluation_results[4]:.3f}")
                print(f"   📊 Precisión: {evaluation_results[5]:.3f}")
                print(f"   📊 Recall: {evaluation_results[6]:.3f}")
                print(f"   📊 AUC: {evaluation_results[7]:.3f}")
                return True
            else:
                print(f"❌ Métricas comprehensivas: FALLÓ")
                print(f"   📊 Métricas esperadas: {expected_metrics}")
                print(f"   📊 Métricas devueltas: {len(evaluation_results)}")
                return False

        except Exception as e:
            print(f"❌ Métricas comprehensivas test: ERROR - {e}")
            return False

    def _run_metrics_diagnostics(self) -> None:
        """🔍 Diagnóstico de métricas comprehensivas"""

        print("🔍 DIAGNÓSTICO DE MÉTRICAS COMPREHENSIVAS")
        print("=" * 50)

        # Test de métricas comprehensivas
        metrics_safe = self._test_comprehensive_metrics()
        if not metrics_safe:
            print("🚨 ADVERTENCIA: Problemas detectados con métricas comprehensivas")
        else:
            print("✅ Métricas comprehensivas funcionando correctamente")

        print()

    def validate_training_requirements(self, symbol: str) -> bool:
        """🎯 Validar requisitos antes de entrenar - EVITA PÉRDIDA DE TIEMPO"""

        print(f"🔍 VALIDANDO REQUISITOS PARA {symbol} ({self.timeframe})...")

        # ✅ VALIDACIÓN 1: Verificar que el directorio models existe
        if not os.path.exists('models'):
            try:
                os.makedirs('models', exist_ok=True)
                print("✅ Directorio 'models' creado")
            except Exception as e:
                print(f"❌ ERROR: No se puede crear directorio 'models': {e}")
                return False

        # ✅ VALIDACIÓN 2: Verificar configuración específica para 1m
        if self.timeframe == '1m':
            print("⚠️  VALIDACIONES ESPECÍFICAS PARA 1M:")

            # Verificar que tenemos suficientes días de datos
            if self.training_days < 7:
                print(f"❌ ERROR: Para 1m necesitas al menos 7 días de datos (tienes {self.training_days})")
                return False

            # Verificar que el horizonte de predicción es razonable
            if self.prediction_horizon > 30:
                print(f"❌ ERROR: Para 1m el horizonte máximo es 30 minutos (tienes {self.prediction_horizon})")
                return False

            # Verificar que la ventana de lookback es apropiada
            if self.lookback_window < 24:
                print(f"❌ ERROR: Para 1m la ventana mínima es 24 períodos (tienes {self.lookback_window})")
                return False

            print("✅ Configuración 1m válida")

        # ✅ VALIDACIÓN 3: Verificar que el símbolo es válido
        valid_symbols = ['BTCUSDT', 'ETHUSDT', 'DOTUSDT', 'XRPUSDT', 'BNBUSDT', 'ADAUSDT']
        if symbol not in valid_symbols:
            print(f"❌ ERROR: Símbolo {symbol} no está en la lista de válidos: {valid_symbols}")
            return False

        # ✅ VALIDACIÓN 4: Verificar que el timeframe es válido
        valid_timeframes = ['1m', '3m', '5m']
        if self.timeframe not in valid_timeframes:
            print(f"❌ ERROR: Timeframe {self.timeframe} no válido. Opciones: {valid_timeframes}")
            return False

        # ✅ VALIDACIÓN 5: Verificar parámetros de entrenamiento
        if self.config.epochs < 10 or self.config.epochs > 200:
            print(f"❌ ERROR: Épocas debe estar entre 10 y 200 (tienes {self.config.epochs})")
            return False

        if self.config.batch_size not in [32, 64, 128]:
            print(f"❌ ERROR: Batch size debe ser 32, 64 o 128 (tienes {self.config.batch_size})")
            return False

        print("✅ Todos los requisitos cumplidos")
        return True


def parse_custom_periods(custom_periods_str: str) -> List[Dict]:
    """
    🎯 Parsear períodos personalizados desde string de línea de comandos
    
    Formato esperado: "2023-01-01:2023-03-31,2023-04-01:2023-06-30"
    
    Returns:
        Lista de diccionarios con períodos de fechas
    """
    try:
        periods = []
        
        if not custom_periods_str or custom_periods_str.strip() == "":
            print("⚠️ String de períodos personalizados vacío")
            return []
        
        # Dividir por comas para obtener múltiples períodos
        period_strings = custom_periods_str.split(',')
        
        for i, period_str in enumerate(period_strings):
            period_str = period_str.strip()
            
            if ':' not in period_str:
                print(f"❌ Formato inválido en período {i+1}: {period_str}")
                print(f"   Formato esperado: 'YYYY-MM-DD:YYYY-MM-DD'")
                continue
            
            try:
                start_str, end_str = period_str.split(':')
                start_str = start_str.strip()
                end_str = end_str.strip()
                
                # Validar formato de fechas
                from datetime import datetime
                start_date = datetime.strptime(start_str, '%Y-%m-%d')
                end_date = datetime.strptime(end_str, '%Y-%m-%d')
                
                # Validar que start < end
                if start_date >= end_date:
                    print(f"❌ Fecha de inicio debe ser anterior a fecha de fin en período {i+1}")
                    continue
                
                # Validar que las fechas no sean futuras
                from datetime import datetime
                today = datetime.now()
                if start_date > today or end_date > today:
                    print(f"⚠️ Período {i+1} contiene fechas futuras, se usará hasta hoy")
                    if end_date > today:
                        end_date = today
                        end_str = today.strftime('%Y-%m-%d')
                
                period = {
                    'start_date': start_str,
                    'end_date': end_str,
                    'description': f'Período personalizado {i+1} ({start_str} a {end_str})'
                }
                
                periods.append(period)
                print(f"✅ Período {i+1} procesado: {start_str} → {end_str}")
                
            except ValueError as e:
                print(f"❌ Error procesando período {i+1} '{period_str}': {e}")
                print(f"   Asegúrate de usar formato YYYY-MM-DD")
                continue
        
        if not periods:
            print("❌ No se pudo procesar ningún período válido")
            return []
        
        print(f"✅ Total de períodos personalizados procesados: {len(periods)}")
        return periods
        
    except Exception as e:
        print(f"❌ Error general procesando períodos personalizados: {e}")
        return []

async def main():
    """🎯 Entrenar modelos con configuración INTERACTIVA - CON VALIDACIÓN PREVIA"""
    
    print("🎯 ENTRENADOR TCN ADAPTATIVO - CONFIGURACIÓN INTERACTIVA")
    print("=" * 70)
    print("🎯 Te voy a preguntar paso a paso qué quieres entrenar")
    print("=" * 70)
    
    # ✅ CONFIGURACIÓN INTERACTIVA
    config = configurar_interactivamente()

    # ✅ CONFIRMACIÓN FINAL
    print(f"\n" + "="*60)
    print(f"📋 RESUMEN DE TU CONFIGURACIÓN:")
    config.print_config()
    print(f"="*60)

    respuesta = input(f"\n👉 ¿Todo correcto? ¿Empezar entrenamiento? [s/N]: ").strip().lower()
    if respuesta not in ['s', 'y', 'yes', 'si', 'sí']:
        print("❌ Entrenamiento cancelado. ¡Hasta luego!")
        return

    # ✅ CREAR TRAINER Y VALIDAR ANTES DE ENTRENAR
    trainer = AdaptiveTCNTrainer(config)

    # ✅ NUEVO: VALIDACIÓN DE CONFIGURACIÓN ANTES DE ENTRENAR
    print(f"🔍 Validando configuración del trainer...")
    trainer.validate_configuration_consistency()

    print(f"\n🚀 INICIANDO ENTRENAMIENTO...")
    print(f"📊 Pares: {', '.join(trainer.pairs)}")
    print(f"⏰ Timeframe: {config.timeframe}")
    print(f"🔮 Horizonte: {config.prediction_horizon} minutos")
    print(f"📊 Ventana: {config.lookback_window} períodos")
    print(f"📅 Datos: {config.training_days} días")
    print(f"🎯 Épocas: {config.epochs}")
    print(f"🎯 Feature Set: {config.feature_set}")
    if config.use_balanced_regimes:
        print(f"⚖️  Regímenes equilibrados: {config.regime_balance_method}")
    if config.use_date_periods:
        print(f"📅 Períodos de fechas: {config.date_periods_method}")
        if config.preset_periods:
            print(f"📅 Períodos predefinidos: {config.preset_periods}")
    if config.use_efficient_tcn_v3:
        print(f"⚡ Arquitectura TCN V3 Eficiente:")
        print(f"   📊 Filtros: {config.tcn_v3_filters}")
        print(f"   🎯 Dilataciones: {config.tcn_v3_dilations}")
        print(f"   🚀 Entrenamiento estimado: 2-3x más rápido")
    print("=" * 70)
    
    results = {}
    for symbol in trainer.pairs:
        print(f"\n🔥 Entrenando {symbol}...")

        # ✅ VALIDACIÓN PREVIA PARA EVITAR PÉRDIDA DE TIEMPO
        if not trainer.validate_training_requirements(symbol):
            print(f"❌ VALIDACIÓN FALLIDA para {symbol}. Saltando...")
            results[symbol] = False
            continue

        # ✅ ENTRENAMIENTO CON VALIDACIONES ESPECÍFICAS
        success = await trainer.train_adaptive_model(symbol)
        results[symbol] = success
    
    print(f"\n🎯 RESUMEN FINAL:")
    print("=" * 40)
    for symbol, success in results.items():
        status = "✅ ÉXITO" if success else "❌ FALLO"
        print(f"   {symbol}: {status}")
    
    successful = sum(results.values())
    print(f"\n🏆 Modelos entrenados exitosamente: {successful}/{len(results)}")

    if successful > 0:
        print(f"📁 Modelos guardados en: models/adaptive_<symbol>_<timeframe>_<config>/")
        print(f"🎯 Cada modelo incluye:")
        print(f"   - model.h5 (modelo entrenado)")
        print(f"   - best_model.h5 (mejor modelo)")
        print(f"   - scaler.pkl (escalador)")
        print(f"   - feature_columns.pkl (columnas)")
        print(f"   - config.json (configuración completa)")
        print(f"🎯 ¡Listo para usar en trading!")
    else:
        print(f"❌ No se pudo entrenar ningún modelo. Revisa los errores arriba.")


def configurar_interactivamente() -> TrainingConfig:
    """🎯 Configuración INTERACTIVA - El usuario elige todo paso a paso"""

    print("🎯 CONFIGURACIÓN INTERACTIVA DE ENTRENAMIENTO")
    print("=" * 60)
    print("Te voy a preguntar paso a paso qué quieres entrenar...")
    print("=" * 60)

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
        ('tcn_definitivo_v3', '54 features (V3 optimizado) ⭐ RECOMENDADO'),
        ('tcn_definitivo_v3_enhanced', '62 features (V3 + 8 bajistas) 🚀 NUEVO ENHANCED'),
        ('tcn_definitivo_v3_volume_enhanced', '64 features (V3 + bajistas + volumen) 🆕 NUEVO VOLUME'),
        ('tcn_definitivo', '88 features (V2 completo)'),
        ('ultra_momentum_combined', '79 features (V3 + Ultra-momentum core) ⚡ NUEVO'),
        ('ultra_momentum', '25 features (Ultra-momentum optimizado) ⚡ NUEVO'),
        ('optimized_crypto', '25 features (optimizado)'),
        ('ultra_optimized', '15 features (ultra optimizado)')
    ]
    
    # ✅ NUEVO: Agregar features 3M especializadas si están disponibles
    if FEATURES_3M_AVAILABLE:
        feature_sets.append(('features_3m_specialized', '~50 features (3M especializadas) 🎯 NUEVO 3M'))
        print("✅ Features 3M especializadas disponibles para selección")
    for i, (fs, desc) in enumerate(feature_sets, 1):
        print(f"  {i}. {fs} - {desc}")

    # ✅ NUEVO: Determinar el rango máximo de opciones
    max_options = len(feature_sets)
    option_range = f"1-{max_options}"

    while True:
        respuesta = input(f"👉 Elige feature set [{option_range}] (default: 1): ").strip()
        if respuesta == '' or respuesta == '1':
            config.feature_set = 'tcn_definitivo_v3'
            break
        elif respuesta == '2':
            config.feature_set = 'tcn_definitivo_v3_enhanced'
            break
        elif respuesta == '3':
            config.feature_set = 'tcn_definitivo_v3_volume_enhanced'
            break
        elif respuesta == '4':
            config.feature_set = 'tcn_definitivo'
            break
        elif respuesta == '5':
            config.feature_set = 'ultra_momentum_combined'
            break
        elif respuesta == '6':
            config.feature_set = 'ultra_momentum'
            break
        elif respuesta == '7':
            config.feature_set = 'optimized_crypto'
            break
        elif respuesta == '8':
            config.feature_set = 'ultra_optimized'
            break
        elif respuesta == '9' and FEATURES_3M_AVAILABLE:
            config.feature_set = 'features_3m_specialized'
            break
        else:
            print(f"❌ Opción inválida. Elige {option_range}")

    # 3️⃣ PARES
    print(f"\n💎 PASO 3: PARES DE TRADING")
    print(f"Pares disponibles:")
    pares_disponibles = ['BTCUSDT', 'ETHUSDT', 'DOTUSDT', 'XRPUSDT', 'BNBUSDT', 'ADAUSDT']
    for i, par in enumerate(pares_disponibles, 1):
        print(f"  {i}. {par}")

    config.pairs = []
    print(f"👉 Selecciona los pares que quieres entrenar (separados por comas):")
    print(f"    Ejemplo: 1,2,4 para BTC, ETH y XRP")

    while True:
        respuesta = input(f"Números [1-6] (default: 1): ").strip()
        if respuesta == '':
            config.pairs = ['BTCUSDT']
            break

        try:
            indices = [int(x.strip()) for x in respuesta.split(',')]
            pares_elegidos = []
            for idx in indices:
                if 1 <= idx <= 6:
                    pares_elegidos.append(pares_disponibles[idx-1])
                else:
                    raise ValueError()
            config.pairs = pares_elegidos
            break
        except:
            print("❌ Formato inválido. Usa números del 1-6 separados por comas")

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
    while True:
        respuesta = input(f"👉 ¿Cuántos días de datos usar? (default: 30): ").strip()
        if respuesta == '':
            config.training_days = 30
            break
        try:
            dias = int(respuesta)
            if 1 <= dias <= 365:
                config.training_days = dias
                break
            else:
                print("❌ Usa entre 1 y 365 días")
        except:
            print("❌ Ingresa un número válido")

    # 7️⃣ ÉPOCAS
    print(f"\n🎯 PASO 7: ÉPOCAS DE ENTRENAMIENTO")
    while True:
        respuesta = input(f"👉 ¿Cuántas épocas entrenar? (default: 50): ").strip()
        if respuesta == '':
            config.epochs = 50
            break
        try:
            epochs = int(respuesta)
            if 10 <= epochs <= 200:
                config.epochs = epochs
                break
            else:
                print("❌ Usa entre 10 y 200 épocas")
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

    # 🚀 NUEVO: PASO 9: ARQUITECTURA TCN
    print(f"\n🏗️ PASO 9: ARQUITECTURA TCN")
    print(f"¿Qué arquitectura TCN quieres usar?")
    architectures = [
        ('enhanced', 'Mejorada - Nueva arquitectura TCN con bloques residuales (RECOMENDADO)'),
        ('original', 'Original - Arquitectura crypto optimizada probada'),
        ('hybrid', 'Híbrida - Combinación de ambas arquitecturas'),
        ('efficient_v3', 'Eficiente V3 - Arquitectura rápida y ligera (2-3x más rápido)')
    ]
    
    # ✅ NUEVO: CONFIGURACIÓN ESPECIAL PARA 1M
    if config.timeframe == '1m':
        print(f"\n⚡ CONFIGURACIÓN ESPECIAL PARA TIMEFRAME 1M:")
        print(f"¿Quieres usar configuración optimizada para alta frecuencia?")
        print(f"  - Optimizada: Más rápida, menos parámetros (recomendado para 1m)")
        print(f"  - Completa: Máxima precisión, más parámetros")
        
        while True:
            optimizada = input(f"👉 ¿Usar configuración optimizada para 1m? [S/n] (default: S): ").strip().lower()
            if optimizada in ['', 's', 'si', 'sí', 'y', 'yes']:
                config.use_optimized_1m = True
                print(f"⚡ Configuración optimizada para 1m habilitada")
                break
            elif optimizada in ['n', 'no']:
                config.use_optimized_1m = False
                print(f"🚀 Configuración completa para 1m habilitada")
                break
            else:
                print("❌ Opción inválida. Elige S o n")
    
    for i, (arch, desc) in enumerate(architectures, 1):
        print(f"  {i}. {arch} - {desc}")

    while True:
        respuesta = input(f"👉 Elige arquitectura [1-4] (default: 1): ").strip()
        if respuesta == '' or respuesta == '1':
            config.tcn_architecture = 'enhanced'
            config.use_enhanced_tcn = True
            config.use_efficient_tcn_v3 = False
            break
        elif respuesta == '2':
            config.tcn_architecture = 'original'
            config.use_enhanced_tcn = False
            config.use_efficient_tcn_v3 = False
            break
        elif respuesta == '3':
            config.tcn_architecture = 'hybrid'
            config.use_enhanced_tcn = True
            config.use_efficient_tcn_v3 = False
            break
        elif respuesta == '4':
            config.tcn_architecture = 'efficient_v3'
            config.use_enhanced_tcn = False
            config.use_efficient_tcn_v3 = True
            print(f"⚡ ¡Excelente elección! TCN V3 Eficiente es 2-3x más rápido para entrenamiento")
            
            # ✅ CONFIGURACIÓN ADICIONAL PARA TCN V3 EFICIENTE
            print(f"\n⚡ CONFIGURACIÓN TCN V3 EFICIENTE")
            print(f"¿Quieres personalizar los parámetros de TCN V3?")
            
            while True:
                custom = input(f"👉 ¿Personalizar filtros y dilataciones? [s/N] (default: N): ").strip().lower()
                if custom in ['s', 'si', 'sí', 'y', 'yes']:
                    # Configurar filtros
                    while True:
                        filters = input(f"👉 Número de filtros por bloque [32-128] (default: 64): ").strip()
                        if filters == '':
                            config.tcn_v3_filters = 64
                            break
                        try:
                            f = int(filters)
                            if 32 <= f <= 128:
                                config.tcn_v3_filters = f
                                break
                            else:
                                print("❌ Usa entre 32 y 128 filtros")
                        except:
                            print("❌ Ingresa un número válido")
                    
                    # Configurar dilataciones
                    print(f"👉 Dilataciones disponibles:")
                    print(f"   Opción 1: [1, 2, 4] - Rápido (3 bloques)")
                    print(f"   Opción 2: [1, 2, 4, 8] - Balanceado (4 bloques) - RECOMENDADO")
                    print(f"   Opción 3: [1, 2, 4, 8, 16] - Completo (5 bloques)")
                    
                    while True:
                        dilation_choice = input(f"👉 Elige opción [1-3] (default: 2): ").strip()
                        if dilation_choice == '' or dilation_choice == '2':
                            config.tcn_v3_dilations = [1, 2, 4, 8]
                            break
                        elif dilation_choice == '1':
                            config.tcn_v3_dilations = [1, 2, 4]
                            break
                        elif dilation_choice == '3':
                            config.tcn_v3_dilations = [1, 2, 4, 8, 16]
                            break
                        else:
                            print("❌ Opción inválida. Elige 1, 2 o 3")
                    
                    print(f"⚡ Configuración TCN V3 personalizada:")
                    print(f"   📊 Filtros: {config.tcn_v3_filters}")
                    print(f"   🎯 Dilataciones: {config.tcn_v3_dilations}")
                    break
                else:
                    # Usar configuración por defecto
                    config.tcn_v3_filters = 64
                    config.tcn_v3_dilations = [1, 2, 4, 8]
                    print(f"⚡ Usando configuración por defecto:")
                    print(f"   📊 Filtros: 64")
                    print(f"   🎯 Dilataciones: [1, 2, 4, 8]")
                    break
            
            break
        else:
            print("❌ Opción inválida. Elige 1, 2, 3 o 4")

    # ✅ NUEVO: PASO 10: REGÍMENES DE MERCADO EQUILIBRADOS
    print(f"\n⚖️ PASO 10: REGÍMENES DE MERCADO EQUILIBRADOS")
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

    # ✅ NUEVO: PASO 10: PERÍODOS DE FECHAS ESPECÍFICOS
    print(f"\n📅 PASO 10: PERÍODOS DE FECHAS ESPECÍFICOS")
    print(f"¿Quieres entrenar con datos de períodos de fechas específicos?")
    print(f"Esto te permite elegir exactamente qué momentos del mercado usar para entrenamiento.")
    print(f"Útil para capturar mercados en diferentes regímenes (alcista, bajista, lateral).")
    
    while True:
        respuesta = input(f"👉 ¿Usar períodos de fechas específicos? [s/N] (default: N): ").strip().lower()
        if respuesta == '' or respuesta in ['n', 'no']:
            config.use_date_periods = False
            break
        elif respuesta in ['s', 'si', 'sí', 'y', 'yes']:
            config.use_date_periods = True
            break
        else:
            print("❌ Responde 's' para sí o 'N' para no")

    # ✅ CONFIGURACIÓN ADICIONAL SI SE SELECCIONAN PERÍODOS DE FECHAS
    if config.use_date_periods:
        print(f"\n📊 CONFIGURACIÓN DE PERÍODOS DE FECHAS")
        print(f"Métodos disponibles:")
        methods = [
            ('preset', 'Períodos predefinidos - Selección automática de regímenes'),
            ('custom', 'Períodos personalizados - Tú eliges las fechas exactas')
        ]
        
        for i, (method, desc) in enumerate(methods, 1):
            print(f"  {i}. {method} - {desc}")

        while True:
            respuesta = input(f"👉 Elige método [1-2] (default: 1): ").strip()
            if respuesta == '' or respuesta == '1':
                config.date_periods_method = 'preset'
                break
            elif respuesta == '2':
                config.date_periods_method = 'custom'
                break
            else:
                print("❌ Opción inválida. Elige 1 o 2")

        # ✅ CONFIGURACIÓN DE PERÍODOS PREDEFINIDOS
        if config.date_periods_method == 'preset':
            print(f"\n📅 PERÍODOS PREDEFINIDOS DISPONIBLES")
            print(f"Estos períodos están diseñados para capturar diferentes regímenes de mercado:")
            
            preset_options = [
                ('crypto_bull_bear_2021_2023', 'Bull 2021 + Bear 2022 + Recuperación 2023'),
                ('crypto_volatility_2020_2023', 'COVID + Bull + Bear + Mixto 2023'),
                ('btc_cycles_2017_2023', 'Ciclos completos de Bitcoin 2017-2023')
            ]
            
            for i, (preset, desc) in enumerate(preset_options, 1):
                print(f"  {i}. {preset}")
                print(f"     {desc}")
                print()

            while True:
                respuesta = input(f"👉 Elige período predefinido [1-3] (default: 1): ").strip()
                if respuesta == '' or respuesta == '1':
                    config.preset_periods = 'crypto_bull_bear_2021_2023'
                    break
                elif respuesta == '2':
                    config.preset_periods = 'crypto_volatility_2020_2023'
                    break
                elif respuesta == '3':
                    config.preset_periods = 'btc_cycles_2017_2023'
                    break
                else:
                    print("❌ Opción inválida. Elige 1, 2 o 3")
            
            # Cargar los períodos predefinidos
            config.date_periods = config.preset_date_periods[config.preset_periods]
            print(f"✅ Períodos cargados: {len(config.date_periods)} períodos")

        # ✅ CONFIGURACIÓN DE PERÍODOS PERSONALIZADOS
        elif config.date_periods_method == 'custom':
            print(f"\n📅 PERÍODOS PERSONALIZADOS")
            print(f"Formato: YYYY-MM-DD:YYYY-MM-DD,YYYY-MM-DD:YYYY-MM-DD")
            print(f"Ejemplo: 2023-01-01:2023-03-31,2023-04-01:2023-06-30")
            print(f"Esto seleccionará:")
            print(f"  - Período 1: 1 de enero a 31 de marzo de 2023")
            print(f"  - Período 2: 1 de abril a 30 de junio de 2023")
            
            while True:
                respuesta = input(f"👉 Ingresa tus períodos personalizados: ").strip()
                if respuesta == '':
                    print("❌ Debes ingresar al menos un período")
                    continue
                
                try:
                    # Procesar períodos personalizados
                    periods = []
                    for period_str in respuesta.split(','):
                        start, end = period_str.split(':')
                        periods.append({
                            'start_date': start.strip(),
                            'end_date': end.strip(),
                            'description': f'Período personalizado {start.strip()} a {end.strip()}'
                        })
                    
                    if len(periods) > 0:
                        config.date_periods = periods
                        config.date_periods_method = 'custom'
                        print(f"✅ {len(periods)} períodos personalizados configurados")
                        break
                    else:
                        print("❌ No se pudieron procesar los períodos")
                        
                except Exception as e:
                    print(f"❌ Error en formato: {e}")
                    print(f"   📋 Usa el formato: 2023-01-01:2023-03-31,2023-04-01:2023-06-30")

    # 🎯 NUEVO: PASO 11: PROFIT-AWARE LOSS
    print(f"\n🎯 PASO 11: PROFIT-AWARE LOSS")
    print(f"¿Quieres usar funciones de pérdida conscientes de rentabilidad?")
    print(f"Esto enseña al modelo a maximizar la rentabilidad real, considerando fees y spreads.")
    print(f"📈 BENEFICIOS:")
    print(f"   • Reduce overtrading y trades no rentables")
    print(f"   • Optimiza para métricas financieras reales (Sharpe, Calmar)")
    print(f"   • Mejora consistencia en diferentes condiciones de mercado")
    
    while True:
        respuesta = input(f"👉 ¿Usar Profit-Aware Loss? [S/n] (default: S): ").strip().lower()
        if respuesta == '' or respuesta in ['s', 'si', 'sí', 'y', 'yes']:
            config.use_profit_aware_loss = True
            break
        elif respuesta in ['n', 'no']:
            config.use_profit_aware_loss = False
            break
        else:
            print("❌ Responde 'S' para sí o 'n' para no")

    # 🚀 NUEVO: CONFIGURACIÓN DE SISTEMA DE ETIQUETADO
    print(f"\n🚀 CONFIGURACIÓN DE SISTEMA DE ETIQUETADO")
    print(f"Selecciona el método de etiquetado:")
    print(f"1. 🚀 Percentiles Dinámicos (PROFESIONAL) - Usado por Renaissance Technologies")
    print(f"   • Garantiza distribución balanceada (15% BUY, 15% SELL, 70% HOLD)")
    print(f"   • Se adapta automáticamente a la volatilidad")
    print(f"   • No requiere ajustar thresholds manualmente")
    print(f"2. 🎯 Thresholds Fijos (TRADICIONAL)")
    print(f"   • Usa thresholds fijos por símbolo")
    print(f"   • Requiere ajuste manual")
    
    while True:
        respuesta = input(f"👉 ¿Qué sistema usar? [1-Percentiles/2-Thresholds] (default: 1): ").strip()
        if respuesta == '' or respuesta == '1':
            config.use_dynamic_percentiles = True
            print(f"✅ Seleccionado: Sistema de Percentiles Dinámicos")
            
            # Configuración de percentiles
            while True:
                respuesta = input(f"👉 ¿% para señales BUY? (default: 15): ").strip()
                if respuesta == '':
                    config.buy_percentile = 85  # Top 15%
                    break
                try:
                    buy_pct = float(respuesta)
                    if 5 <= buy_pct <= 25:
                        config.buy_percentile = 100 - buy_pct  # Si usuario quiere 15%, config.buy_percentile = 85
                        break
                    else:
                        print("❌ Usa entre 5% y 25%")
                except ValueError:
                    print("❌ Introduce un número válido")
            
            # ✅ CORRECCIÓN: sell_percentile debe ser simétrico al buy_percentile
            config.sell_percentile = 100 - config.buy_percentile  # Si buy_percentile = 85, sell_percentile = 15
            break
            
        elif respuesta == '2':
            config.use_dynamic_percentiles = False
            print(f"✅ Seleccionado: Sistema de Thresholds Fijos")
            break
        else:
            print("❌ Selecciona 1 o 2")

    # ✅ CONFIGURACIÓN ADICIONAL SI SE SELECCIONA PROFIT-AWARE LOSS
    if config.use_profit_aware_loss:
        print(f"\n💰 CONFIGURACIÓN DE PROFIT-AWARE LOSS")
        print(f"Tipos de funciones de pérdida disponibles:")
        loss_types = [
            ('combined', 'Combinada - Balance entre precisión y rentabilidad (RECOMENDADO)'),
            ('profit_weighted', 'Peso por beneficio - Enfoque máximo en rentabilidad'),
            ('sharpe_aware', 'Consciente de Sharpe - Optimiza ratio riesgo-retorno'),
            ('risk_adjusted', 'Ajustada por riesgo - Balance conservador')
        ]
        
        for i, (loss_type, desc) in enumerate(loss_types, 1):
            print(f"  {i}. {loss_type} - {desc}")

        while True:
            respuesta = input(f"👉 Elige tipo de loss [1-4] (default: 1): ").strip()
            if respuesta == '' or respuesta == '1':
                config.loss_type = 'combined'
                break
            elif respuesta == '2':
                config.loss_type = 'profit_weighted'
                break
            elif respuesta == '3':
                config.loss_type = 'sharpe_aware'
                break
            elif respuesta == '4':
                config.loss_type = 'risk_adjusted'
                break
            else:
                print("❌ Opción inválida. Elige 1, 2, 3 o 4")

        # Configuración de fees y costos
        print(f"\n💸 CONFIGURACIÓN DE COSTOS DE TRADING")
        while True:
            respuesta = input(f"👉 Fee por trade (%) (default: 0.1): ").strip()
            if respuesta == '':
                config.base_fee = 0.001  # 0.1%
                break
            try:
                fee_percent = float(respuesta)
                if 0.01 <= fee_percent <= 1.0:  # Entre 0.01% y 1%
                    config.base_fee = fee_percent / 100  # Convertir a decimal
                    break
                else:
                    print("❌ Usa un fee entre 0.01% y 1%")
            except:
                print("❌ Ingresa un número válido")

        while True:
            respuesta = input(f"👉 Spread cost (%) (default: 0.05): ").strip()
            if respuesta == '':
                config.spread_cost = 0.0005  # 0.05%
                break
            try:
                spread_percent = float(respuesta)
                if 0.01 <= spread_percent <= 0.5:  # Entre 0.01% y 0.5%
                    config.spread_cost = spread_percent / 100  # Convertir a decimal
                    break
                else:
                    print("❌ Usa un spread entre 0.01% y 0.5%")
            except:
                print("❌ Ingresa un número válido")

    return config


if __name__ == "__main__":
    import argparse

    # 🎯 CONFIGURAR ARGUMENTOS DE LÍNEA DE COMANDOS
    parser = argparse.ArgumentParser(description='🎯 Entrenador TCN Adaptativo con Feature Sets Optimizados')

    # Argumentos de feature sets
    feature_set_choices = ['tcn_definitivo_v3', 'tcn_definitivo_v3_enhanced', 'tcn_definitivo_v3_volume_enhanced', 'tcn_definitivo', 'optimized_crypto', 'ultra_optimized', 'ultra_momentum', 'ultra_momentum_combined']
    
    # ✅ NUEVO: Agregar features 3M especializadas si están disponibles
    if 'features3m' in globals() or FEATURES_3M_AVAILABLE:
        feature_set_choices.append('features_3m_specialized')
    
    parser.add_argument('--feature_set', type=str, default='tcn_definitivo_v3_enhanced',
                       choices=feature_set_choices,
                       help='Conjunto de features a usar (default: tcn_definitivo_v3_enhanced) 🚀 NUEVO: Enhanced con detección bajista 🆕 Volume Enhanced disponible 🎯 Features 3M especializadas disponibles')

    # ✅ NUEVO: Argumentos para regímenes equilibrados
    parser.add_argument('--balanced_regimes', action='store_true',
                       help='Usar selección de regímenes de mercado equilibrados')
    parser.add_argument('--regime_method', type=str,
                       choices=['auto', 'manual', 'stratified'],
                       help='Método de balance de regímenes (default: auto)')
    parser.add_argument('--samples_per_regime', type=int,
                       help='Número de muestras objetivo por régimen (para método manual)')
    
    # ✅ NUEVO: Argumentos para períodos de fechas específicos
    parser.add_argument('--use_date_periods', action='store_true',
                       help='Usar selección por períodos de fechas específicos')
    parser.add_argument('--date_periods_method', type=str,
                       choices=['manual', 'preset', 'custom'],
                       help='Método de selección de períodos (default: manual)')
    parser.add_argument('--preset_periods', type=str,
                       choices=['crypto_bull_bear_2021_2023', 'crypto_volatility_2020_2023', 'btc_cycles_2017_2023'],
                       help='Períodos predefinidos para diferentes regímenes de mercado')
    parser.add_argument('--custom_periods', type=str,
                       help='Períodos personalizados para entrenamiento. Formato: "YYYY-MM-DD:YYYY-MM-DD,YYYY-MM-DD:YYYY-MM-DD". Ejemplo: "2023-01-01:2023-03-31,2023-07-01:2023-09-30" para entrenar solo en Q1 y Q3 de 2023.')

    # Argumentos de configuración
    parser.add_argument('--timeframe', type=str, choices=['1m', '3m', '5m'],
                       help='Timeframe para entrenamiento')
    parser.add_argument('--pairs', nargs='+',
                       help='Pares de trading (ej: BTCUSDT ETHUSDT)')
    parser.add_argument('--prediction_horizon', type=int,
                       help='Horizonte de predicción en minutos')
    parser.add_argument('--lookback_window', type=int,
                       help='Ventana de análisis histórica')
    parser.add_argument('--training_days', type=int,
                       help='Días de datos para entrenamiento')
    parser.add_argument('--epochs', type=int,
                       help='Número de épocas de entrenamiento')
    parser.add_argument('--batch_size', type=int,
                       help='Tamaño de batch')
    
    # 🎯 NUEVOS ARGUMENTOS PARA PROFIT-AWARE LOSS
    parser.add_argument('--profit_aware_loss', action='store_true',
                       help='Usar función de pérdida consciente de rentabilidad')
    parser.add_argument('--loss_type', type=str,
                       choices=['profit_weighted', 'sharpe_aware', 'risk_adjusted', 'combined'],
                       default='combined',
                       help='Tipo de función de pérdida profit-aware')
    parser.add_argument('--base_fee', type=float, default=0.001,
                       help='Fee base por transacción (default: 0.001 = 0.1%)')
    parser.add_argument('--spread_cost', type=float, default=0.0005,
                       help='Costo de spread estimado (default: 0.0005 = 0.05%)')
    parser.add_argument('--profit_amplifier', type=float, default=2.0,
                       help='Factor para amplificar recompensas por trades rentables')
    parser.add_argument('--loss_amplifier', type=float, default=1.5,
                       help='Factor para amplificar penalizaciones por pérdidas')
    parser.add_argument('--disable_enhanced_metrics', action='store_true',
                       help='Deshabilitar métricas de trading mejoradas')

    # 🚀 NUEVO: Argumentos para arquitectura TCN mejorada
    parser.add_argument('--enhanced_tcn', action='store_true',
                       help='Usar arquitectura TCN mejorada (RECOMENDADO)')
    parser.add_argument('--tcn_architecture', type=str,
                       choices=['enhanced', 'original', 'hybrid'],
                       default='enhanced',
                       help='Arquitectura TCN a usar (default: enhanced)')
    
    # Argumento para modo no interactivo
    parser.add_argument('--non_interactive', action='store_true',
                       help='Ejecutar sin configuración interactiva')

    args = parser.parse_args()

    # 🎯 CREAR CONFIGURACIÓN DESDE ARGUMENTOS
    config = TrainingConfig()

    # Aplicar argumentos si están presentes
    if args.feature_set:
        config.feature_set = args.feature_set
    if args.balanced_regimes:
        config.use_balanced_regimes = True
    if args.regime_method:
        config.regime_balance_method = args.regime_method
    if args.samples_per_regime:
        config.target_samples_per_regime = args.samples_per_regime
    
    # ✅ NUEVO: APLICAR ARGUMENTOS DE PERÍODOS DE FECHAS
    if args.use_date_periods:
        config.use_date_periods = True
    if args.date_periods_method:
        config.date_periods_method = args.date_periods_method
    if args.preset_periods:
        config.preset_periods = args.preset_periods
    if args.custom_periods:
        config.custom_periods = args.custom_periods
        # 🎯 PROCESAR PERÍODOS PERSONALIZADOS
        config.use_date_periods = True
        config.date_periods = parse_custom_periods(args.custom_periods)
        
    if args.timeframe:
        config.timeframe = args.timeframe
    if args.pairs:
        config.pairs = args.pairs
    if args.prediction_horizon:
        config.prediction_horizon = args.prediction_horizon
    if args.lookback_window:
        config.lookback_window = args.lookback_window
    if args.training_days:
        config.training_days = args.training_days
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    
    # 🎯 NUEVOS ARGUMENTOS PROFIT-AWARE LOSS
    if args.profit_aware_loss:
        config.use_profit_aware_loss = True
    if args.loss_type:
        config.loss_type = args.loss_type
    if args.base_fee:
        config.base_fee = args.base_fee
    if args.spread_cost:
        config.spread_cost = args.spread_cost
    if args.profit_amplifier:
        config.profit_amplifier = args.profit_amplifier
    if args.loss_amplifier:
        config.loss_amplifier = args.loss_amplifier
    if args.disable_enhanced_metrics:
        config.use_enhanced_metrics = False
    
    # 🚀 NUEVO: APLICAR ARGUMENTOS DE ARQUITECTURA TCN
    if args.enhanced_tcn:
        config.use_enhanced_tcn = True
        config.tcn_architecture = 'enhanced'
    if args.tcn_architecture:
        config.tcn_architecture = args.tcn_architecture
        config.use_enhanced_tcn = (args.tcn_architecture in ['enhanced', 'hybrid'])

    # 🎯 EJECUTAR ENTRENAMIENTO
    if args.non_interactive:
        # Modo no interactivo con argumentos
        print("🎯 ENTRENADOR TCN ADAPTATIVO - MODO NO INTERACTIVO")
        print("=" * 70)
        config.print_config()

        trainer = AdaptiveTCNTrainer(config)
        trainer.validate_configuration_consistency()

        print(f"\n🚀 INICIANDO ENTRENAMIENTO...")
        print(f"📊 Pares: {', '.join(trainer.pairs)}")
        print(f"⏰ Timeframe: {config.timeframe}")
        print(f"🔮 Horizonte: {config.prediction_horizon} minutos")
        print(f"📊 Ventana: {config.lookback_window} períodos")
        print(f"📅 Datos: {config.training_days} días")
        print(f"🎯 Épocas: {config.epochs}")
        print(f"🎯 Feature Set: {config.feature_set}")
        print(f"🏗️ Arquitectura TCN: {config.tcn_architecture}")
        if config.use_enhanced_tcn:
            print(f"🚀 TCN Mejorado: HABILITADO")
        if config.use_balanced_regimes:
            print(f"⚖️  Regímenes equilibrados: {config.regime_balance_method}")
        if config.use_date_periods:
            print(f"📅 Períodos de fechas: {config.date_periods_method}")
            if config.preset_periods:
                print(f"📅 Períodos predefinidos: {config.preset_periods}")
        print("=" * 70)

        async def run_training():
            results = {}
            for symbol in trainer.pairs:
                print(f"\n🔥 Entrenando {symbol}...")
                if not trainer.validate_training_requirements(symbol):
                    print(f"❌ VALIDACIÓN FALLIDA para {symbol}. Saltando...")
                    results[symbol] = False
                    continue

                success = await trainer.train_adaptive_model(symbol)
                results[symbol] = success

            print(f"\n🎯 RESUMEN FINAL:")
            print("=" * 40)
            for symbol, success in results.items():
                status = "✅ ÉXITO" if success else "❌ FALLO"
                print(f"   {symbol}: {status}")

            successful = sum(results.values())
            print(f"\n🏆 Modelos entrenados exitosamente: {successful}/{len(results)}")

            if successful > 0:
                print(f"📁 Modelos guardados en: models/adaptive_<symbol>_<timeframe>_<config>/")
                print(f"🎯 ¡Listo para usar en trading!")
            else:
                print(f"❌ No se pudo entrenar ningún modelo. Revisa los errores arriba.")

        asyncio.run(run_training())
    else:
        # Modo interactivo (comportamiento original)
        asyncio.run(main()) 

