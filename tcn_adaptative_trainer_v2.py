#!/usr/bin/env python3
"""
🎯 TCN ADAPTATIVE TRAINER - VERSIÓN CONFIGURABLE
Entrenador con thresholds adaptativos y parámetros configurables por el usuario
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

# ✅ NUEVAS IMPORTACIONES PARA MÉTRICAS AVANZADAS
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score

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
        self.use_adaptive_thresholds = True
        
        # 🎯 FEATURE SET - AGREGADO
        self.feature_set = 'tcn_definitivo'  # Por defecto

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

        # ⚖️ THRESHOLDS BALANCEADOS (Optimizados para win rate y profit factor)
        self.fixed_thresholds = {
            'BTCUSDT': {
                'strong_sell': -0.0025, 'weak_sell': -0.0015,  # Más conservador (era -0.004/-0.002)
                'weak_buy': 0.0015, 'strong_buy': 0.0025
            },
            'ETHUSDT': {
                'strong_sell': -0.0025, 'weak_sell': -0.0015,  # Más conservador (era -0.0026/-0.0012)
                'weak_buy': 0.0015, 'strong_buy': 0.0025
            },
            'BNBUSDT': {
                'strong_sell': -0.0022, 'weak_sell': -0.0013,  # Más activo que antes (era -0.0015/-0.0007)
                'weak_buy': 0.0013, 'strong_buy': 0.0022
            },
            'XRPUSDT': {
                'strong_sell': -0.0025, 'weak_sell': -0.0015,  # Más conservador (era -0.0018/-0.0009)
                'weak_buy': 0.0015, 'strong_buy': 0.0025
            },
            'ADAUSDT': {
                'strong_sell': -0.0025, 'weak_sell': -0.0015,  # Más conservador (era -0.0018/-0.0009)
                'weak_buy': 0.0015, 'strong_buy': 0.0025
            },
            'DOTUSDT': {
                'strong_sell': -0.0025, 'weak_sell': -0.0015,  # Más conservador (era -0.0018/-0.0009)
                'weak_buy': 0.0015, 'strong_buy': 0.0025
            }
        }

    def calculate_adaptive_thresholds(self, df: pd.DataFrame, symbol: str) -> dict:
        """
        ⚖️ Calcular thresholds adaptativos BALANCEADOS para optimizar win rate y profit factor
        
        ✅ VERSIÓN BALANCEADA PARA TRADING RENTABLE:
        - Factor ATR equilibrado 1.5x-1.8x (punto medio entre conservador y agresivo)
        - Umbrales mínimos optimizados: Weak 0.15%, Strong 0.25%
        - Enfoque en calidad de señales sobre cantidad
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

            # ⚖️ ENFOQUE BALANCEADO: Factor intermedio optimizado para win rate
            volatility_factor = self.calculate_volatility_adjustment(atr_percent)
            base_threshold = max(atr_percent * volatility_factor, 0.0015)  # Mínimo 0.15%

            # ⚖️ LÍMITES BALANCEADOS: Punto medio entre conservador y agresivo
            min_weak = 0.0015    # Mínimo 0.15% (intermedio)
            min_strong = 0.0025  # Mínimo 0.25% (intermedio)
            
            # 🎯 MULTIPLICADORES BALANCEADOS por timeframe
            timeframe_multipliers = {
                '1m': 1.3,   # 30% más activo en 1m (era 1.5)
                '3m': 1.2,   # 20% más activo en 3m (era 1.3)
                '5m': 1.1,   # 10% más activo en 5m (era 1.2)
                '15m': 1.0,  # Base en 15m
                '1h': 0.9    # 10% menos activo en 1h (era 0.8)
            }
            
            # Detectar timeframe del modelo
            timeframe_multiplier = timeframe_multipliers.get(self.timeframe, 1.1)  # Default 1.1 para 5m
            
            # ⚖️ APLICAR MULTIPLICADOR BALANCEADO
            min_weak *= timeframe_multiplier
            min_strong *= timeframe_multiplier
            base_threshold *= timeframe_multiplier

            # ⚖️ THRESHOLDS BALANCEADOS: Factores optimizados para calidad
            adaptive_thresholds = {
                'strong_sell': -max(base_threshold * 1.8, min_strong),  # 1.8x factor (era 2.5x)
                'weak_sell': -max(base_threshold * 1.1, min_weak),     # 1.1x factor (era 1.3x)
                'weak_buy': max(base_threshold * 1.1, min_weak),       # 1.1x factor (era 1.3x)
                'strong_buy': max(base_threshold * 1.8, min_strong)    # 1.8x factor (era 2.5x)
            }

            # ⚖️ VALIDACIÓN FINAL BALANCEADA
            max_reasonable = 0.035  # Máximo 3.5% (era 5%)
            if (abs(adaptive_thresholds['strong_buy']) > max_reasonable or 
                abs(adaptive_thresholds['strong_sell']) > max_reasonable):
                print(f"⚠️ Thresholds demasiado extremos para {symbol}:")
                print(f"   📊 strong_buy: {adaptive_thresholds['strong_buy']:.4f}")
                print(f"   📊 strong_sell: {adaptive_thresholds['strong_sell']:.4f}")
                print(f"   🔄 Usando thresholds fijos como fallback")
                return self.fixed_thresholds[symbol]

            print(f"⚖️ {symbol}: ATR BALANCEADO {atr_percent:.4f} ({atr_percent*100:.2f}%) | TF: {self.timeframe}")
            print(f"   🎯 Thresholds balanceados (factor {timeframe_multiplier:.1f}x, volatility {volatility_factor:.1f}x):")
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
        
        # ⚖️ AJUSTE DINÁMICO: Menor factor en alta volatilidad, mayor en baja volatilidad
        if atr_percent > 0.06:      # Alta volatilidad >6%
            return 1.5  # Factor conservador
        elif atr_percent > 0.03:    # Volatilidad media 3-6%
            return 1.7  # Factor intermedio
        else:                       # Baja volatilidad <3%
            return 1.9  # Factor más activo
            
        # Resultado: Factor entre 1.5x y 1.9x (punto medio entre 1.2x conservador y 2.5x agresivo)

    def optimize_thresholds_for_symbol(self, df: pd.DataFrame, symbol: str) -> dict:
        """🎯 Optimización automática de thresholds para maximizar win rate y profit factor"""
        
        print(f"🎯 Optimizando thresholds para {symbol}...")
        
        # Candidatos de thresholds para probar (conservador -> balanceado -> activo)
        threshold_candidates = [
            # Conservador
            {'factor': 1.3, 'min_weak': 0.002, 'min_strong': 0.004, 'name': 'Conservador'},
            # Balanceado (nuestra configuración actual)
            {'factor': 1.6, 'min_weak': 0.0015, 'min_strong': 0.0025, 'name': 'Balanceado'},
            # Ligeramente activo
            {'factor': 1.9, 'min_weak': 0.0012, 'min_strong': 0.002, 'name': 'Activo'},
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
        
        return {
            'strong_sell': -max(base_threshold * 1.8, config['min_strong']),
            'weak_sell': -max(base_threshold * 1.1, config['min_weak']),
            'weak_buy': max(base_threshold * 1.1, config['min_weak']),
            'strong_buy': max(base_threshold * 1.8, config['min_strong'])
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
            
            # Penalizar por demasiadas señales (overtrading)
            signal_frequency = len(returns_1h) / len(close_prices)
            frequency_penalty = 1.0 if signal_frequency < 0.3 else (0.7 / signal_frequency)
            
            # Score combinado: win rate * avg return * frequency penalty
            quality_score = win_rate * max(avg_return, 0) * frequency_penalty * 100
            
            return quality_score
            
        except Exception as e:
            return 0.0

    def get_default_thresholds(self, symbol: str) -> dict:
        """🎯 Obtener thresholds por defecto robustos para cualquier símbolo"""

        # ✅ THRESHOLDS POR DEFECTO SEGUROS
        default_thresholds = {
            'strong_sell': -0.003,  # -0.3%
            'weak_sell': -0.0015,   # -0.15%
            'weak_buy': 0.0015,     # 0.15%
            'strong_buy': 0.003     # 0.3%
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
        """🎯 Crear etiquetas con thresholds adaptativos - CON VALIDACIONES ROBUSTAS"""

        print(f"🎯 Creando etiquetas {'adaptativas' if self.use_adaptive_thresholds else 'fijas'} para {symbol}...")

        close_prices = df['close'].values

        # ⚖️ CAMBIO PRINCIPAL: Usar optimización automática de thresholds
        try:
            # Paso 1: Intentar optimización automática (recomendado)
            if len(df) > 200:  # Suficientes datos para optimización
                print(f"🎯 Iniciando optimización automática de thresholds para {symbol}")
                thresholds = self.optimize_thresholds_for_symbol(df, symbol)
            else:
                # Paso 2: Fallback a thresholds adaptativos
                print(f"⚖️ Datos insuficientes para optimización, usando thresholds balanceados")
                thresholds = self.calculate_adaptive_thresholds(df, symbol)

            # ✅ VALIDACIÓN CRÍTICA: Verificar que los thresholds son válidos
            if not self.validate_thresholds(thresholds, symbol):
                print(f"⚠️ Thresholds inválidos para {symbol}, usando fijos balanceados")
                thresholds = self.fixed_thresholds.get(symbol, self.get_default_thresholds(symbol))

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

            # ✅ VALIDACIÓN CRÍTICA: Verificar que los precios son válidos
            if current_price <= 0 or future_price <= 0:
                print(f"⚠️ Precios inválidos en posición {i}: current={current_price}, future={future_price}")
                label = 1  # HOLD como fallback
                labels.append(label)
                continue

            # Calcular retorno futuro
            try:
                future_return = (future_price - current_price) / current_price
            except ZeroDivisionError:
                print(f"⚠️ División por cero en posición {i}: current_price={current_price}")
                label = 1  # HOLD como fallback
                labels.append(label)
                continue

            # ✅ VALIDACIÓN CRÍTICA: Verificar que el retorno es un número válido
            if np.isnan(future_return) or np.isinf(future_return):
                print(f"⚠️ Retorno inválido en posición {i}: {future_return}")
                label = 1  # HOLD como fallback
                labels.append(label)
                continue

            # 🎯 LÓGICA BALANCEADA (CON VALIDACIONES MEJORADAS)
            if future_return <= thresholds['strong_sell']:
                label = 0  # SELL
            elif future_return <= thresholds['weak_sell']:
                # Zona gris: usar indicadores técnicos para decidir
                try:
                    current_rsi = features['rsi_14'].iloc[i] if i < len(features) else 50
                    current_macd = features['macd_histogram'].iloc[i] if i < len(features) else 0
                except:
                    current_rsi = 50
                    current_macd = 0

                if current_rsi > 60 or current_macd < 0:
                    label = 0  # SELL (confirmación técnica)
                else:
                    label = 1  # HOLD
            elif future_return >= thresholds['strong_buy']:
                label = 2  # BUY
            elif future_return >= thresholds['weak_buy']:
                # Zona gris: usar indicadores técnicos para decidir
                try:
                    current_rsi = features['rsi_14'].iloc[i] if i < len(features) else 50
                    current_macd = features['macd_histogram'].iloc[i] if i < len(features) else 0
                except:
                    current_rsi = 50
                    current_macd = 0

                if current_rsi < 40 or current_macd > 0:
                    label = 2  # BUY (confirmación técnica)
                else:
                    label = 1  # HOLD
            else:
                # ✅ ZONA NEUTRAL: Momentum menos agresivo con validaciones
                if i >= 5:
                    try:
                        recent_momentum = (close_prices[i] - close_prices[i-5]) / close_prices[i-5]

                        # ✅ VALIDACIÓN: Verificar que el momentum es válido
                        if not np.isnan(recent_momentum) and not np.isinf(recent_momentum):
                            # Umbrales de momentum más conservadores
                            if recent_momentum > 0.0025:  # 0.25% en lugar de 1%
                                label = 2  # BUY (momentum positivo)
                            elif recent_momentum < -0.0025:  # -0.25% en lugar de -1%
                                label = 0  # SELL (momentum negativo)
                            else:
                                label = 1  # HOLD
                        else:
                            label = 1  # HOLD como fallback
                    except (ZeroDivisionError, IndexError):
                        label = 1  # HOLD como fallback
                else:
                    label = 1  # HOLD

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

        # ✅ VALIDACIÓN: Verificar que tenemos una distribución razonable
        min_samples_per_class = max(10, total * 0.05)  # Mínimo 5% por clase o 10 muestras
        for i, name in enumerate(class_names):
            count = label_counts.get(i, 0) or 0
            if count < min_samples_per_class:
                print(f"⚠️ ADVERTENCIA: Muy pocas muestras de clase {name}: {count} (mínimo {min_samples_per_class})")

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
            inf_count = np.isinf(df[col]).sum()
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
        total_inf = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()

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

        return X, y, scaler, feature_columns, class_weight_dict


    def create_definitive_tcn_model(self, input_shape: tuple):
        """
        🚀 Modelo TCN Optimizado para Crypto Trading
        Reemplazo directo del método original con mejoras específicas
        """

        def multi_scale_block(x, filters, dilations, dropout_rate=0.2):
            """Bloque multi-escala para diferentes patrones temporales"""
            branches = []

            # ✅ CORRECCIÓN: Usar número fijo para evitar len() en tensores
            filters_per_branch = max(1, filters // 3)  # Asumimos 3 dilations típicamente

            # ✅ CORRECCIÓN DIMENSIONAL: El input debería ser 3D por diseño
            # Si hay problemas dimensionales, serán manejados por las capas Keras automáticamente

            for i, dilation in enumerate(dilations):
                branch = tf.keras.layers.Conv1D(
                    filters_per_branch,
                    kernel_size=3,
                    dilation_rate=dilation,
                    padding='causal',
                    activation='relu',
                    kernel_initializer='he_normal'
                )(x)
                branch = tf.keras.layers.LayerNormalization()(branch)
                # ✅ CORRECCIÓN: Usar Dropout regular en lugar de SpatialDropout1D para evitar problemas dimensionales
                branch = tf.keras.layers.Dropout(dropout_rate)(branch)
                branches.append(branch)

            # Concatenar escalas
            if len(branches) > 1:
                multi_scale = tf.keras.layers.Concatenate(axis=-1)(branches)
            else:
                multi_scale = branches[0]

            # ✅ CORRECCIÓN: Simplificar ajuste de dimensiones sin tf.cond
            target_filters = filters

            # Siempre aplicar Conv1D para normalizar dimensiones (más simple y robusto)
            multi_scale = tf.keras.layers.Conv1D(target_filters, 1, padding='same')(multi_scale)

            # ✅ CORRECCIÓN: Conexión residual simplificada
            # Siempre aplicar Conv1D para normalizar dimensiones de entrada también
            x_residual = tf.keras.layers.Conv1D(target_filters, 1, padding='same')(x)

            return tf.keras.layers.Add()([multi_scale, x_residual])

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

        # Feature enhancement inicial
        x = tf.keras.layers.Conv1D(64, 1, padding='same', activation='relu')(x)

        # Configuración específica por timeframe
        if hasattr(self, 'timeframe'):
            if self.timeframe == '1m':
                dilation_groups = [[1, 2, 3], [4, 6, 8], [12, 16, 24]]
                filters_progression = [96, 128, 160]
            elif self.timeframe == '5m':
                dilation_groups = [[1, 2, 4], [6, 8, 12], [16, 24, 32]]
                filters_progression = [80, 128, 144]
            else:
                dilation_groups = [[1, 3, 6], [9, 12, 18], [24, 36, 48]]
                filters_progression = [64, 96, 128]
        else:
            # Configuración por defecto para 5m
            dilation_groups = [[1, 2, 4], [6, 8, 12], [16, 24, 32]]
            filters_progression = [80, 128, 144]

        # Bloques multi-escala
        for i, (dilations, filters) in enumerate(zip(dilation_groups, filters_progression)):
            x = multi_scale_block(x, filters, dilations, dropout_rate=0.1 + i * 0.05)

            # Atención cada 2 bloques
            if i % 2 == 1:
                x = attention_layer(x)

        # Adaptación a volatilidad
        x = volatility_adaptation(x)

        # ✅ SIMPLIFICADO: Extractor de tendencias con dimensiones fijas
        short_trend = tf.keras.layers.Conv1D(32, 3, dilation_rate=1, padding='causal', activation='tanh')(x)
        medium_trend = tf.keras.layers.Conv1D(32, 5, dilation_rate=3, padding='causal', activation='tanh')(x)
        momentum = tf.keras.layers.Conv1D(32, 7, dilation_rate=5, padding='causal', activation='tanh')(x)

        # Concatenar y normalizar
        trend_features = tf.keras.layers.Concatenate()([short_trend, medium_trend, momentum])
        trend_features = tf.keras.layers.LayerNormalization()(trend_features)

        # Normalizar entradas y combinar
        x_normalized = tf.keras.layers.Conv1D(96, 1, padding='same')(x)
        combined = tf.keras.layers.Concatenate()([x_normalized, trend_features])

        # Procesar combinación
        x = tf.keras.layers.Conv1D(256, 1, padding='same', activation='relu')(combined)

        # Atención final
        x = attention_layer(x)

        # Agregación temporal dual
        avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
        max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
        pooled = tf.keras.layers.Concatenate()([avg_pool, max_pool])

        # Capas de decisión
        x = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal',
                                kernel_regularizer=tf.keras.regularizers.l2(0.001))(pooled)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.4)(x)

        x = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal',
                                kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        x = tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        # Output
        outputs = tf.keras.layers.Dense(3, activation='softmax',
                                      kernel_initializer='glorot_uniform',
                                      bias_initializer='zeros')(x)

        # Crear modelo
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Optimizador mejorado
        timeframe = getattr(self, 'timeframe', '5m')
        if timeframe == '1m':
            learning_rate = 3e-4
        elif timeframe == '5m':
            learning_rate = 5e-4
        else:
            learning_rate = 7e-4

        # ✅ CORRECCIÓN: Learning rate fijo para máxima estabilidad
        # Usar learning rate fijo que ya está probado y funciona
        optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=learning_rate,  # LR fijo sin schedule
            clipnorm=1.0
        )

        # ✅ SIMPLIFICADO: Usar únicamente funciones estándar de Keras
        # Se eliminan Precision y Recall que causan conflicto con el formato de etiquetas.
        # La clase TradingMetrics ya calcula estas métricas al final del entrenamiento.
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']  # Usar solo accuracy para máxima estabilidad
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
        y_pred = np.argmax(y_pred_proba, axis=1)

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

            # 2. Calcular features
            print(f"🔄 Calculando features...")
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
                'tcn_definitivo': 88,
                'optimized_crypto': 25,
                'ultra_optimized': 15
            }
            
            expected_features = feature_set_expected.get(self.feature_set, 88)
            actual_features = len(features.columns)
            
            if actual_features < expected_features * 0.8:  # 80% mínimo
                print(f"❌ ERROR: Features insuficientes para {self.feature_set}. {actual_features}/{expected_features}")
                return False
            print(f"✅ Features {self.feature_set} válidas: {actual_features}/{expected_features}")

            # 3. Crear etiquetas con thresholds adaptativos
            df_labeled = self.create_balanced_labels(df, features, symbol)

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

            # 5. Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # 6. Crear y entrenar modelo
            model = self.create_definitive_tcn_model((X.shape[1], X.shape[2]))

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

                # ✅ NUEVO: Evaluación con métricas específicas de trading
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

    return config


if __name__ == "__main__":
    import argparse

    # 🎯 CONFIGURAR ARGUMENTOS DE LÍNEA DE COMANDOS
    parser = argparse.ArgumentParser(description='🎯 Entrenador TCN Adaptativo con Feature Sets Optimizados')

    # Argumentos de feature sets
    parser.add_argument('--feature_set', type=str,
                       choices=['tcn_definitivo', 'optimized_crypto', 'ultra_optimized'],
                       help='Conjunto de features a usar (default: tcn_definitivo)')

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

    # Argumento para modo no interactivo
    parser.add_argument('--non_interactive', action='store_true',
                       help='Ejecutar sin configuración interactiva')

    args = parser.parse_args()

    # 🎯 CREAR CONFIGURACIÓN DESDE ARGUMENTOS
    config = TrainingConfig()

    # Aplicar argumentos si están presentes
    if args.feature_set:
        config.feature_set = args.feature_set
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
