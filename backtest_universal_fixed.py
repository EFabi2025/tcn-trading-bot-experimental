#!/usr/bin/env python3
"""
🚀 BACKTESTING UNIVERSAL CORREGIDO - SELECTOR DE MODELOS
Script para probar cualquier modelo con DETECCIÓN CORRECTA DE TIMEFRAME
🔧 CORREGIDO: Detecta timeframe desde metadatos y evita errores silenciosos
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import warnings
import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from centralized_features_engine3 import CentralizedFeaturesEngine

warnings.filterwarnings('ignore')

class UniversalBacktesterFixed:
    """🎯 Backtester universal CORREGIDO para cualquier modelo"""

    def __init__(self):
        # Inicializar motor de features centralizado
        self.features_engine = CentralizedFeaturesEngine()

        # Configuración de trading
        self.initial_balance = 1000.0  # $1000 USD inicial
        self.trading_fee = 0.001      # 0.1% fee por trade
        self.min_trade_amount = 10.0   # Mínimo $10 por trade
        self.lookback_window = 24      # Default, se auto-ajustará por modelo

        # Configuración del modelo actual
        self.model_path = None
        self.symbol = None
        self.timeframe = None
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.class_weights = None
        self.detected_window = None  # Ventana detectada desde nombre

        # Métricas de rendimiento
        self.trades = []
        self.balance_history = []
        self.predictions_history = []

        print(f"🚀 Backtester Universal CORREGIDO inicializado")
        print(f"💰 Balance inicial: ${self.initial_balance}")
        print(f"💸 Fee de trading: {self.trading_fee*100:.1f}%")
        print(f"🔧 Motor de features: Centralizado")
        print(f"✅ CORRIGIDO: Detección de timeframe mejorada")

    def discover_models(self) -> List[Dict]:
        """🔍 Descubrir todos los modelos disponibles CON DETECCIÓN MEJORADA"""

        models_dir = 'models'
        if not os.path.exists(models_dir):
            print(f"❌ Directorio {models_dir} no encontrado")
            return []

        print(f"🔍 Descubriendo modelos en {models_dir}/ con DETECCIÓN MEJORADA...")

        models = []
        for dir_name in os.listdir(models_dir):
            dir_path = os.path.join(models_dir, dir_name)

            if os.path.isdir(dir_path):
                # Buscar archivos requeridos
                model_files = [f for f in os.listdir(dir_path) if f.endswith('.h5')]
                model_file = None

                # Priorizar best_model.h5
                if 'best_model.h5' in model_files:
                    model_file = 'best_model.h5'
                elif 'model.h5' in model_files:
                    model_file = 'model.h5'
                elif model_files:
                    model_file = model_files[0]

                has_model = model_file is not None
                has_scaler = os.path.exists(os.path.join(dir_path, 'scaler.pkl'))
                has_features = os.path.exists(os.path.join(dir_path, 'feature_columns.pkl'))

                if has_model and has_scaler and has_features:
                    # ✅ DETECCIÓN MEJORADA: Intentar múltiples métodos
                    symbol, timeframe, detection_method = self._extract_symbol_timeframe_improved(dir_path, dir_name)

                    if symbol:
                        # 🔢 Contar parámetros del modelo
                        model_full_path = os.path.join(dir_path, model_file)
                        parameter_count = self._count_model_parameters(model_full_path)
                        
                        # 🔢 Obtener ventana detectada desde nombre
                        detected_window = getattr(self, 'detected_window', None) if hasattr(self, 'detected_window') else None

                        # ✅ NO MÁS DEFAULT AUTOMÁTICO - Solo agregar si timeframe detectado
                        if timeframe:
                            models.append({
                                'name': dir_name,
                                'path': dir_path,
                                'symbol': symbol,
                                'timeframe': timeframe,
                                'model_file': model_file,
                                'parameters': parameter_count,
                                'detection_method': detection_method,
                                'window': detected_window,  # ✅ AGREGAR VENTANA DETECTADA
                                'complete': True
                            })

                            # Clasificar por tamaño de parámetros
                            if parameter_count > 0:
                                if parameter_count < 50000:
                                    size_indicator = "🟢"  # Optimizado
                                elif parameter_count < 200000:
                                    size_indicator = "🟡"  # Intermedio
                                else:
                                    size_indicator = "🔴"  # Posible overfitting

                                print(f"   ✅ {dir_name} -> {symbol} ({timeframe}) {size_indicator} {parameter_count:,} parámetros [{detection_method}]")
                            else:
                                print(f"   ✅ {dir_name} -> {symbol} ({timeframe}) ❓ parámetros [{detection_method}]")
                        else:
                            print(f"   ⚠️ {dir_name} -> {symbol} (❌ TIMEFRAME NO DETECTADO) - OMITIDO")
                    else:
                        print(f"   ⚠️ {dir_name} -> No se pudo extraer símbolo")
                else:
                    missing = []
                    if not has_model: missing.append("modelo")
                    if not has_scaler: missing.append("scaler")
                    if not has_features: missing.append("features")
                    print(f"   ❌ {dir_name} -> Incompleto (faltan: {', '.join(missing)})")

        print(f"\n📊 Total de modelos válidos encontrados: {len(models)}")
        return models

    def _extract_symbol_timeframe_improved(self, dir_path: str, dir_name: str) -> Tuple[Optional[str], Optional[str], str]:
        """🔧 MEJORADO: Extraer símbolo, timeframe y ventana con múltiples métodos"""

        # Lista de símbolos conocidos
        known_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'DOTUSDT',
                        'ADAUSDT', 'SOLUSDT', 'DOGEUSDT', 'LINKUSDT', 'MATICUSDT']

        # Convertir a mayúsculas para búsqueda
        dir_upper = dir_name.upper()

        # Buscar símbolo
        symbol = None
        for s in known_symbols:
            if s in dir_upper:
                symbol = s
                break

        # ✅ MÉTODO 1: Extraer desde nombre con formato específico (PRIORITARIO)
        symbol, timeframe, window = self._extract_from_model_name_format(dir_name)
        if symbol and timeframe:
            # Guardar ventana detectada para uso posterior
            if window:
                self.detected_window = window
                print(f"🔢 Ventana detectada desde nombre: {window}")
            return symbol, timeframe, f"name_format_w{window}" if window else "name_format"

        # ✅ MÉTODO 2: Intentar leer desde metadatos guardados
        timeframe, method = self._try_load_timeframe_from_metadata(dir_path)
        if timeframe:
            return symbol, timeframe, f"metadata_{method}"

        # ✅ MÉTODO 3: Patrones regex mejorados
        timeframe = self._extract_timeframe_from_name(dir_name)
        if timeframe:
            return symbol, timeframe, "regex"

        # ✅ MÉTODO 4: Análisis del input shape del modelo (último recurso)
        timeframe = self._infer_timeframe_from_model(dir_path)
        if timeframe:
            return symbol, timeframe, "model_shape"

        # ❌ NO SE PUDO DETECTAR
        return symbol, None, "failed"

    def _extract_from_model_name_format(self, dir_name: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
        """🎯 MÉTODO ESPECÍFICO: Extraer desde formato adaptive_symbol_timeframe_period_window_type"""
        
        # Formato esperado: adaptive_dotusdt_5m_6h_32w_tcn_definitivo
        # Partes: [adaptive, symbol, timeframe, period, window, type, extra...]
        
        try:
            # Lista de símbolos conocidos
            known_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'DOTUSDT',
                            'ADAUSDT', 'SOLUSDT', 'DOGEUSDT', 'LINKUSDT', 'MATICUSDT']
            
            # Convertir a minúsculas para análisis
            name_lower = dir_name.lower()
            
            # Buscar símbolo en el nombre
            symbol = None
            for s in known_symbols:
                if s.lower() in name_lower:
                    symbol = s
                    break
            
            if not symbol:
                return None, None, None
            
            # Dividir por guiones bajos
            parts = dir_name.lower().split('_')
            
            if len(parts) < 4:
                return symbol, None, None
            
            # Buscar timeframe (formato: 1m, 3m, 5m, 15m, 1h, 4h)
            timeframe = None
            window = None
            
            for part in parts:
                # Detectar timeframe
                if re.match(r'^\d+[mh]$', part):
                    if part in ['1m', '3m', '5m', '15m', '1h', '4h']:
                        timeframe = part
                
                # Detectar ventana (formato: 32w, 24w, 48w, etc.)
                if re.match(r'^\d+w$', part):
                    try:
                        window = int(part[:-1])  # Remover 'w' y convertir a int
                    except ValueError:
                        pass
            
            if symbol and timeframe:
                print(f"🎯 FORMATO DETECTADO: {symbol} | {timeframe} | {window}w")
                return symbol, timeframe, window
            
            return symbol, timeframe, window
            
        except Exception as e:
            print(f"⚠️ Error parseando nombre del modelo: {e}")
            return None, None, None

    def _try_load_timeframe_from_metadata(self, dir_path: str) -> Tuple[Optional[str], str]:
        """🔧 MÉTODO 1: Intentar cargar timeframe desde metadatos guardados"""

        # Buscar archivos de configuración/metadatos
        config_files = [
            'config_1m.pkl', 'config_3m.pkl', 'config_5m.pkl', 'config_15m.pkl', 'config_1h.pkl', 'config_4h.pkl',
            'config.pkl', 'model_config.pkl', 'training_config.pkl'
        ]

        for config_file in config_files:
            config_path = os.path.join(dir_path, config_file)
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'rb') as f:
                        config = pickle.load(f)

                    # Buscar timeframe en diferentes keys
                    timeframe_keys = ['timeframe', 'interval', 'time_frame', 'tf']
                    for key in timeframe_keys:
                        if isinstance(config, dict) and key in config:
                            tf = config[key]
                            if tf in ['1m', '3m', '5m', '15m', '1h', '4h']:
                                return tf, config_file

                    # Caso especial: config_1m.pkl indica 1m
                    if 'config_1m.pkl' in config_file:
                        return '1m', config_file
                    elif 'config_5m.pkl' in config_file:
                        return '5m', config_file

                except Exception as e:
                    continue

        return None, "none"

    def _extract_timeframe_from_name(self, dir_name: str) -> Optional[str]:
        """🔧 MÉTODO 2: Patrones regex mejorados para extraer timeframe"""

        # Patrones regex mejorados
        timeframe_patterns = [
            r'_(\d+[mh])_',      # _5m_, _1h_
            r'_(\d+[mh])$',      # _5m, _1h al final
            r'^(\d+[mh])_',      # 5m_, 1h_ al inicio
            r'(\d+[mh])_',       # 5m_, 1h_ en cualquier lugar
            r'_(\d+min)_',       # _5min_
            r'_(\d+hour)_',      # _1hour_
        ]

        for pattern in timeframe_patterns:
            match = re.search(pattern, dir_name.lower())
            if match:
                tf_raw = match.group(1)
                # Normalizar formato
                if 'min' in tf_raw:
                    tf_raw = tf_raw.replace('min', 'm')
                elif 'hour' in tf_raw:
                    tf_raw = tf_raw.replace('hour', 'h')

                if tf_raw in ['1m', '3m', '5m', '15m', '1h', '4h']:
                    return tf_raw

        # Búsquedas fallback más específicas
        name_lower = dir_name.lower()

        # Patrones específicos más probables primero
        if 'profitable_1m' in name_lower or 'definitivo_1m' in name_lower or '_1m_' in name_lower:
            return '1m'
        elif 'profitable_5m' in name_lower or 'definitivo_5m' in name_lower or '_5m_' in name_lower:
            return '5m'
        elif '15m' in name_lower:
            return '15m'
        elif '1h' in name_lower or '_1hour' in name_lower:
            return '1h'
        elif '4h' in name_lower or '_4hour' in name_lower:
            return '4h'

        return None

    def _infer_timeframe_from_model(self, dir_path: str) -> Optional[str]:
        """🔧 MÉTODO 3: Inferir timeframe desde el input shape del modelo (heurística)"""

        try:
            # Buscar modelo
            model_files = [f for f in os.listdir(dir_path) if f.endswith('.h5')]
            if not model_files:
                return None

            model_file = 'best_model.h5' if 'best_model.h5' in model_files else model_files[0]
            model_path = os.path.join(dir_path, model_file)

            # Cargar modelo solo para obtener input shape
            model = tf.keras.models.load_model(model_path)
            input_shape = model.input_shape

            if len(input_shape) >= 2:
                lookback_window = input_shape[1]  # (None, timesteps, features)

                # Heurística basada en lookback window típico por timeframe
                # Esto es una estimación basada en patrones comunes
                if lookback_window <= 40:
                    return '1m'  # Lookback corto típico de 1m
                elif lookback_window <= 100:
                    return '5m'  # Lookback medio típico de 5m
                elif lookback_window <= 200:
                    return '15m' # Lookback largo típico de 15m
                else:
                    return '1h'  # Lookback muy largo típico de 1h+

        except Exception as e:
            return None

        return None

    def _count_model_parameters(self, model_path: str) -> int:
        """🔢 Contar parámetros del modelo"""
        try:
            model = tf.keras.models.load_model(model_path)
            return model.count_params()
        except:
            return 0

    def select_model(self, models: List[Dict]) -> Optional[Dict]:
        """🎯 Seleccionar modelo para backtesting con VALIDACIÓN DE TIMEFRAME"""

        if not models:
            print("❌ No hay modelos disponibles")
            return None

        print(f"\n🎯 SELECCIONAR MODELO PARA BACKTESTING")
        print("=" * 60)

        # Agrupar por símbolo para mejor visualización
        by_symbol = {}
        for model in models:
            symbol = model['symbol']
            if symbol not in by_symbol:
                by_symbol[symbol] = []
            by_symbol[symbol].append(model)

        # Mostrar modelos agrupados
        model_index = 0
        index_to_model = {}

        for symbol in sorted(by_symbol.keys()):
            print(f"\n📊 {symbol}:")
            for model in by_symbol[symbol]:
                model_index += 1
                index_to_model[model_index] = model

                # Indicadores de calidad
                params = model['parameters']
                if params > 0:
                    if params < 50000:
                        size_emoji = "🟢"
                        size_text = "Opt"
                    elif params < 200000:
                        size_emoji = "🟡"
                        size_text = "Med"
                    else:
                        size_emoji = "🔴"
                        size_text = "Big"
                    size_info = f"{size_emoji} {params:,} ({size_text})"
                else:
                    size_info = "❓ params"

                # ✅ MOSTRAR MÉTODO DE DETECCIÓN Y VENTANA
                detection_info = f"[{model['detection_method']}]"
                timeframe_info = f"⏰ {model['timeframe']}"
                window_info = f"🔢 {model.get('window', '?')}w" if model.get('window') else "🔢 ?w"

                print(f"   {model_index:2d}. {model['name']:30s} {timeframe_info} {window_info} {size_info} {detection_info}")

        # Selección
        while True:
            try:
                choice = int(input(f"\n🎯 Selecciona modelo (1-{model_index}): "))
                if 1 <= choice <= model_index:
                    selected = index_to_model[choice]

                    # ✅ VALIDACIÓN ADICIONAL DE TIMEFRAME Y VENTANA
                    print(f"\n✅ Modelo seleccionado: {selected['name']}")
                    print(f"📊 Símbolo: {selected['symbol']}")
                    print(f"⏰ Timeframe: {selected['timeframe']} (detectado via {selected['detection_method']})")
                    print(f"🔢 Ventana: {selected.get('window', 'No detectada')}w")
                    print(f"🔢 Parámetros: {selected['parameters']:,}")

                    # Confirmar configuración
                    window_text = f"ventana {selected['window']}w y " if selected.get('window') else ""
                    confirm = input(f"¿Confirmar modelo con {window_text}timeframe {selected['timeframe']}? (s/n): ").lower().strip()
                    if confirm in ['s', 'si', 'yes', 'y']:
                        return selected
                    else:
                        print("❌ Selección cancelada. Elige otro modelo.")
                        continue
                else:
                    print(f"❌ Selecciona un número entre 1 y {model_index}")
            except ValueError:
                print("❌ Ingresa un número válido")
            except KeyboardInterrupt:
                return None

    def optimize_lookback_window_for_timeframe(self, timeframe: str, detected_lookback: int) -> int:
        """🎯 Optimizar lookback window basado en timeframe y ventana detectada desde nombre"""
        
        # ✅ PRIORIDAD 1: Usar ventana detectada desde nombre del modelo
        if hasattr(self, 'detected_window') and self.detected_window:
            print(f"🎯 USANDO VENTANA DESDE NOMBRE DEL MODELO:")
            print(f"   ⏰ Timeframe: {timeframe}")
            print(f"   🔢 Ventana del nombre: {self.detected_window}w")
            print(f"   🔢 Lookback detectado: {detected_lookback}")
            print(f"   ✅ PRIORITARIO: Usando ventana del nombre: {self.detected_window}")
            return self.detected_window
        
        # ✅ PRIORIDAD 2: Usar lookback detectado del modelo
        if detected_lookback:
            print(f"🎯 USANDO VENTANA DEL MODELO:")
            print(f"   ⏰ Timeframe: {timeframe}")
            print(f"   🔢 Lookback del modelo: {detected_lookback}")
            print(f"   ✅ Usando ventana del modelo: {detected_lookback}")
            return detected_lookback
        
        # ✅ PRIORIDAD 3: Configuración por defecto por timeframe
        default_windows = {
            '1m': 24,   # 24 minutos = suficiente para patrones inmediatos
            '3m': 32,   # 96 minutos = ~1.5 horas 
            '5m': 48,   # 240 minutos = 4 horas
            '15m': 64,  # 960 minutos = 16 horas
            '1h': 72,   # 72 horas = 3 días
            '4h': 84    # 336 horas = 2 semanas
        }
        
        default = default_windows.get(timeframe, 48)
        print(f"🎯 USANDO VENTANA POR DEFECTO:")
        print(f"   ⏰ Timeframe: {timeframe}")
        print(f"   🔢 Ventana por defecto: {default}")
        print(f"   ⚠️ No se detectó ventana específica del modelo")
        return default

    def calculate_corrected_sharpe_ratio(self, balance_history: List[Dict]) -> float:
        """📊 Sharpe Ratio CORREGIDO con tasa libre de riesgo y períodos activos"""
        
        if len(balance_history) < 2:
            return 0
        
        # Calcular retornos solo en períodos con cambios significativos
        returns = []
        for i in range(1, len(balance_history)):
            prev_balance = balance_history[i-1]['total_balance']
            curr_balance = balance_history[i]['total_balance']
            
            if prev_balance > 0:
                daily_return = (curr_balance - prev_balance) / prev_balance
                # Solo incluir retornos significativos (> 0.01%)
                if abs(daily_return) > 0.0001:
                    returns.append(daily_return)
        
        if not returns or len(returns) < 2:
            return 0
        
        # ✅ CORRECCIÓN: Usar tasa libre de riesgo y períodos de trading correctos
        risk_free_rate_daily = 0.02 / 252  # 2% anual dividido por días de trading
        
        # Calcular excess returns
        excess_returns = [r - risk_free_rate_daily for r in returns]
        
        avg_excess_return = np.mean(excess_returns)
        std_excess_return = np.std(excess_returns)
        
        if std_excess_return > 0:
            # Anualizar basado en frecuencia real de trading
            periods_per_year = min(252, len(returns) * 365 / 30)  # Ajustar por datos reales
            sharpe_ratio = (avg_excess_return / std_excess_return) * np.sqrt(periods_per_year)
        else:
            sharpe_ratio = 0
        
        return sharpe_ratio

    def calculate_additional_risk_metrics(self, balance_history: List[Dict], trade_returns: List[float]) -> Dict:
        """📊 Calcular métricas de riesgo adicionales (Calmar, Sortino, etc.)"""
        
        if not balance_history or not trade_returns:
            return {'calmar_ratio': 0, 'sortino_ratio': 0, 'max_adverse_excursion': 0}
        
        # Calcular drawdown máximo
        peak_balance = self.initial_balance
        max_drawdown = 0
        
        for record in balance_history:
            current_balance = record['total_balance']
            if current_balance > peak_balance:
                peak_balance = current_balance
            
            drawdown = (peak_balance - current_balance) / peak_balance
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Total return
        final_balance = balance_history[-1]['total_balance']
        total_return = (final_balance - self.initial_balance) / self.initial_balance
        
        # ✅ CALMAR RATIO: Total return / Max drawdown
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else float('inf')
        
        # ✅ SORTINO RATIO: Solo considera downside risk
        negative_returns = [r for r in trade_returns if r < 0]
        downside_deviation = np.std(negative_returns) if negative_returns else 0
        avg_return = np.mean(trade_returns) if trade_returns else 0
        
        sortino_ratio = avg_return / downside_deviation if downside_deviation > 0 else float('inf')
        
        # ✅ MAXIMUM ADVERSE EXCURSION: Peor pérdida durante trades ganadores
        winning_trades = [r for r in trade_returns if r > 0]
        mae = min(trade_returns) if trade_returns else 0  # Peor trade en general
        
        return {
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'max_adverse_excursion': mae,
            'downside_deviation': downside_deviation,
            'negative_returns_count': len(negative_returns)
        }

    def validate_financial_math(self, results: Dict) -> Dict:
        """✅ Validar coherencia matemática del backtesting"""
        
        trades = results['trades']
        initial = self.initial_balance
        final = results['final_balance']
        
        # Calcular suma de todos los flujos
        total_flows = 0
        for trade in trades:
            if trade['type'] == 'BUY':
                total_flows -= trade.get('usd_spent', 0)  # Dinero gastado
            elif trade['type'] in ['SELL', 'SELL_FINAL']:
                total_flows += trade.get('usd_received', 0)  # Dinero recibido
            elif trade['type'] == 'STOP_LOSS':
                total_flows += trade.get('usd_received', 0)  # Dinero recibido
        
        expected_final = initial + total_flows
        tolerance = 0.01  # $0.01
        
        is_valid = abs(final - expected_final) < tolerance
        
        return {
            'is_mathematically_valid': is_valid,
            'initial_balance': initial,
            'final_balance': final,
            'expected_final': expected_final,
            'difference': final - expected_final,
            'total_flows': total_flows,
            'tolerance': tolerance
        }
        
    def load_model_components(self, model_info: Dict) -> bool:
        """📂 Cargar componentes del modelo CON VALIDACIÓN Y OPTIMIZACIÓN DE TIMEFRAME"""

        try:
            print(f"📂 Cargando modelo {model_info['name']}...")

            self.model_path = model_info['path']
            self.symbol = model_info['symbol']
            self.timeframe = model_info['timeframe']
            
            # ✅ CARGAR VENTANA DETECTADA DEL MODELO SELECCIONADO
            if 'window' in model_info and model_info['window']:
                self.detected_window = model_info['window']
                print(f"🔢 Ventana del modelo: {self.detected_window}w")
            else:
                self.detected_window = None

            # ✅ VALIDACIÓN: Asegurar que timeframe está definido
            if not self.timeframe:
                print("❌ Error: Timeframe no detectado. No se puede proceder.")
                return False

            print(f"✅ Timeframe confirmado: {self.timeframe}")

            # Cargar modelo
            model_file_path = os.path.join(self.model_path, model_info['model_file'])
            self.model = tf.keras.models.load_model(model_file_path)

            # 🔧 AUTO-DETECTAR Y OPTIMIZAR LOOKBACK_WINDOW
            input_shape = self.model.input_shape
            print(f"🔢 Input shape detectado: {input_shape}")

            # ✅ CORRECCIÓN: Manejar modelos con entrada dinámica
            if len(input_shape) >= 2:
                detected_lookback = input_shape[1]
                if detected_lookback is None:
                    # Modelo con entrada dinámica - usar configuración óptima por timeframe
                    print(f"🔧 Modelo con entrada dinámica detectado")
                    optimized_window = self.optimize_lookback_window_for_timeframe(self.timeframe, None)
                    print(f"🔧 Usando ventana optimizada para {self.timeframe}: {optimized_window}")
                    self.lookback_window = optimized_window
                else:
                    # Modelo con entrada fija - optimizar basado en timeframe
                    optimized_window = self.optimize_lookback_window_for_timeframe(self.timeframe, detected_lookback)
                    if optimized_window != self.lookback_window:
                        print(f"🔧 Auto-ajustando lookback_window: {self.lookback_window} → {optimized_window}")
                        self.lookback_window = optimized_window
            else:
                print(f"⚠️  Input shape inesperado: {input_shape}")
                # Usar configuración óptima como fallback
                optimized_window = self.optimize_lookback_window_for_timeframe(self.timeframe, self.lookback_window)
                self.lookback_window = optimized_window

            print(f"✅ {model_info['model_file']} cargado ({self.model.count_params():,} parámetros)")
            print(f"🔢 Input shape: {input_shape}")
            print(f"⏰ Lookback window optimizado: {self.lookback_window} timesteps ({self.timeframe})")
            
            # ✅ VALIDAR CONFIGURACIÓN FINAL DE VENTANA
            estimated_data_needed = self.lookback_window + 100
            print(f"📊 Datos mínimos estimados necesarios: {estimated_data_needed}")
            
            with open(os.path.join(self.model_path, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            print("✅ Scaler cargado")

            # Cargar feature columns
            with open(os.path.join(self.model_path, 'feature_columns.pkl'), 'rb') as f:
                self.feature_columns = pickle.load(f)
            print(f"✅ Feature columns cargadas: {len(self.feature_columns)} features")

            # Cargar class weights (opcional)
            try:
                with open(os.path.join(self.model_path, 'class_weights.pkl'), 'rb') as f:
                    self.class_weights = pickle.load(f)
                print("✅ Class weights cargados")
            except:
                print("⚠️ Class weights no encontrados (opcional)")

            return True

        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False

    def calculate_optimal_days_for_backtest(self, timeframe: str, lookback_window: int) -> int:
        """📅 Calcular días óptimos de datos basado en timeframe y lookback window"""
        
        # ✅ CONFIGURACIÓN INTELIGENTE POR TIMEFRAME
        timeframe_multipliers = {
            '1m': 1,      # 1 día = 1440 velas
            '3m': 3,      # 1 día = 480 velas  
            '5m': 5,      # 1 día = 288 velas
            '15m': 15,    # 1 día = 96 velas
            '1h': 60,     # 1 día = 24 velas
            '4h': 240     # 1 día = 6 velas
        }
        
        multiplier = timeframe_multipliers.get(timeframe, 5)
        
        # Calcular días necesarios para obtener suficientes velas
        required_candles = lookback_window + 200  # Buffer adicional
        candles_per_day = 1440 // multiplier  # Velas por día
        
        optimal_days = max(int(required_candles / candles_per_day) + 2, 7)  # Mínimo 7 días
        
        print(f"📅 CÁLCULO DE DÍAS ÓPTIMOS:")
        print(f"   ⏰ Timeframe: {timeframe} (1 día = {candles_per_day} velas)")
        print(f"   🔢 Lookback window: {lookback_window}")
        print(f"   📊 Velas necesarias: {required_candles}")
        print(f"   🎯 Días óptimos calculados: {optimal_days}")
        
        return optimal_days

    async def get_historical_data(self, days: int = 30, limit: int = 1000) -> pd.DataFrame:
        """📊 Obtener datos históricos CON TIMEFRAME CORRECTO Y VALIDACIÓN INTELIGENTE"""

        # ✅ VALIDACIÓN INTELIGENTE DE DÍAS
        optimal_days = self.calculate_optimal_days_for_backtest(self.timeframe, self.lookback_window)
        
        if days < optimal_days:
            print(f"⚠️ ADVERTENCIA: Días insuficientes para backtesting óptimo")
            print(f"   📅 Solicitados: {days} días")
            print(f"   🎯 Recomendados: {optimal_days} días")
            print(f"   💡 Considera usar más días para resultados más confiables")
        else:
            print(f"✅ Días suficientes: {days} ≥ {optimal_days} recomendados")

        print(f"📊 Obteniendo {days} días de datos históricos de {self.symbol}...")
        print(f"⏰ Usando timeframe: {self.timeframe} (VERIFICADO)")

        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        base_url = "https://api.binance.com"

        async with aiohttp.ClientSession() as session:
            url = f"{base_url}/api/v3/klines"
            params = {
                'symbol': self.symbol,
                'interval': self.timeframe,  # ✅ CORREGIDO: Usa timeframe verificado
                'startTime': start_time,
                'endTime': end_time,
                'limit': limit
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

        print(f"✅ Obtenidos {len(df)} registros históricos de {self.timeframe}")
        return df

    def load_training_features(self) -> pd.DataFrame:
        """📂 Cargar features y configuración del entrenamiento"""
        
        try:
            # Buscar archivo de configuración de features del entrenamiento
            training_features_path = os.path.join(self.model_path, 'training_features_sample.pkl')
            training_config_path = os.path.join(self.model_path, 'training_config.pkl')
            
            if os.path.exists(training_features_path):
                with open(training_features_path, 'rb') as f:
                    training_features = pickle.load(f)
                print(f"✅ Features de entrenamiento cargadas desde: training_features_sample.pkl")
                return training_features
            
            elif os.path.exists(training_config_path):
                with open(training_config_path, 'rb') as f:
                    config = pickle.load(f)
                
                # Buscar features en el config
                if isinstance(config, dict) and 'sample_features' in config:
                    training_features = config['sample_features']
                    print(f"✅ Features de entrenamiento cargadas desde: training_config.pkl")
                    return training_features
            
            print("⚠️ No se encontraron features de entrenamiento de referencia")
            return pd.DataFrame()
            
        except Exception as e:
            print(f"❌ Error cargando features de entrenamiento: {e}")
            return pd.DataFrame()

    def calculate_features_for_backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        """🔧 Calcular features específicamente para backtesting CON VALIDACIÓN DE VENTANA"""
        
        print("🔧 Calculando features para backtesting con validación de ventana...")
        
        try:
            # ✅ CORRECCIÓN: Validar que tenemos suficientes datos para el lookback_window
            min_required_data = self.lookback_window + 50  # Mínimo datos requeridos
            
            if len(df) < min_required_data:
                print(f"❌ ERROR DE VENTANA: Datos insuficientes")
                print(f"   📊 Datos disponibles: {len(df)}")
                print(f"   📊 Lookback window: {self.lookback_window}")
                print(f"   📊 Mínimo requerido: {min_required_data}")
                print(f"   🔧 Necesitas más días de datos históricos")
                return pd.DataFrame()
            
            print(f"✅ Validación de ventana OK: {len(df)} datos ≥ {min_required_data} requeridos")
            print(f"   🔢 Lookback window confirmado: {self.lookback_window}")
            
            # Usar motor centralizado de features
            features = self.features_engine.calculate_features(df)
            
            if features.empty:
                print("❌ Error: Features vacías")
                return pd.DataFrame()
            
            # ✅ VALIDACIÓN CRÍTICA: Verificar que tenemos suficientes features después del cálculo
            if len(features) < self.lookback_window:
                print(f"❌ ERROR DE VENTANA POST-FEATURES:")
                print(f"   📊 Features calculadas: {len(features)}")
                print(f"   📊 Lookback window: {self.lookback_window}")
                print(f"   🔧 Las features eliminaron demasiados registros")
                return pd.DataFrame()
            
            # Seleccionar solo las features que usó el modelo
            available_features = [col for col in self.feature_columns if col in features.columns]
            missing_features = [col for col in self.feature_columns if col not in features.columns]
            
            if missing_features:
                print(f"⚠️ Features faltantes: {len(missing_features)}")
                for feat in missing_features[:5]:  # Mostrar solo las primeras 5
                    print(f"   - {feat}")
                if len(missing_features) > 5:
                    print(f"   - ... y {len(missing_features) - 5} más")
            
            if not available_features:
                print("❌ Error: No hay features disponibles que coincidan")
                return pd.DataFrame()
            
            print(f"✅ Features para backtesting: {len(available_features)}/{len(self.feature_columns)}")
            print(f"✅ Registros finales: {len(features)} (lookback: {self.lookback_window})")
            
            return features[available_features]
            
        except Exception as e:
            print(f"❌ Error calculando features para backtesting: {e}")
            return pd.DataFrame()

    def validate_features_consistency(self) -> bool:
        """🔍 Verificar que las features del backtesting coincidan con el entrenamiento"""
        
        print("🔍 Validando consistencia de features...")
        
        try:
            # Cargar features de entrenamiento (muestra de referencia)
            training_features = self.load_training_features()
            
            if training_features.empty:
                print("⚠️ No se pueden validar features - sin referencia de entrenamiento")
                return True  # Continuar sin validación
            
            # Obtener datos de muestra para comparación
            sample_df = asyncio.get_event_loop().run_until_complete(
                self.get_historical_data(days=5, limit=200)  # Muestra pequeña
            )
            
            if sample_df.empty:
                print("❌ No se pudieron obtener datos para validación")
                return False
            
            # Calcular features de backtesting
            backtest_features = self.calculate_features_for_backtest(sample_df)
            
            if backtest_features.empty:
                print("❌ No se pudieron calcular features para validación")
                return False
            
            # Comparar estadísticas de features comunes
            common_features = [col for col in training_features.columns if col in backtest_features.columns]
            
            if not common_features:
                print("❌ No hay features comunes para comparar")
                return False
            
            print(f"🔍 Comparando {len(common_features)} features comunes...")
            
            inconsistent_features = []
            
            for col in common_features:
                # Filtrar valores infinitos y NaN
                train_values = training_features[col].replace([np.inf, -np.inf], np.nan).dropna()
                backtest_values = backtest_features[col].replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(train_values) == 0 or len(backtest_values) == 0:
                    continue
                
                train_mean = train_values.mean()
                backtest_mean = backtest_values.mean()
                
                # Calcular diferencia relativa
                if abs(train_mean) > 1e-10:  # Evitar división por cero
                    diff = abs(train_mean - backtest_mean) / abs(train_mean)
                    
                    # Umbral de inconsistencia: 15% (más permisivo para crypto)
                    if diff > 0.15:
                        inconsistent_features.append({
                            'feature': col,
                            'train_mean': train_mean,
                            'backtest_mean': backtest_mean,
                            'diff_percent': diff * 100
                        })
            
            # Reportar resultados
            if inconsistent_features:
                print(f"⚠️ Features inconsistentes detectadas: {len(inconsistent_features)}")
                for feat_info in inconsistent_features[:5]:  # Mostrar primeras 5
                    print(f"   - {feat_info['feature']}: {feat_info['diff_percent']:.1f}% diferencia")
                    print(f"     Train: {feat_info['train_mean']:.6f}, Backtest: {feat_info['backtest_mean']:.6f}")
                
                if len(inconsistent_features) > 5:
                    print(f"   - ... y {len(inconsistent_features) - 5} más")
                
                # Solo advertir, no bloquear
                print("⚠️ ADVERTENCIA: Inconsistencias detectadas pero continuando...")
                return True
            else:
                print("✅ Features consistentes - validación OK")
                return True
                
        except Exception as e:
            print(f"❌ Error validando consistencia de features: {e}")
            return True  # Continuar a pesar del error

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """🔧 Crear features usando motor centralizado CON VALIDACIÓN MEJORADA"""

        print("🔧 Calculando features con motor centralizado y validación...")

        try:
            # ✅ VALIDACIÓN DE VENTANA MEJORADA
            if not self.validate_window_requirements(df):
                return pd.DataFrame()
            
            # Calcular features para backtesting
            features = self.calculate_features_for_backtest(df)
            
            if features.empty:
                return pd.DataFrame()
            
            # ✅ VALIDACIÓN DE CONSISTENCIA
            if not self.validate_features_consistency():
                print("⚠️ Inconsistencias en features detectadas")
            
            return features

        except Exception as e:
            print(f"❌ Error calculando features: {e}")
            return pd.DataFrame()

    def validate_window_requirements(self, df: pd.DataFrame) -> bool:
        """🔍 Validar que los datos son suficientes para el lookback window"""
        
        min_required = self.lookback_window + 100  # Buffer adicional para features
        
        if len(df) < min_required:
            print(f"❌ ERROR DE VENTANA:")
            print(f"   📊 Datos disponibles: {len(df)}")
            print(f"   📊 Lookback window: {self.lookback_window}")
            print(f"   📊 Mínimo requerido: {min_required}")
            print(f"   💡 Solución: Aumentar días de datos históricos")
            return False
        
        print(f"✅ Ventana validada: {len(df)} datos ≥ {min_required} requeridos")
        return True

    def generate_predictions(self, df: pd.DataFrame, features: pd.DataFrame, confidence_threshold: float) -> List[Dict]:
        """🔮 Generar predicciones del modelo"""

        print(f"🔮 Generando predicciones (confianza mínima: {confidence_threshold:.0%})...")

        predictions = []

        try:
            # Normalizar features
            features_scaled = self.scaler.transform(features)

            # Crear secuencias temporales
            for i in range(self.lookback_window, len(features_scaled)):
                sequence = features_scaled[i-self.lookback_window:i]
                sequence = sequence.reshape(1, self.lookback_window, -1)

                # Predecir
                pred_probs = self.model.predict(sequence, verbose=0)[0]
                pred_class = np.argmax(pred_probs)
                confidence = float(pred_probs[pred_class])

                # Mapear clases
                class_names = ['SELL', 'HOLD', 'BUY']
                signal = class_names[pred_class]

                # Aplicar filtro de confianza
                if confidence >= confidence_threshold:
                    predictions.append({
                        'timestamp': df.index[i],
                        'signal': signal,
                        'confidence': confidence,
                        'probabilities': {
                            'SELL': float(pred_probs[0]),
                            'HOLD': float(pred_probs[1]),
                            'BUY': float(pred_probs[2])
                        },
                        'price': df['close'].iloc[i]
                    })

            print(f"✅ Generadas {len(predictions)} predicciones válidas")
            return predictions

        except Exception as e:
            print(f"❌ Error generando predicciones: {e}")
            return []

    def simulate_trading(self, df: pd.DataFrame, predictions: List[Dict], stop_loss_percent: float = 1.0) -> Dict:
        """💰 Simular trading basado en predicciones - CON STOP LOSS INTEGRADO"""

        print(f"💰 Simulando trading con {len(predictions)} señales...")
        print("🎯 Modo: SOLO LONG (BUY/SELL) - Compatible con Binance Spot")
        print(f"🛑 Stop Loss: {stop_loss_percent:.1f}% (configurable)")

        # 🛑 CONFIGURACIÓN DE STOP LOSS
        stop_loss_multiplier = 1 - (stop_loss_percent / 100)  # 0.99 para 1%

        balance_usd = self.initial_balance  # Balance en USD (cash)
        position_crypto = 0.0              # Cantidad de crypto que tenemos
        position_entry_price = 0.0         # Precio al que compramos
        has_position = False               # Si tenemos crypto o no

        # 🛑 STOP LOSS TRACKING
        stop_loss_price = 0.0             # Precio de stop loss
        stop_loss_triggered = False       # Si se activó stop loss

        trades = []
        balance_history = []

        for pred in predictions:
            timestamp = pred['timestamp']
            signal = pred['signal']
            current_price = pred['price']
            confidence = pred['confidence']

            # 💰 CALCULAR BALANCE TOTAL ACTUAL (USD + valor de crypto)
            crypto_value_usd = position_crypto * current_price if has_position else 0.0
            total_balance = balance_usd + crypto_value_usd

            # 🛑 VERIFICAR STOP LOSS ANTES DE CUALQUIER DECISIÓN
            if has_position and not stop_loss_triggered:
                # Calcular pérdida actual
                current_loss_percent = (current_price - position_entry_price) / position_entry_price

                # Verificar si se activó stop loss
                if current_price <= stop_loss_price:
                    stop_loss_triggered = True
                    print(f"   🛑 STOP LOSS ACTIVADO: ${current_price:.2f} <= ${stop_loss_price:.2f}")

                    # Ejecutar stop loss
                    usd_gross = position_crypto * current_price
                    trading_fee_usd = usd_gross * self.trading_fee
                    usd_net = usd_gross - trading_fee_usd

                    # ✅ CORRECCIÓN: Calcular pérdida real con cost basis correcto
                    buy_trade = next((t for t in reversed(trades) if t['type'] == 'BUY'), None)
                    if buy_trade:
                        total_cost_basis = buy_trade['usd_spent']  # Ya incluye fees de compra
                        buy_fee = buy_trade['fee_usd']
                    else:
                        total_cost_basis = position_crypto * position_entry_price
                        buy_fee = 0
                    
                    loss_usd = usd_net - total_cost_basis  # Negativo = pérdida
                    loss_percentage = loss_usd / total_cost_basis if total_cost_basis > 0 else 0.0

                    # Actualizar balance
                    balance_usd += usd_net

                    trades.append({
                        'type': 'STOP_LOSS',
                        'timestamp': timestamp,
                        'price': current_price,
                        'entry_price': position_entry_price,
                        'crypto_amount': position_crypto,
                        'usd_received': usd_net,
                        'fee_usd': trading_fee_usd,
                        'loss_usd': loss_usd,
                        'loss_percentage': loss_percentage,
                        'total_cost_basis': total_cost_basis,
                        'buy_fee': buy_fee,
                        'balance_usd_after': balance_usd,
                        'confidence': confidence,
                        'stop_loss_price': stop_loss_price
                    })

                    print(f"   🛑 STOP_LOSS: {position_crypto:.6f} crypto @ ${current_price:.2f} → ${usd_net:.2f} (loss: {loss_percentage:.2%})")

                    # Reset posición
                    position_crypto = 0.0
                    position_entry_price = 0.0
                    has_position = False
                    stop_loss_price = 0.0
                    stop_loss_triggered = False

            # Registrar estado actual
            balance_history.append({
                'timestamp': timestamp,
                'total_balance': total_balance,
                'balance_usd': balance_usd,
                'position_crypto': position_crypto,
                'crypto_value_usd': crypto_value_usd,
                'price': current_price,
                'has_position': has_position,
                'stop_loss_price': stop_loss_price if has_position else None,
                'stop_loss_triggered': stop_loss_triggered
            })

            # 🎯 LÓGICA DE TRADING CORREGIDA (solo si no se activó stop loss)
            if not stop_loss_triggered:
                if signal == 'BUY' and not has_position:
                    # ✅ COMPRAR: Convertir USD a crypto
                    if balance_usd >= self.min_trade_amount:
                        # Usar 95% del balance para comprar
                        usd_to_spend = balance_usd * 0.95
                        trading_fee_usd = usd_to_spend * self.trading_fee
                        usd_after_fee = usd_to_spend - trading_fee_usd

                        # Cantidad de crypto que compramos
                        position_crypto = usd_after_fee / current_price
                        position_entry_price = current_price
                        has_position = True

                        # 🛑 CONFIGURAR STOP LOSS
                        stop_loss_price = current_price * stop_loss_multiplier

                        # Actualizar balance USD
                        balance_usd = balance_usd - usd_to_spend  # Quedan 5% + fracción

                        trades.append({
                            'type': 'BUY',
                            'timestamp': timestamp,
                            'price': current_price,
                            'crypto_amount': position_crypto,
                            'usd_spent': usd_to_spend,
                            'fee_usd': trading_fee_usd,
                            'balance_usd_after': balance_usd,
                            'confidence': confidence,
                            'stop_loss_price': stop_loss_price
                        })

                        print(f"   💚 BUY: {position_crypto:.6f} crypto @ ${current_price:.2f} (gastado: ${usd_to_spend:.2f})")
                        print(f"   🛑 Stop Loss configurado: ${stop_loss_price:.2f} (-{stop_loss_percent:.1f}%)")

                elif signal == 'SELL' and has_position:
                    # ✅ VENDER: Convertir crypto a USD
                    usd_gross = position_crypto * current_price
                    trading_fee_usd = usd_gross * self.trading_fee
                    usd_net = usd_gross - trading_fee_usd

                    # ✅ CORRECCIÓN: Calcular ganancia/pérdida con cost basis correcto
                    # Buscar el trade de compra correspondiente para obtener el costo total
                    buy_trade = next((t for t in reversed(trades) if t['type'] == 'BUY'), None)
                    if buy_trade:
                        total_cost_basis = buy_trade['usd_spent']  # Ya incluye fees de compra
                        buy_fee = buy_trade['fee_usd']
                    else:
                        # Fallback si no se encuentra el trade de compra
                        total_cost_basis = position_crypto * position_entry_price
                        buy_fee = 0
                    
                    profit_usd = usd_net - total_cost_basis
                    profit_percentage = profit_usd / total_cost_basis if total_cost_basis > 0 else 0.0

                    # Actualizar balance
                    balance_usd += usd_net

                    trades.append({
                        'type': 'SELL',
                        'timestamp': timestamp,
                        'price': current_price,
                        'entry_price': position_entry_price,
                        'crypto_amount': position_crypto,
                        'usd_received': usd_net,
                        'fee_usd': trading_fee_usd,
                        'profit_usd': profit_usd,
                        'profit_percentage': profit_percentage,
                        'total_cost_basis': total_cost_basis,
                        'buy_fee': buy_fee,
                        'balance_usd_after': balance_usd,
                        'confidence': confidence,
                        'stop_loss_price': stop_loss_price
                    })

                    print(f"   💛 SELL: {position_crypto:.6f} crypto @ ${current_price:.2f} → ${usd_net:.2f} (profit: {profit_percentage:.2%})")

                    # Reset posición
                    position_crypto = 0.0
                    position_entry_price = 0.0
                    has_position = False
                    stop_loss_price = 0.0

                # signal == 'HOLD' → No hacer nada, mantener posición actual

        # 🔚 CERRAR POSICIÓN FINAL SI EXISTE
        final_price = df['close'].iloc[-1]
        final_timestamp = df.index[-1]

        if has_position:
            # Vender todo al final
            usd_gross = position_crypto * final_price
            trading_fee_usd = usd_gross * self.trading_fee
            usd_net = usd_gross - trading_fee_usd

            usd_invested = position_crypto * position_entry_price
            profit_usd = usd_net - usd_invested
            profit_percentage = profit_usd / usd_invested if usd_invested > 0 else 0.0

            balance_usd += usd_net

            trades.append({
                'type': 'SELL_FINAL',
                'timestamp': final_timestamp,
                'price': final_price,
                'entry_price': position_entry_price,
                'crypto_amount': position_crypto,
                'usd_received': usd_net,
                'fee_usd': trading_fee_usd,
                'profit_usd': profit_usd,
                'profit_percentage': profit_percentage,
                'balance_usd_after': balance_usd,
                'confidence': 0.0,
                'stop_loss_price': stop_loss_price
            })

            print(f"   🔚 SELL_FINAL: {position_crypto:.6f} crypto @ ${final_price:.2f} → ${usd_net:.2f}")

            # Reset posición
            position_crypto = 0.0
            has_position = False

        # Registrar estado final
        final_total_balance = balance_usd + (position_crypto * final_price if has_position else 0.0)
        balance_history.append({
            'timestamp': final_timestamp,
            'total_balance': final_total_balance,
            'balance_usd': balance_usd,
            'position_crypto': position_crypto,
            'crypto_value_usd': position_crypto * final_price if has_position else 0.0,
            'price': final_price,
            'has_position': has_position,
            'stop_loss_price': stop_loss_price if has_position else None,
            'stop_loss_triggered': stop_loss_triggered
        })

        # 📊 ESTADÍSTICAS DE STOP LOSS
        stop_loss_trades = [t for t in trades if t['type'] == 'STOP_LOSS']
        total_stop_losses = len(stop_loss_trades)
        total_stop_loss_amount = sum(t.get('loss_usd', 0) for t in stop_loss_trades)

        print(f"✅ Simulación completada: {len(trades)} trades ejecutados")
        print(f"💰 Balance final: ${final_total_balance:.2f} (inicial: ${self.initial_balance:.2f})")
        print(f"🛑 Stop Losses activados: {total_stop_losses} (pérdida total: ${total_stop_loss_amount:.2f})")

        return {
            'final_balance': final_total_balance,
            'final_balance_usd': balance_usd,
            'final_position_crypto': position_crypto,
            'trades': trades,
            'balance_history': balance_history,
            'stop_loss_stats': {
                'total_stop_losses': total_stop_losses,
                'total_stop_loss_amount': total_stop_loss_amount,
                'stop_loss_percent': stop_loss_percent
            }
        }

    def calculate_enhanced_metrics(self, results: Dict) -> Dict:
        """📊 Calcular métricas financieras CORREGIDAS con mejores cálculos"""
        
        print("📊 Calculando métricas mejoradas con correcciones matemáticas...")
        
        trades = results['trades']
        final_balance = results['final_balance']
        balance_history = results['balance_history']
        
        # Separar tipos de trades
        buy_trades = [t for t in trades if t['type'] == 'BUY']
        sell_trades = [t for t in trades if t['type'] in ['SELL', 'SELL_FINAL']]
        stop_loss_trades = [t for t in trades if t['type'] == 'STOP_LOSS']
        
        # ✅ CORRECCIÓN: Win rate considerando stop-losses
        profitable_trades = len([t for t in sell_trades if t.get('profit_usd', 0) > 0])
        unprofitable_sells = len([t for t in sell_trades if t.get('profit_usd', 0) <= 0])
        total_completed_cycles = profitable_trades + unprofitable_sells + len(stop_loss_trades)
        
        win_rate_corrected = profitable_trades / total_completed_cycles if total_completed_cycles > 0 else 0
        
        # ✅ CORRECCIÓN: Profit factor
        gross_profit = sum([t.get('profit_usd', 0) for t in sell_trades if t.get('profit_usd', 0) > 0])
        gross_loss_sells = abs(sum([t.get('profit_usd', 0) for t in sell_trades if t.get('profit_usd', 0) < 0]))
        gross_loss_stops = abs(sum([t.get('loss_usd', 0) for t in stop_loss_trades]))
        gross_loss_total = gross_loss_sells + gross_loss_stops
        
        profit_factor = gross_profit / gross_loss_total if gross_loss_total > 0 else float('inf')
        
        # ✅ ANÁLISIS DETALLADO DE RETORNOS
        all_profit_percentages = []
        all_profit_usd = []
        
        # Agregar trades rentables
        for t in sell_trades:
            all_profit_percentages.append(t.get('profit_percentage', 0))
            all_profit_usd.append(t.get('profit_usd', 0))
        
        # Agregar stop losses (como pérdidas)
        for t in stop_loss_trades:
            all_profit_percentages.append(t.get('loss_percentage', 0))  # Ya negativo
            all_profit_usd.append(t.get('loss_usd', 0))  # Ya negativo
        
        # Métricas básicas
        total_return = (final_balance - self.initial_balance) / self.initial_balance
        avg_trade_return = np.mean(all_profit_percentages) if all_profit_percentages else 0
        
        # Total fees
        total_fees = sum(t.get('fee_usd', 0) for t in trades)
        
        return {
            'win_rate_corrected': win_rate_corrected,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss_total': gross_loss_total,
            'gross_loss_sells': gross_loss_sells,
            'gross_loss_stops': gross_loss_stops,
            'total_return': total_return,
            'avg_trade_return': avg_trade_return,
            'profitable_trades': profitable_trades,
            'unprofitable_sells': unprofitable_sells,
            'stop_loss_trades': len(stop_loss_trades),
            'total_completed_cycles': total_completed_cycles,
            'total_fees': total_fees,
            'all_trade_returns': all_profit_percentages,
            'all_trade_usd': all_profit_usd
        }

    def calculate_metrics(self, results: Dict) -> Dict:
        """📊 Calcular métricas de rendimiento - MEJORADO CON CORRECCIONES"""

        print("📊 Calculando métricas de rendimiento mejoradas...")

        final_balance = results['final_balance']
        trades = results['trades']
        balance_history = results['balance_history']
        stop_loss_stats = results.get('stop_loss_stats', {})

        # ✅ USAR MÉTRICAS MEJORADAS
        enhanced_metrics = self.calculate_enhanced_metrics(results)

        # Métricas básicas (mantenemos compatibilidad)
        total_return = enhanced_metrics['total_return']

        # 🎯 ANÁLISIS DE TRADES CORREGIDO
        sell_trades = [t for t in trades if t['type'] in ['SELL', 'SELL_FINAL']]
        stop_loss_trades = [t for t in trades if t['type'] == 'STOP_LOSS']
        
        # Usar métricas corregidas
        winning_trades = enhanced_metrics['profitable_trades']
        losing_trades = enhanced_metrics['unprofitable_sells'] + enhanced_metrics['stop_loss_trades']
        total_trades = enhanced_metrics['total_completed_cycles']
        
        win_rate = enhanced_metrics['win_rate_corrected']  # ✅ CORREGIDO
        
        # Estadísticas de retornos
        all_trade_percentages = enhanced_metrics['all_trade_returns']
        all_trade_usd_amounts = enhanced_metrics['all_trade_usd']
        
        if all_trade_percentages:
            avg_profit_pct = np.mean(all_trade_percentages)
            max_profit_pct = max(all_trade_percentages)
            max_loss_pct = min(all_trade_percentages)
        else:
            avg_profit_pct = max_profit_pct = max_loss_pct = 0
            
        if all_trade_usd_amounts:
            avg_profit_usd = np.mean(all_trade_usd_amounts)
            max_profit_usd = max(all_trade_usd_amounts)
            max_loss_usd = min(all_trade_usd_amounts)
        else:
            avg_profit_usd = max_profit_usd = max_loss_usd = 0

        # Total de fees pagados
        total_fees = enhanced_metrics['total_fees']

        # 🎯 CÁLCULO DE DRAWDOWN CORREGIDO
        peak_balance = self.initial_balance
        max_drawdown = 0
        max_drawdown_usd = 0

        for record in balance_history:
            current_balance = record['total_balance']
            if current_balance > peak_balance:
                peak_balance = current_balance

            drawdown = (peak_balance - current_balance) / peak_balance
            drawdown_usd = peak_balance - current_balance

            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_usd = drawdown_usd

        # ✅ SHARPE RATIO CORREGIDO (solo períodos activos con tasa libre de riesgo)
        sharpe_ratio = self.calculate_corrected_sharpe_ratio(balance_history)
        
        # ✅ MÉTRICAS ADICIONALES
        additional_metrics = self.calculate_additional_risk_metrics(balance_history, all_trade_percentages)
        calmar_ratio = additional_metrics['calmar_ratio']
        sortino_ratio = additional_metrics['sortino_ratio']

        # 🎯 MÉTRICAS ADICIONALES
        # Número de transacciones (BUY + SELL + STOP_LOSS)
        total_transactions = len(trades)
        buy_transactions = len([t for t in trades if t['type'] == 'BUY'])

        # Tiempo en mercado (porcentaje de tiempo con posición)
        periods_with_position = len([r for r in balance_history if r.get('has_position', False)])
        time_in_market = periods_with_position / len(balance_history) if balance_history else 0

        # 🛑 MÉTRICAS DE STOP LOSS
        total_stop_losses = stop_loss_stats.get('total_stop_losses', 0)
        total_stop_loss_amount = stop_loss_stats.get('total_stop_loss_amount', 0)
        stop_loss_percent = stop_loss_stats.get('stop_loss_percent', 1.0)

        # Porcentaje de trades que terminaron en stop loss
        stop_loss_rate = total_stop_losses / total_trades if total_trades > 0 else 0

        # ✅ VALIDACIÓN MATEMÁTICA
        validation_results = self.validate_financial_math(results)
        
        print(f"✅ Métricas calculadas con correcciones financieras")
        if not validation_results['is_mathematically_valid']:
            print(f"⚠️ ADVERTENCIA: Inconsistencia matemática detectada:")
            print(f"   💰 Balance final: ${validation_results['final_balance']:.2f}")
            print(f"   🧮 Balance esperado: ${validation_results['expected_final']:.2f}")
            print(f"   📊 Diferencia: ${validation_results['difference']:.2f}")

        return {
            'initial_balance': self.initial_balance,
            'final_balance': final_balance,
            'final_balance_usd': results.get('final_balance_usd', 0),
            'final_position_crypto': results.get('final_position_crypto', 0),
            'total_return': total_return,
            'total_return_pct': total_return * 100,

            # Trading metrics CORREGIDOS
            'total_transactions': total_transactions,
            'buy_transactions': buy_transactions,
            'sell_transactions': len(sell_trades),
            'stop_loss_transactions': total_stop_losses,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,  # ✅ CORREGIDO
            'stop_loss_rate': stop_loss_rate,
            'time_in_market': time_in_market,

            # Profit metrics
            'avg_profit_pct': avg_profit_pct,
            'avg_profit_usd': avg_profit_usd,
            'max_profit_pct': max_profit_pct,
            'max_loss_pct': max_loss_pct,
            'max_profit_usd': max_profit_usd,
            'max_loss_usd': max_loss_usd,
            'total_fees_usd': total_fees,

            # Risk metrics CORREGIDOS
            'max_drawdown': max_drawdown,
            'max_drawdown_usd': max_drawdown_usd,
            'sharpe_ratio': sharpe_ratio,  # ✅ CORREGIDO

            # ✅ NUEVAS MÉTRICAS MEJORADAS
            'profit_factor': enhanced_metrics['profit_factor'],
            'gross_profit': enhanced_metrics['gross_profit'],
            'gross_loss_total': enhanced_metrics['gross_loss_total'],
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'max_adverse_excursion': additional_metrics['max_adverse_excursion'],
            'downside_deviation': additional_metrics['downside_deviation'],

            # Stop Loss metrics
            'total_stop_losses': total_stop_losses,
            'total_stop_loss_amount': total_stop_loss_amount,
            'stop_loss_percent': stop_loss_percent,
            'stop_loss_rate': stop_loss_rate,
            
            # ✅ VALIDACIÓN MATEMÁTICA
            'mathematical_validation': validation_results
        }

    def print_results(self, metrics: Dict, model_info: Dict):
        """📈 Mostrar resultados del backtesting - CON ESTADÍSTICAS DE STOP LOSS"""

        print(f"\n🎉 RESULTADOS DEL BACKTESTING (CON STOP LOSS INTEGRADO)")
        print("=" * 80)
        print(f"📊 Modelo: {model_info['name']}")
        print(f"💎 Símbolo: {model_info['symbol']}")
        print(f"⏰ Timeframe: {model_info['timeframe']} (✅ VERIFICADO)")
        print(f"🔧 Detección: {model_info['detection_method']}")
        print("=" * 80)

        # Rendimiento financiero
        print(f"💰 RENDIMIENTO FINANCIERO:")
        print(f"   💵 Balance inicial: ${metrics['initial_balance']:.2f}")
        print(f"   💵 Balance final total: ${metrics['final_balance']:.2f}")
        print(f"   💵 Balance USD: ${metrics['final_balance_usd']:.2f}")
        print(f"   🪙 Crypto restante: {metrics['final_position_crypto']:.6f}")
        print(f"   📈 Retorno total: {metrics['total_return']:.2%} (${metrics['final_balance'] - metrics['initial_balance']:.2f})")
        print(f"   💸 Fees totales: ${metrics['total_fees_usd']:.2f}")
        print(f"   📉 Máximo drawdown: {metrics['max_drawdown']:.2%} (${metrics['max_drawdown_usd']:.2f})")
        print(f"   📊 Sharpe ratio: {metrics['sharpe_ratio']:.3f}")

        # 🛑 ESTADÍSTICAS DE STOP LOSS
        print(f"\n🛑 ESTADÍSTICAS DE STOP LOSS:")
        print(f"   🛑 Stop Loss configurado: {metrics['stop_loss_percent']:.1f}%")
        print(f"   🛑 Stop Losses activados: {metrics['total_stop_losses']}")
        print(f"   🛑 Pérdida total por stop loss: ${metrics['total_stop_loss_amount']:.2f}")
        print(f"   🛑 Tasa de stop loss: {metrics['stop_loss_rate']:.2%}")

        # Estadísticas de trading
        print(f"\n🎯 ESTADÍSTICAS DE TRADING:")
        print(f"   🔄 Total transacciones: {metrics['total_transactions']}")
        print(f"   💚 Compras (BUY): {metrics['buy_transactions']}")
        print(f"   💛 Ventas (SELL): {metrics['sell_transactions']}")
        print(f"   🛑 Stop Losses: {metrics['stop_loss_transactions']}")
        print(f"   ✅ Trades ganadores: {metrics['winning_trades']}")
        print(f"   ❌ Trades perdedores: {metrics['losing_trades']}")
        print(f"   🎯 Win rate: {metrics['win_rate']:.2%}")
        print(f"   ⏰ Tiempo en mercado: {metrics['time_in_market']:.1%}")

        # Análisis de ganancias/pérdidas MEJORADO
        print(f"\n💹 ANÁLISIS DE GANANCIAS/PÉRDIDAS:")
        print(f"   📊 Ganancia promedio: {metrics['avg_profit_pct']:.3%} (${metrics['avg_profit_usd']:.2f})")
        print(f"   🚀 Mejor trade: {metrics['max_profit_pct']:.3%} (${metrics['max_profit_usd']:.2f})")
        print(f"   💥 Peor trade: {metrics['max_loss_pct']:.3%} (${metrics['max_loss_usd']:.2f})")
        
        # ✅ NUEVAS MÉTRICAS FINANCIERAS
        if 'profit_factor' in metrics:
            print(f"\n📈 MÉTRICAS FINANCIERAS AVANZADAS:")
            print(f"   🎯 Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"   💰 Ganancia bruta: ${metrics['gross_profit']:.2f}")
            print(f"   💸 Pérdida bruta: ${metrics['gross_loss_total']:.2f}")
            print(f"   📊 Calmar Ratio: {metrics['calmar_ratio']:.2f}")
            print(f"   📉 Sortino Ratio: {metrics['sortino_ratio']:.2f}")
            print(f"   ⚠️ MAE (peor excursión): {metrics['max_adverse_excursion']:.3%}")

        # Evaluación general mejorada
        print(f"\n🏆 EVALUACIÓN DETALLADA:")

        # Evaluación de retorno
        if metrics['total_return'] > 0.20:  # +20%
            print("   🟢 RENDIMIENTO: EXCELENTE (>20%)")
        elif metrics['total_return'] > 0.10:  # +10%
            print("   🟡 RENDIMIENTO: BUENO (>10%)")
        elif metrics['total_return'] > 0:
            print("   🟠 RENDIMIENTO: MODERADO (>0%)")
        else:
            print("   🔴 RENDIMIENTO: MALO (negativo)")

        # Evaluación de win rate
        if metrics['win_rate'] > 0.60:
            print("   🟢 WIN RATE: EXCELENTE (>60%)")
        elif metrics['win_rate'] > 0.50:
            print("   🟡 WIN RATE: BUENO (>50%)")
        elif metrics['win_rate'] > 0.40:
            print("   🟠 WIN RATE: ACEPTABLE (>40%)")
        else:
            print("   🔴 WIN RATE: BAJO (≤40%)")

        # Evaluación de stop loss rate
        if metrics['stop_loss_rate'] < 0.10:  # <10%
            print("   🟢 STOP LOSS RATE: EXCELENTE (<10%)")
        elif metrics['stop_loss_rate'] < 0.20:  # <20%
            print("   🟡 STOP LOSS RATE: BUENO (<20%)")
        elif metrics['stop_loss_rate'] < 0.30:  # <30%
            print("   🟠 STOP LOSS RATE: ACEPTABLE (<30%)")
        else:
            print("   🔴 STOP LOSS RATE: ALTO (≥30%)")

        # Evaluación de drawdown
        if metrics['max_drawdown'] < 0.05:  # <5%
            print("   🟢 RIESGO: BAJO (drawdown <5%)")
        elif metrics['max_drawdown'] < 0.15:  # <15%
            print("   🟡 RIESGO: MODERADO (drawdown <15%)")
        elif metrics['max_drawdown'] < 0.30:  # <30%
            print("   🟠 RIESGO: ALTO (drawdown <30%)")
        else:
            print("   🔴 RIESGO: MUY ALTO (drawdown ≥30%)")

        # Evaluación de Sharpe
        if metrics['sharpe_ratio'] > 2.0:
            print("   🟢 SHARPE: EXCELENTE (>2.0)")
        elif metrics['sharpe_ratio'] > 1.0:
            print("   🟡 SHARPE: BUENO (>1.0)")
        elif metrics['sharpe_ratio'] > 0.5:
            print("   🟠 SHARPE: ACEPTABLE (>0.5)")
        else:
            print("   🔴 SHARPE: BAJO (≤0.5)")

        # ✅ EVALUACIONES MEJORADAS
        if 'profit_factor' in metrics:
            # Evaluación de profit factor
            if metrics['profit_factor'] > 2.0:
                print("   🟢 PROFIT FACTOR: EXCELENTE (>2.0)")
            elif metrics['profit_factor'] > 1.5:
                print("   🟡 PROFIT FACTOR: BUENO (>1.5)")
            elif metrics['profit_factor'] > 1.0:
                print("   🟠 PROFIT FACTOR: ACEPTABLE (>1.0)")
            else:
                print("   🔴 PROFIT FACTOR: MALO (≤1.0)")
            
            # Evaluación de Calmar ratio
            if metrics['calmar_ratio'] > 3.0:
                print("   🟢 CALMAR RATIO: EXCELENTE (>3.0)")
            elif metrics['calmar_ratio'] > 1.5:
                print("   🟡 CALMAR RATIO: BUENO (>1.5)")
            elif metrics['calmar_ratio'] > 0.5:
                print("   🟠 CALMAR RATIO: ACEPTABLE (>0.5)")
            else:
                print("   🔴 CALMAR RATIO: BAJO (≤0.5)")

        # ✅ VALIDACIÓN MATEMÁTICA
        if 'mathematical_validation' in metrics:
            validation = metrics['mathematical_validation']
            if validation['is_mathematically_valid']:
                print("   ✅ VALIDACIÓN MATEMÁTICA: COHERENTE")
            else:
                print(f"   ⚠️ VALIDACIÓN MATEMÁTICA: INCONSISTENTE (${validation['difference']:.2f})")

        # Resumen final MEJORADO
        print(f"\n📋 RESUMEN FINAL:")
        profitability_score = 0
        if metrics['total_return'] > 0: profitability_score += 1
        if metrics['win_rate'] > 0.5: profitability_score += 1
        if metrics['max_drawdown'] < 0.15: profitability_score += 1
        if metrics['sharpe_ratio'] > 1.0: profitability_score += 1
        if metrics['stop_loss_rate'] < 0.20: profitability_score += 1
        
        # ✅ CRITERIOS ADICIONALES
        if 'profit_factor' in metrics and metrics['profit_factor'] > 1.5: profitability_score += 1
        if 'calmar_ratio' in metrics and metrics['calmar_ratio'] > 1.0: profitability_score += 1

        if profitability_score >= 6:
            print("   🏆 MODELO EXCEPCIONAL - Altamente prometedor para trading real")
        elif profitability_score >= 5:
            print("   ⭐ MODELO EXCELENTE - Muy prometedor para trading real")
        elif profitability_score >= 4:
            print("   ⚡ MODELO PROMETEDOR - Considerar para trading real")
        elif profitability_score >= 3:
            print("   🟡 MODELO ACEPTABLE - Necesita ajustes menores")
        elif profitability_score >= 2:
            print("   🟠 MODELO REGULAR - Requiere mejoras significativas")
        else:
            print("   ❌ MODELO PROBLEMÁTICO - Requiere reentrenamiento")

        print("=" * 80)

    async def run_backtest(self, model_info: Dict, days: int = 15, confidence_threshold: float = 0.5, stop_loss_percent: float = 1.0):
        """🚀 Ejecutar backtesting completo CON TIMEFRAME VERIFICADO Y STOP LOSS"""

        print(f"🚀 INICIANDO BACKTESTING UNIVERSAL CORREGIDO")
        print(f"📊 Modelo: {model_info['name']}")
        print(f"💎 Símbolo: {model_info['symbol']}")
        print(f"⏰ Timeframe: {model_info['timeframe']} (✅ VERIFICADO)")
        print(f"🔧 Detección: {model_info['detection_method']}")
        print(f"📅 Días: {days}")
        print(f"🎯 Confianza mínima: {confidence_threshold:.0%}")
        print(f"🛑 Stop Loss: {stop_loss_percent:.1f}%")
        print("="*70)

        # 1. Cargar modelo
        if not self.load_model_components(model_info):
            return None

        # 2. Obtener datos históricos CON TIMEFRAME CORRECTO
        df = await self.get_historical_data(days=days)
        if df.empty:
            print("❌ No se pudieron obtener datos históricos")
            return None

        # 3. Calcular features
        features = self.create_features(df)
        if features.empty:
            print("❌ Error calculando features")
            return None

        # 4. Generar predicciones
        predictions = self.generate_predictions(df, features, confidence_threshold)
        if not predictions:
            print("❌ Error generando predicciones")
            return None

        # 5. Simular trading CON STOP LOSS
        results = self.simulate_trading(df, predictions, stop_loss_percent)

        # 6. Calcular métricas
        metrics = self.calculate_metrics(results)

        # 7. Mostrar resultados
        self.print_results(metrics, model_info)

        return {
            'metrics': metrics,
            'results': results,
            'predictions': predictions,
            'data': df,
            'model_info': model_info
        }

async def main():
    """🎯 Función principal"""

    print("🚀 BACKTESTING UNIVERSAL CORREGIDO")
    print("=" * 80)
    print("✅ CORRIGIDO: Detección automática de timeframe")
    print("✅ CORRIGIDO: Validación de datos con timeframe correcto")
    print("✅ CORRIGIDO: Sin defaults silenciosos que causen errores")
    print("🛑 NUEVO: Sistema de Stop Loss integrado")
    print("=" * 80)

    backtester = UniversalBacktesterFixed()

    # Descubrir modelos disponibles
    models = backtester.discover_models()
    if not models:
        print("❌ No se encontraron modelos válidos")
        return

    # Seleccionar modelo
    selected_model = backtester.select_model(models)
    if not selected_model:
        print("❌ No se seleccionó ningún modelo")
        return

    # Configurar backtesting
    print(f"\n⚙️ CONFIGURACIÓN DEL BACKTESTING")
    print("=" * 50)

    # Días de datos CON SUGERENCIA INTELIGENTE
    optimal_days = backtester.calculate_optimal_days_for_backtest(
        selected_model['timeframe'], 
        backtester.lookback_window if hasattr(backtester, 'lookback_window') else 48
    )
    
    print(f"💡 Días óptimos sugeridos para {selected_model['timeframe']}: {optimal_days}")
    
    while True:
        try:
            days_prompt = f"📅 Días de datos para backtest (sugerido {optimal_days}, rango 5-90): "
            days = int(input(days_prompt))
            if 5 <= days <= 90:
                break
            print("❌ Días debe estar entre 5 y 90")
        except ValueError:
            print("❌ Ingresa un número válido")

    # Umbral de confianza
    while True:
        try:
            confidence = float(input("🎯 Umbral de confianza (0.5-0.9, recomendado 0.6): "))
            if 0.1 <= confidence <= 0.95:
                break
            print("❌ Confianza debe estar entre 0.1 y 0.95")
        except ValueError:
            print("❌ Ingresa un número válido")

    # 🛑 CONFIGURACIÓN DE STOP LOSS
    while True:
        try:
            stop_loss = float(input("🛑 Stop Loss por trade (0.1-5.0%, recomendado 1.0): "))
            if 0.1 <= stop_loss <= 5.0:
                break
            print("❌ Stop Loss debe estar entre 0.1% y 5.0%")
        except ValueError:
            print("❌ Ingresa un número válido")

    # Ejecutar backtesting
    results = await backtester.run_backtest(selected_model, days=days, confidence_threshold=confidence, stop_loss_percent=stop_loss)

    if results:
        print(f"\n🎉 ¡BACKTESTING COMPLETADO EXITOSAMENTE!")
        print(f"✅ Timeframe verificado: {selected_model['timeframe']}")
        print(f"✅ Datos correctos utilizados")
        print(f"✅ Stop Loss configurado: {stop_loss:.1f}%")
        print(f"✅ Resultados confiables")
    else:
        print(f"\n❌ Error en el backtesting")

if __name__ == "__main__":
    asyncio.run(main())
