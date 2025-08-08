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
from centralized_features_engine2 import CentralizedFeaturesEngine

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
        """🔧 MEJORADO: Extraer símbolo y timeframe con múltiples métodos"""

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

        # ✅ MÉTODO 1: Intentar leer desde metadatos guardados
        timeframe, method = self._try_load_timeframe_from_metadata(dir_path)
        if timeframe:
            return symbol, timeframe, f"metadata_{method}"

        # ✅ MÉTODO 2: Patrones regex mejorados
        timeframe = self._extract_timeframe_from_name(dir_name)
        if timeframe:
            return symbol, timeframe, "regex"

        # ✅ MÉTODO 3: Análisis del input shape del modelo (último recurso)
        timeframe = self._infer_timeframe_from_model(dir_path)
        if timeframe:
            return symbol, timeframe, "model_shape"

        # ❌ NO SE PUDO DETECTAR
        return symbol, None, "failed"

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

                # ✅ MOSTRAR MÉTODO DE DETECCIÓN
                detection_info = f"[{model['detection_method']}]"
                timeframe_info = f"⏰ {model['timeframe']}"

                print(f"   {model_index:2d}. {model['name']:30s} {timeframe_info} {size_info} {detection_info}")

        # Selección
        while True:
            try:
                choice = int(input(f"\n🎯 Selecciona modelo (1-{model_index}): "))
                if 1 <= choice <= model_index:
                    selected = index_to_model[choice]

                    # ✅ VALIDACIÓN ADICIONAL DE TIMEFRAME
                    print(f"\n✅ Modelo seleccionado: {selected['name']}")
                    print(f"📊 Símbolo: {selected['symbol']}")
                    print(f"⏰ Timeframe: {selected['timeframe']} (detectado via {selected['detection_method']})")
                    print(f"🔢 Parámetros: {selected['parameters']:,}")

                    # Confirmar timeframe
                    confirm = input(f"¿Confirmar que este modelo fue entrenado en {selected['timeframe']}? (s/n): ").lower().strip()
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

    def load_model_components(self, model_info: Dict) -> bool:
        """📂 Cargar componentes del modelo CON VALIDACIÓN DE TIMEFRAME"""

        try:
            print(f"📂 Cargando modelo {model_info['name']}...")

            self.model_path = model_info['path']
            self.symbol = model_info['symbol']
            self.timeframe = model_info['timeframe']

            # ✅ VALIDACIÓN: Asegurar que timeframe está definido
            if not self.timeframe:
                print("❌ Error: Timeframe no detectado. No se puede proceder.")
                return False

            print(f"✅ Timeframe confirmado: {self.timeframe}")

            # Cargar modelo
            model_file_path = os.path.join(self.model_path, model_info['model_file'])
            self.model = tf.keras.models.load_model(model_file_path)

            # 🔧 AUTO-DETECTAR LOOKBACK_WINDOW del modelo
            input_shape = self.model.input_shape
            print(f"🔢 Input shape detectado: {input_shape}")

            # ✅ CORRECCIÓN: Manejar modelos con entrada dinámica
            if len(input_shape) >= 2:
                detected_lookback = input_shape[1]
                if detected_lookback is None:
                    # Modelo con entrada dinámica - usar lookback_window por defecto
                    print(f"🔧 Modelo con entrada dinámica detectado")
                    print(f"🔧 Usando lookback_window por defecto: {self.lookback_window}")
                else:
                    # Modelo con entrada fija - auto-ajustar
                    if detected_lookback != self.lookback_window:
                        print(f"🔧 Auto-ajustando lookback_window: {self.lookback_window} → {detected_lookback}")
                        self.lookback_window = detected_lookback
            else:
                print(f"⚠️  Input shape inesperado: {input_shape}")

            print(f"✅ {model_info['model_file']} cargado ({self.model.count_params():,} parámetros)")
            print(f"🔢 Input shape: {input_shape}")
            print(f"⏰ Lookback window: {self.lookback_window} timesteps")
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

    async def get_historical_data(self, days: int = 30, limit: int = 1000) -> pd.DataFrame:
        """📊 Obtener datos históricos CON TIMEFRAME CORRECTO"""

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

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """🔧 Crear features usando motor centralizado"""

        print("🔧 Calculando features con motor centralizado...")

        try:
            # Usar motor centralizado de features
            features = self.features_engine.calculate_features(df)

            if features.empty:
                print("❌ Error: Features vacías")
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

            print(f"✅ Features calculadas: {len(available_features)}/{len(self.feature_columns)}")
            return features[available_features]

        except Exception as e:
            print(f"❌ Error calculando features: {e}")
            return pd.DataFrame()

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

                    # Calcular pérdida real
                    usd_invested = position_crypto * position_entry_price
                    loss_usd = usd_net - usd_invested
                    loss_percentage = loss_usd / usd_invested if usd_invested > 0 else 0.0

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

                    # Calcular ganancia/pérdida
                    usd_invested = position_crypto * position_entry_price
                    profit_usd = usd_net - usd_invested
                    profit_percentage = profit_usd / usd_invested if usd_invested > 0 else 0.0

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

    def calculate_metrics(self, results: Dict) -> Dict:
        """📊 Calcular métricas de rendimiento - CON ESTADÍSTICAS DE STOP LOSS"""

        print("📊 Calculando métricas de rendimiento...")

        final_balance = results['final_balance']
        trades = results['trades']
        balance_history = results['balance_history']
        stop_loss_stats = results.get('stop_loss_stats', {})

        # Métricas básicas
        total_return = (final_balance - self.initial_balance) / self.initial_balance

        # 🎯 ANÁLISIS DE TRADES CORREGIDO
        # Solo trades de SELL tienen profit (BUY son inversiones)
        sell_trades = [t for t in trades if t['type'] in ['SELL', 'SELL_FINAL']]
        stop_loss_trades = [t for t in trades if t['type'] == 'STOP_LOSS']
        total_trades = len(sell_trades) + len(stop_loss_trades)

        if total_trades > 0:
            # Usar profit_percentage para análisis (más relevante que profit_usd)
            profit_percentages = [t['profit_percentage'] for t in sell_trades]
            profit_usd_amounts = [t['profit_usd'] for t in sell_trades]

            # 🛑 ANÁLISIS DE STOP LOSS
            stop_loss_percentages = [t.get('loss_percentage', 0) for t in stop_loss_trades]
            stop_loss_usd_amounts = [t.get('loss_usd', 0) for t in stop_loss_trades]

            # Combinar trades normales y stop losses para análisis completo
            all_trade_percentages = profit_percentages + stop_loss_percentages
            all_trade_usd_amounts = profit_usd_amounts + stop_loss_usd_amounts

            winning_trades = len([p for p in profit_percentages if p > 0])
            losing_trades = len([p for p in profit_percentages if p <= 0])
            stop_loss_count = len(stop_loss_trades)

            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            avg_profit_pct = np.mean(all_trade_percentages) if all_trade_percentages else 0
            avg_profit_usd = np.mean(all_trade_usd_amounts) if all_trade_usd_amounts else 0
            max_profit_pct = max(all_trade_percentages) if all_trade_percentages else 0
            max_loss_pct = min(all_trade_percentages) if all_trade_percentages else 0
            max_profit_usd = max(all_trade_usd_amounts) if all_trade_usd_amounts else 0
            max_loss_usd = min(all_trade_usd_amounts) if all_trade_usd_amounts else 0

            # Total de fees pagados
            total_fees = sum(t.get('fee_usd', 0) for t in trades)
        else:
            winning_trades = losing_trades = stop_loss_count = 0
            win_rate = avg_profit_pct = avg_profit_usd = 0
            max_profit_pct = max_loss_pct = 0
            max_profit_usd = max_loss_usd = total_fees = 0

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

        # 🎯 SHARPE RATIO CORREGIDO (basado en balance total)
        if len(balance_history) > 1:
            daily_returns = []
            for i in range(1, len(balance_history)):
                prev_balance = balance_history[i-1]['total_balance']
                curr_balance = balance_history[i]['total_balance']
                if prev_balance > 0:
                    daily_return = (curr_balance - prev_balance) / prev_balance
                    daily_returns.append(daily_return)

            if daily_returns and np.std(daily_returns) > 0:
                avg_daily_return = np.mean(daily_returns)
                std_daily_return = np.std(daily_returns)
                # Anualizar (asumiendo 365 días)
                sharpe_ratio = (avg_daily_return / std_daily_return) * np.sqrt(365)
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0

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

        print("✅ Métricas calculadas con estadísticas de stop loss")

        return {
            'initial_balance': self.initial_balance,
            'final_balance': final_balance,
            'final_balance_usd': results.get('final_balance_usd', 0),
            'final_position_crypto': results.get('final_position_crypto', 0),
            'total_return': total_return,
            'total_return_pct': total_return * 100,

            # Trading metrics
            'total_transactions': total_transactions,
            'buy_transactions': buy_transactions,
            'sell_transactions': len(sell_trades),
            'stop_loss_transactions': total_stop_losses,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
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

            # Risk metrics
            'max_drawdown': max_drawdown,
            'max_drawdown_usd': max_drawdown_usd,
            'sharpe_ratio': sharpe_ratio,

            # Stop Loss metrics
            'total_stop_losses': total_stop_losses,
            'total_stop_loss_amount': total_stop_loss_amount,
            'stop_loss_percent': stop_loss_percent,
            'stop_loss_rate': stop_loss_rate
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

        # Análisis de ganancias/pérdidas
        print(f"\n💹 ANÁLISIS DE GANANCIAS/PÉRDIDAS:")
        print(f"   📊 Ganancia promedio: {metrics['avg_profit_pct']:.3%} (${metrics['avg_profit_usd']:.2f})")
        print(f"   🚀 Mejor trade: {metrics['max_profit_pct']:.3%} (${metrics['max_profit_usd']:.2f})")
        print(f"   💥 Peor trade: {metrics['max_loss_pct']:.3%} (${metrics['max_loss_usd']:.2f})")

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

        # Resumen final
        print(f"\n📋 RESUMEN:")
        profitability_score = 0
        if metrics['total_return'] > 0: profitability_score += 1
        if metrics['win_rate'] > 0.5: profitability_score += 1
        if metrics['max_drawdown'] < 0.15: profitability_score += 1
        if metrics['sharpe_ratio'] > 1.0: profitability_score += 1
        if metrics['stop_loss_rate'] < 0.20: profitability_score += 1  # Nueva métrica

        if profitability_score >= 4:
            print("   🏆 MODELO EXCELENTE - Muy prometedor para trading real")
        elif profitability_score >= 3:
            print("   ⚡ MODELO PROMETEDOR - Considerar para trading real")
        elif profitability_score >= 2:
            print("   🟡 MODELO ACEPTABLE - Necesita ajustes")
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

    # Días de datos
    while True:
        try:
            days = int(input("📅 Días de datos para backtest (recomendado 15-30): "))
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
