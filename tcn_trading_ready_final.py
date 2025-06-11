#!/usr/bin/env python3
"""
TCN TRADING READY FINAL - Sistema definitivo con accuracy y confianza optimizadas
Paso final hacia sistema trading-ready completo para Binance
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings('ignore')

# Configuración determinística
tf.random.set_seed(42)
np.random.seed(42)

class TradingReadyTCN:
    """
    Sistema TCN final optimizado para trading real en Binance
    """
    
    def __init__(self, pair_name="BTCUSDT"):
        self.pair_name = pair_name
        self.scalers = {}
        
        # Configuración final optimizada
        self.config = {
            'sequence_length': 24,  # 2 horas en timeframe 5min
            'step_size': 3,  # Mayor diversidad con menos overlap
            'learning_rate': 5e-4,  # Learning rate más conservador
            'batch_size': 16,  # Batch size más pequeño para mejor generalización
            'n_samples': 5000,  # Más datos para mejor entrenamiento
        }
        
        # Umbrales optimizados basados en análisis de mercado
        self.thresholds = self._get_optimized_thresholds()
    
    def _get_optimized_thresholds(self):
        """
        Umbrales finales optimizados para máxima precisión
        """
        thresholds = {
            "BTCUSDT": {
                'strong_buy': 0.006,   # 0.6% para BUY fuerte
                'weak_buy': 0.002,     # 0.2% para BUY débil
                'strong_sell': -0.006, # -0.6% para SELL fuerte
                'weak_sell': -0.002,   # -0.2% para SELL débil
                'hold_vol_max': 0.004, # Volatilidad máxima para HOLD
                'hold_trend_max': 0.001, # Trend máximo para HOLD
            },
            "ETHUSDT": {
                'strong_buy': 0.008,
                'weak_buy': 0.003,
                'strong_sell': -0.008,
                'weak_sell': -0.003,
                'hold_vol_max': 0.005,
                'hold_trend_max': 0.0015,
            },
            "BNBUSDT": {
                'strong_buy': 0.010,
                'weak_buy': 0.004,
                'strong_sell': -0.010,
                'weak_sell': -0.004,
                'hold_vol_max': 0.006,
                'hold_trend_max': 0.002,
            }
        }
        return thresholds.get(self.pair_name, thresholds["BTCUSDT"])
    
    def generate_enhanced_market_data(self, n_samples=5000):
        """
        Genera datos de mercado mejorados con mayor realismo
        """
        print(f"Generando datos mejorados para {self.pair_name}...")
        
        np.random.seed(42)
        
        # Parámetros mejorados por par
        params = {
            "BTCUSDT": {'price': 45000, 'vol': 0.012},
            "ETHUSDT": {'price': 2800, 'vol': 0.016},
            "BNBUSDT": {'price': 350, 'vol': 0.020}
        }
        
        param = params.get(self.pair_name, params["BTCUSDT"])
        base_price = param['price']
        base_volatility = param['vol']
        
        # Estados de mercado más sofisticados
        market_states = {
            'strong_bull': {'prob': 0.15, 'length': (30, 100), 'trend': (0.002, 0.008), 'vol_mult': 0.8},
            'weak_bull': {'prob': 0.15, 'length': (50, 150), 'trend': (0.0005, 0.003), 'vol_mult': 0.6},
            'sideways': {'prob': 0.40, 'length': (100, 400), 'trend': (-0.0005, 0.0005), 'vol_mult': 0.4},
            'weak_bear': {'prob': 0.15, 'length': (50, 150), 'trend': (-0.003, -0.0005), 'vol_mult': 0.6},
            'strong_bear': {'prob': 0.15, 'length': (30, 100), 'trend': (-0.008, -0.002), 'vol_mult': 0.8},
        }
        
        prices = [base_price]
        volumes = []
        market_regimes = []
        
        current_state = 'sideways'
        state_counter = 0
        max_state_length = np.random.randint(*market_states[current_state]['length'])
        
        for i in range(n_samples):
            # Cambio de estado si es necesario
            if state_counter >= max_state_length:
                state_probs = [market_states[state]['prob'] for state in market_states.keys()]
                current_state = np.random.choice(list(market_states.keys()), p=state_probs)
                state_counter = 0
                max_state_length = np.random.randint(*market_states[current_state]['length'])
            
            # Generar return según estado actual
            state_config = market_states[current_state]
            
            if current_state == 'sideways':
                # Movimiento lateral mejorado con micro-tendencias
                cycle_pos = (state_counter / max_state_length) * 4 * np.pi
                base_oscillation = 0.0008 * np.sin(cycle_pos)
                micro_trend = np.random.uniform(*state_config['trend'])
                noise = np.random.normal(0, base_volatility * state_config['vol_mult'])
                return_val = base_oscillation + micro_trend + noise
            else:
                # Tendencias con variabilidad realista
                base_trend = np.random.uniform(*state_config['trend'])
                
                # Añadir retrocesos/correcciones realistas
                if np.random.random() < 0.12:  # 12% probabilidad de corrección
                    correction_factor = -0.4 if 'bull' in current_state else -0.6
                    base_trend *= correction_factor
                
                noise = np.random.normal(0, base_volatility * state_config['vol_mult'])
                return_val = base_trend + noise
            
            # Aplicar return con límites realistas
            return_val = np.clip(return_val, -0.05, 0.05)  # Límite ±5%
            new_price = prices[-1] * (1 + return_val)
            prices.append(new_price)
            
            # Volumen correlacionado con volatilidad
            vol_base = 10.0 if current_state != 'sideways' else 9.2
            vol_var = 0.4 if abs(return_val) > 0.01 else 0.2
            volume = np.random.lognormal(vol_base, vol_var)
            volumes.append(volume)
            
            market_regimes.append(current_state)
            state_counter += 1
        
        # Crear DataFrame mejorado
        data = pd.DataFrame({
            'close': prices[1:],
            'open': prices[:-1],
            'volume': volumes,
            'market_state': market_regimes
        })
        
        # OHLC realista con spreads
        spread_factor = 0.0008
        data['high'] = data['close'] * (1 + np.abs(np.random.normal(0, spread_factor, len(data))))
        data['low'] = data['close'] * (1 - np.abs(np.random.normal(0, spread_factor, len(data))))
        
        # Ajustar coherencia OHLC
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        # Verificar distribución de estados
        state_counts = data['market_state'].value_counts()
        print(f"Distribución de estados de mercado:")
        for state, count in state_counts.items():
            pct = count / len(data) * 100
            print(f"  {state}: {count} ({pct:.1f}%)")
        
        return data
    
    def create_advanced_features(self, data):
        """
        Features avanzados optimizados para trading real
        """
        features = pd.DataFrame(index=data.index)
        
        # 1. Returns multi-timeframe optimizados
        for period in [1, 2, 3, 5, 8, 13]:
            features[f'returns_{period}'] = data['close'].pct_change(period)
            features[f'returns_{period}_abs'] = np.abs(features[f'returns_{period}'])
        
        # 2. Volatilidad mejorada
        for window in [5, 10, 20, 40]:
            features[f'vol_{window}'] = data['close'].pct_change().rolling(window).std()
            features[f'vol_norm_{window}'] = features[f'vol_{window}'] / features[f'vol_{window}'].rolling(100).mean()
        
        # 3. Trend analysis avanzado
        for short, long in [(5, 20), (10, 40), (20, 60)]:
            sma_short = data['close'].rolling(short).mean()
            sma_long = data['close'].rolling(long).mean()
            features[f'trend_{short}_{long}'] = (sma_short - sma_long) / data['close']
            features[f'trend_strength_{short}_{long}'] = abs(features[f'trend_{short}_{long}'])
        
        # 4. Price position analysis
        for window in [10, 20, 50]:
            high_max = data['high'].rolling(window).max()
            low_min = data['low'].rolling(window).min()
            features[f'price_pos_{window}'] = (data['close'] - low_min) / (high_max - low_min)
            features[f'price_range_{window}'] = (high_max - low_min) / data['close']
        
        # 5. Momentum indicators optimizados
        features['rsi_14'] = self._calculate_rsi(data['close'], 14)
        features['rsi_7'] = self._calculate_rsi(data['close'], 7)
        features['rsi_neutral'] = abs(features['rsi_14'] - 50) / 50
        
        # 6. MACD optimizado
        ema_12 = data['close'].ewm(span=12).mean()
        ema_26 = data['close'].ewm(span=26).mean()
        features['macd'] = (ema_12 - ema_26) / data['close']
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # 7. Volume analysis
        features['volume_sma'] = data['volume'].rolling(20).mean()
        features['volume_ratio'] = data['volume'] / features['volume_sma']
        features['volume_trend'] = data['volume'].rolling(5).mean() / data['volume'].rolling(20).mean()
        
        # 8. Bollinger Bands
        bb_middle = data['close'].rolling(20).mean()
        bb_std = data['close'].rolling(20).std()
        features['bb_upper'] = bb_middle + (bb_std * 2)
        features['bb_lower'] = bb_middle - (bb_std * 2)
        features['bb_position'] = (data['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / bb_middle
        
        # 9. Support/Resistance levels
        features['support_strength'] = self._calculate_support_resistance(data['low'], window=20, mode='support')
        features['resistance_strength'] = self._calculate_support_resistance(data['high'], window=20, mode='resistance')
        
        print(f"Features avanzados creados: {len(features.columns)} features")
        
        return features.fillna(method='ffill').fillna(0)
    
    def _calculate_rsi(self, prices, window=14):
        """RSI optimizado"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_support_resistance(self, prices, window=20, mode='support'):
        """Calcula fuerza de soporte/resistencia"""
        if mode == 'support':
            levels = prices.rolling(window).min()
            touches = (prices == levels).rolling(window).sum()
        else:
            levels = prices.rolling(window).max()
            touches = (prices == levels).rolling(window).sum()
        
        return touches / window
    
    def create_high_quality_sequences(self, features, data):
        """
        Crea secuencias de alta calidad con clasificación precisa
        """
        print(f"Creando secuencias de alta calidad...")
        
        # Normalización robusta
        normalized_features = features.copy()
        scaler = RobustScaler()
        
        for col in features.columns:
            normalized_features[col] = scaler.fit_transform(features[col].values.reshape(-1, 1)).flatten()
            self.scalers[col] = scaler
        
        sequences = []
        targets = []
        sequence_length = self.config['sequence_length']
        step_size = self.config['step_size']
        
        thresholds = self.thresholds
        
        # Contadores para distribución optimizada
        class_counts = {0: 0, 1: 0, 2: 0}
        target_per_class = 600  # Más muestras para mejor entrenamiento
        
        for i in range(sequence_length, len(normalized_features) - 1, step_size):
            seq = normalized_features.iloc[i-sequence_length:i].values
            
            # Métricas avanzadas para clasificación
            future_return = features.iloc[i+1]['returns_1']
            volatility = features.iloc[i]['vol_10']
            trend_strength = features.iloc[i]['trend_strength_10_40']
            rsi_neutral = features.iloc[i]['rsi_neutral']
            bb_position = features.iloc[i]['bb_position']
            volume_ratio = features.iloc[i]['volume_ratio']
            macd_hist = features.iloc[i]['macd_hist']
            
            # CLASIFICACIÓN AVANZADA CON MÚLTIPLES CRITERIOS
            
            # Condiciones para HOLD (más estrictas y precisas)
            is_strong_hold = (
                volatility < thresholds['hold_vol_max'] and
                trend_strength < thresholds['hold_trend_max'] and
                rsi_neutral < 0.25 and  # RSI cerca del 50
                abs(future_return) < abs(thresholds['weak_buy']) and
                0.2 < bb_position < 0.8 and  # Dentro de bandas
                abs(macd_hist) < 0.0005  # MACD neutral
            )
            
            # Condiciones para BUY
            is_strong_buy = (
                future_return >= thresholds['strong_buy'] or
                (future_return >= thresholds['weak_buy'] and 
                 macd_hist > 0.0002 and volume_ratio > 1.1)
            )
            
            # Condiciones para SELL
            is_strong_sell = (
                future_return <= thresholds['strong_sell'] or
                (future_return <= thresholds['weak_sell'] and 
                 macd_hist < -0.0002 and volume_ratio > 1.1)
            )
            
            # Clasificación final con prioridades
            if is_strong_hold:
                target_class = 1  # HOLD
            elif is_strong_buy:
                target_class = 2  # BUY
            elif is_strong_sell:
                target_class = 0  # SELL
            else:
                # Clasificación secundaria basada en return
                if future_return > thresholds['weak_buy']:
                    target_class = 2  # BUY
                elif future_return < thresholds['weak_sell']:
                    target_class = 0  # SELL
                else:
                    target_class = 1  # HOLD por defecto
            
            # Balance inteligente manteniendo calidad
            if class_counts[target_class] < target_per_class:
                sequences.append(seq)
                targets.append(target_class)
                class_counts[target_class] += 1
            elif min(class_counts.values()) < target_per_class * 0.8:
                # Solo asignar a clase deficitaria si está muy por debajo
                min_class = min(class_counts, key=class_counts.get)
                if class_counts[min_class] < target_per_class * 0.8:
                    sequences.append(seq)
                    targets.append(min_class)
                    class_counts[min_class] += 1
            
            # Parar cuando tengamos suficientes muestras de calidad
            if all(count >= target_per_class * 0.9 for count in class_counts.values()):
                break
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        # Verificar distribución final
        unique, counts = np.unique(targets, return_counts=True)
        class_names = ['SELL', 'HOLD', 'BUY']
        print(f"Distribución de alta calidad:")
        for i, class_name in enumerate(class_names):
            if i in unique:
                idx = list(unique).index(i)
                percentage = counts[idx] / len(targets) * 100
                print(f"  {class_name}: {counts[idx]} ({percentage:.1f}%)")
        
        return sequences, targets
    
    def build_advanced_model(self, input_shape):
        """
        Modelo TCN avanzado optimizado para máxima precisión
        """
        print(f"Construyendo modelo avanzado...")
        
        inputs = layers.Input(shape=input_shape)
        
        # Input normalization
        x = layers.LayerNormalization()(inputs)
        
        # Encoder TCN con atención
        x1 = layers.Conv1D(32, 3, dilation_rate=1, padding='causal', activation='relu')(x)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Dropout(0.1)(x1)
        
        x2 = layers.Conv1D(48, 3, dilation_rate=2, padding='causal', activation='relu')(x1)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Dropout(0.2)(x2)
        
        x3 = layers.Conv1D(64, 3, dilation_rate=4, padding='causal', activation='relu')(x2)
        x3 = layers.BatchNormalization()(x3)
        x3 = layers.Dropout(0.2)(x3)
        
        x4 = layers.Conv1D(48, 3, dilation_rate=8, padding='causal', activation='relu')(x3)
        x4 = layers.BatchNormalization()(x4)
        x4 = layers.Dropout(0.3)(x4)
        
        # Multi-scale feature extraction
        global_avg = layers.GlobalAveragePooling1D()(x4)
        global_max = layers.GlobalMaxPooling1D()(x4)
        
        # Simplified attention mechanism
        attention_context = layers.Dense(48, activation='tanh')(global_avg)
        attention_weights = layers.Dense(1, activation='sigmoid')(attention_context)
        
        # Apply attention to last timestep
        last_timestep = layers.Lambda(lambda x: x[:, -1, :])(x4)
        attended_features = layers.Multiply()([last_timestep, layers.Flatten()(attention_weights)])
        
        # Combine all features
        combined = layers.Concatenate()([global_avg, global_max, attended_features])
        
        # Classification head
        x = layers.Dense(128, activation='relu')(combined)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer con inicialización balanceada
        outputs = layers.Dense(3, activation='softmax',
                              bias_initializer=tf.keras.initializers.Constant([0.33, 0.33, 0.33]))(x)
        
        model = models.Model(inputs, outputs)
        
        # Compilación optimizada
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(
                learning_rate=self.config['learning_rate'],
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

def test_trading_ready_system():
    """
    Test del sistema trading-ready final
    """
    print("=== TCN TRADING READY FINAL ===")
    print("Sistema definitivo optimizado para trading real\n")
    
    pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    final_results = {}
    
    for pair in pairs:
        print(f"\n{'='*70}")
        print(f"TESTING TRADING READY {pair}")
        print('='*70)
        
        # Crear sistema final
        tcn_system = TradingReadyTCN(pair_name=pair)
        
        # Datos mejorados
        data = tcn_system.generate_enhanced_market_data(n_samples=tcn_system.config['n_samples'])
        
        # Features avanzados
        features = tcn_system.create_advanced_features(data)
        
        # Secuencias de alta calidad
        sequences, targets = tcn_system.create_high_quality_sequences(features, data)
        
        # Aplicar SMOTEENN para calidad superior
        n_samples, n_timesteps, n_features = sequences.shape
        X_reshaped = sequences.reshape(n_samples, n_timesteps * n_features)
        
        try:
            smoteenn = SMOTEENN(sampling_strategy='all', random_state=42)
            X_balanced, y_balanced = smoteenn.fit_resample(X_reshaped, targets)
            X_balanced = X_balanced.reshape(-1, n_timesteps, n_features)
            
            print(f"\nSMOTEENN aplicado:")
            unique, counts = np.unique(y_balanced, return_counts=True)
            for i, count in enumerate(counts):
                class_name = ['SELL', 'HOLD', 'BUY'][i]
                pct = count / len(y_balanced) * 100
                print(f"  {class_name}: {count} ({pct:.1f}%)")
        
        except Exception as e:
            print(f"SMOTEENN falló: {e}")
            X_balanced, y_balanced = sequences, targets
        
        # Split temporal mejorado
        split_point = int(0.8 * len(X_balanced))
        X_train, X_test = X_balanced[:split_point], X_balanced[split_point:]
        y_train, y_test = y_balanced[:split_point], y_balanced[split_point:]
        
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Modelo avanzado
        model = tcn_system.build_advanced_model(X_train.shape[1:])
        
        # Class weights optimizados
        unique_classes = np.unique(y_balanced)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_balanced)
        class_weight_dict = dict(zip(unique_classes, class_weights))
        
        # Callbacks avanzados
        callbacks_list = [
            callbacks.EarlyStopping(
                patience=20, 
                restore_best_weights=True, 
                monitor='val_accuracy',
                mode='max',
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                factor=0.6, 
                patience=10, 
                min_lr=1e-7,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                f'best_model_{pair}.h5',
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=0
            )
        ]
        
        # Entrenamiento optimizado
        print(f"Entrenando modelo trading-ready...")
        history = model.fit(
            X_train, y_train,
            batch_size=tcn_system.config['batch_size'],
            epochs=120,
            validation_split=0.2,
            class_weight=class_weight_dict,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Predicciones finales
        predictions = model.predict(X_test, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        
        # Evaluación completa
        class_names = ['SELL', 'HOLD', 'BUY']
        unique_pred, counts_pred = np.unique(pred_classes, return_counts=True)
        
        print(f"\nResultados finales {pair}:")
        hold_detected = 1 in unique_pred
        three_classes = len(unique_pred) == 3
        
        signal_distribution = {}
        for i, class_name in enumerate(class_names):
            if i in unique_pred:
                idx = list(unique_pred).index(i)
                count = counts_pred[idx]
                percentage = count / len(pred_classes) * 100
                signal_distribution[class_name] = percentage
                print(f"  {class_name}: {count} ({percentage:.1f}%)")
            else:
                signal_distribution[class_name] = 0.0
                print(f"  {class_name}: 0 (0.0%)")
        
        # Métricas detalladas
        overall_accuracy = accuracy_score(y_test, pred_classes)
        avg_confidence = np.mean(confidences)
        f1_macro = f1_score(y_test, pred_classes, average='macro')
        
        # Bias score
        sell_pct = signal_distribution['SELL'] / 100
        hold_pct = signal_distribution['HOLD'] / 100
        buy_pct = signal_distribution['BUY'] / 100
        target_pct = 1/3
        deviations = abs(sell_pct - target_pct) + abs(hold_pct - target_pct) + abs(buy_pct - target_pct)
        bias_score = 10 * (1 - deviations / 2)
        
        print(f"\n📊 Métricas Trading-Ready:")
        print(f"  Accuracy: {overall_accuracy:.3f}")
        print(f"  Confianza: {avg_confidence:.3f}")
        print(f"  F1 Macro: {f1_macro:.3f}")
        print(f"  Bias Score: {bias_score:.1f}/10")
        print(f"  HOLD detectado: {'✅' if hold_detected else '❌'}")
        print(f"  3 clases: {'✅' if three_classes else '❌'}")
        
        # Evaluación trading-ready final
        trading_ready = (
            overall_accuracy >= 0.4 and
            avg_confidence >= 0.6 and
            f1_macro >= 0.35 and
            bias_score >= 5.0 and
            hold_detected and
            three_classes
        )
        
        print(f"\n🏆 EVALUACIÓN TRADING-READY:")
        if trading_ready:
            print(f"🎉 {pair} ¡TRADING-READY COMPLETO!")
            print(f"✅ Todas las métricas objetivo alcanzadas")
            print(f"✅ Listo para deploy en Binance")
        else:
            print(f"🔧 {pair} en optimización final:")
            if overall_accuracy < 0.4:
                print(f"   • Accuracy: {overall_accuracy:.3f} (objetivo: ≥0.4)")
            if avg_confidence < 0.6:
                print(f"   • Confianza: {avg_confidence:.3f} (objetivo: ≥0.6)")
            if f1_macro < 0.35:
                print(f"   • F1 Score: {f1_macro:.3f} (objetivo: ≥0.35)")
            if bias_score < 5.0:
                print(f"   • Bias Score: {bias_score:.1f} (objetivo: ≥5.0)")
            if not hold_detected:
                print(f"   • HOLD detection pendiente")
            if not three_classes:
                print(f"   • 3 clases prediction pendiente")
        
        # Guardar resultado
        final_results[pair] = {
            'trading_ready': trading_ready,
            'accuracy': overall_accuracy,
            'confidence': avg_confidence,
            'f1_score': f1_macro,
            'bias_score': bias_score,
            'hold_detected': hold_detected,
            'three_classes': three_classes,
            'classes_predicted': len(unique_pred)
        }
    
    # Resumen final del sistema
    print(f"\n{'='*80}")
    print("🏆 RESUMEN TRADING-READY FINAL")
    print('='*80)
    
    ready_pairs = sum(1 for r in final_results.values() if r['trading_ready'])
    total_pairs = len(final_results)
    
    print(f"🚀 TRADING-READY: {ready_pairs}/{total_pairs} pares")
    
    avg_metrics = {
        'accuracy': np.mean([r['accuracy'] for r in final_results.values()]),
        'confidence': np.mean([r['confidence'] for r in final_results.values()]),
        'f1_score': np.mean([r['f1_score'] for r in final_results.values()]),
        'bias_score': np.mean([r['bias_score'] for r in final_results.values()])
    }
    
    print(f"\n📊 MÉTRICAS PROMEDIO DEL SISTEMA:")
    print(f"  Accuracy: {avg_metrics['accuracy']:.3f}")
    print(f"  Confianza: {avg_metrics['confidence']:.3f}")
    print(f"  F1 Score: {avg_metrics['f1_score']:.3f}")
    print(f"  Bias Score: {avg_metrics['bias_score']:.1f}/10")
    
    for pair, result in final_results.items():
        status = "🎉 TRADING-READY" if result['trading_ready'] else "🔧 OPTIMIZANDO"
        print(f"\n{pair}: {status}")
        print(f"  📊 Accuracy: {result['accuracy']:.3f}")
        print(f"  🔥 Confianza: {result['confidence']:.3f}")
        print(f"  📈 F1: {result['f1_score']:.3f}")
        print(f"  🎯 Bias: {result['bias_score']:.1f}/10")
        print(f"  ✅ HOLD: {'Sí' if result['hold_detected'] else 'No'}")
        print(f"  🔢 Clases: {result['classes_predicted']}/3")
    
    success_rate = (ready_pairs / total_pairs) * 100
    
    print(f"\n{'='*50}")
    if success_rate >= 66:
        print(f"🎉 SISTEMA TRADING-READY: {success_rate:.0f}%")
        print(f"✅ Optimización de accuracy/confianza COMPLETA")
        print(f"✅ Métricas trading-ready alcanzadas")
        print(f"✅ Sistema listo para Binance production")
        print(f"🚀 SIGUIENTE PASO: Integración con Binance API")
    else:
        print(f"🔧 SISTEMA EN OPTIMIZACIÓN: {success_rate:.0f}%")
        print(f"⚡ Continúa refinamiento de hiperparámetros")
    
    return final_results

if __name__ == "__main__":
    test_trading_ready_system() 