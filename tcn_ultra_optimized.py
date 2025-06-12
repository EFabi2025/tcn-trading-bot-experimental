#!/usr/bin/env python3
"""
TCN ULTRA OPTIMIZADO - Sistema final para alcanzar métricas trading-ready
Implementa optimizaciones avanzadas para superar thresholds de accuracy y confianza
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import ADASYN
import warnings
warnings.filterwarnings('ignore')

tf.random.set_seed(42)
np.random.seed(42)

class UltraOptimizedTCN:
    """Sistema TCN ultra-optimizado para trading-ready"""

    def __init__(self, pair_name="BTCUSDT"):
        self.pair_name = pair_name
        self.scalers = {}

        # Configuración ultra-optimizada
        self.config = {
            'sequence_length': 32,  # Secuencias más largas para mejor contexto
            'step_size': 2,        # Menor step para más diversidad
            'learning_rate': 2e-4,  # LR más bajo para convergencia estable
            'batch_size': 8,       # Batch size pequeño para mejor generalización
            'n_samples': 8000,     # Más datos para entrenamiento robusto
            'epochs': 150,         # Más epochs con early stopping agresivo
        }

        self.thresholds = self._get_ultra_thresholds()

    def _get_ultra_thresholds(self):
        """Umbrales ultra-optimizados para máxima precisión"""
        thresholds = {
            "BTCUSDT": {
                'strong_buy': 0.008, 'weak_buy': 0.003,
                'strong_sell': -0.008, 'weak_sell': -0.003,
                'hold_vol_max': 0.003, 'hold_trend_max': 0.0008
            },
            "ETHUSDT": {
                'strong_buy': 0.010, 'weak_buy': 0.004,
                'strong_sell': -0.010, 'weak_sell': -0.004,
                'hold_vol_max': 0.004, 'hold_trend_max': 0.001
            },
            "BNBUSDT": {
                'strong_buy': 0.012, 'weak_buy': 0.005,
                'strong_sell': -0.012, 'weak_sell': -0.005,
                'hold_vol_max': 0.005, 'hold_trend_max': 0.0015
            }
        }
        return thresholds.get(self.pair_name, thresholds["BTCUSDT"])

    def generate_ultra_realistic_data(self, n_samples=8000):
        """Genera datos ultra-realistas optimizados"""
        print(f"Generando datos ultra-realistas para {self.pair_name}...")

        np.random.seed(42)

        # Parámetros optimizados por par
        params = {
            "BTCUSDT": {'price': 50000, 'vol': 0.015},
            "ETHUSDT": {'price': 3000, 'vol': 0.020},
            "BNBUSDT": {'price': 400, 'vol': 0.025}
        }

        param = params.get(self.pair_name, params["BTCUSDT"])
        base_price = param['price']
        base_volatility = param['vol']

        # Estados de mercado con probabilidades realistas
        market_states = {
            'consolidation': {'prob': 0.45, 'length': (80, 300), 'trend': (-0.0003, 0.0003), 'vol_mult': 0.3},
            'uptrend': {'prob': 0.20, 'length': (40, 120), 'trend': (0.001, 0.006), 'vol_mult': 0.7},
            'downtrend': {'prob': 0.20, 'length': (40, 120), 'trend': (-0.006, -0.001), 'vol_mult': 0.7},
            'pump': {'prob': 0.08, 'length': (15, 50), 'trend': (0.004, 0.012), 'vol_mult': 1.2},
            'dump': {'prob': 0.07, 'length': (15, 50), 'trend': (-0.012, -0.004), 'vol_mult': 1.2},
        }

        prices = [base_price]
        volumes = []
        market_regimes = []

        current_state = 'consolidation'
        state_counter = 0
        max_state_length = np.random.randint(*market_states[current_state]['length'])

        for i in range(n_samples):
            if state_counter >= max_state_length:
                state_probs = [market_states[state]['prob'] for state in market_states.keys()]
                current_state = np.random.choice(list(market_states.keys()), p=state_probs)
                state_counter = 0
                max_state_length = np.random.randint(*market_states[current_state]['length'])

            state_config = market_states[current_state]

            if current_state == 'consolidation':
                # Consolidación mejorada con micro-ciclos
                cycle_factor = np.sin((state_counter / max_state_length) * 6 * np.pi)
                base_oscillation = 0.0005 * cycle_factor
                micro_trend = np.random.uniform(*state_config['trend'])
                noise = np.random.normal(0, base_volatility * state_config['vol_mult'])
                return_val = base_oscillation + micro_trend + noise
            else:
                # Tendencias con momentum realista
                base_trend = np.random.uniform(*state_config['trend'])

                # Momentum factor
                progress = state_counter / max_state_length
                if current_state in ['pump', 'dump']:
                    momentum = 1.2 if progress < 0.7 else 0.6  # Aceleración inicial, desaceleración final
                else:
                    momentum = 0.8 + 0.4 * progress  # Aceleración gradual

                # Correcciones realistas
                if np.random.random() < 0.08:
                    correction_factor = -0.3 if 'up' in current_state or current_state == 'pump' else -0.4
                    base_trend *= correction_factor

                noise = np.random.normal(0, base_volatility * state_config['vol_mult'])
                return_val = base_trend * momentum + noise

            return_val = np.clip(return_val, -0.08, 0.08)
            new_price = prices[-1] * (1 + return_val)
            prices.append(new_price)

            # Volumen correlacionado mejorado
            vol_base = 10.5 if current_state in ['pump', 'dump'] else 9.8
            vol_variance = 0.6 if abs(return_val) > 0.015 else 0.3
            volume = np.random.lognormal(vol_base, vol_variance)
            volumes.append(volume)

            market_regimes.append(current_state)
            state_counter += 1

        data = pd.DataFrame({
            'close': prices[1:], 'open': prices[:-1],
            'volume': volumes, 'market_state': market_regimes
        })

        # OHLC mejorado
        spread_factor = 0.0006
        data['high'] = data['close'] * (1 + np.abs(np.random.normal(0, spread_factor, len(data))))
        data['low'] = data['close'] * (1 - np.abs(np.random.normal(0, spread_factor, len(data))))
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))

        state_counts = data['market_state'].value_counts()
        print("Estados de mercado:")
        for state, count in state_counts.items():
            pct = count / len(data) * 100
            print(f"  {state}: {count} ({pct:.1f}%)")

        return data

    def create_ultra_features(self, data):
        """Features ultra-optimizados para trading preciso"""
        features = pd.DataFrame(index=data.index)

        # 1. Returns multi-timeframe con suavizado
        for period in [1, 2, 3, 5, 8, 13, 21]:
            raw_returns = data['close'].pct_change(period)
            features[f'returns_{period}'] = raw_returns
            features[f'returns_{period}_ema'] = raw_returns.ewm(span=3).mean()
            features[f'returns_{period}_abs'] = np.abs(raw_returns)

        # 2. Volatilidad adaptativa
        for window in [5, 10, 20, 40, 60]:
            vol = data['close'].pct_change().rolling(window).std()
            features[f'vol_{window}'] = vol
            features[f'vol_norm_{window}'] = vol / vol.rolling(200).mean()
            features[f'vol_rank_{window}'] = vol.rolling(100).rank(pct=True)

        # 3. Trend strength mejorado
        for short, long in [(5, 20), (10, 40), (20, 60), (40, 120)]:
            sma_s, sma_l = data['close'].rolling(short).mean(), data['close'].rolling(long).mean()
            trend = (sma_s - sma_l) / data['close']
            features[f'trend_{short}_{long}'] = trend
            features[f'trend_strength_{short}_{long}'] = abs(trend)
            features[f'trend_accel_{short}_{long}'] = trend.diff()

        # 4. Momentum suite completo
        features['rsi_14'] = self._rsi(data['close'], 14)
        features['rsi_7'] = self._rsi(data['close'], 7)
        features['rsi_21'] = self._rsi(data['close'], 21)
        features['rsi_neutral'] = abs(features['rsi_14'] - 50) / 50
        features['rsi_divergence'] = features['rsi_14'] - features['rsi_21']

        # 5. MACD avanzado
        ema12, ema26 = data['close'].ewm(span=12).mean(), data['close'].ewm(span=26).mean()
        macd = (ema12 - ema26) / data['close']
        signal = macd.ewm(span=9).mean()
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = macd - signal
        features['macd_momentum'] = features['macd_hist'].diff()

        # 6. Bollinger Bands optimizado
        bb_mid = data['close'].rolling(20).mean()
        bb_std = data['close'].rolling(20).std()
        features['bb_position'] = (data['close'] - bb_mid) / (2 * bb_std)
        features['bb_width'] = (4 * bb_std) / bb_mid
        features['bb_squeeze'] = features['bb_width'] < features['bb_width'].rolling(20).quantile(0.2)

        # 7. Price action features
        for window in [10, 20, 50, 100]:
            high_max, low_min = data['high'].rolling(window).max(), data['low'].rolling(window).min()
            features[f'price_rank_{window}'] = (data['close'] - low_min) / (high_max - low_min)
            features[f'range_norm_{window}'] = (high_max - low_min) / data['close']

        # 8. Volume analysis mejorado
        vol_sma = data['volume'].rolling(20).mean()
        features['volume_ratio'] = data['volume'] / vol_sma
        features['volume_trend'] = data['volume'].rolling(5).mean() / data['volume'].rolling(20).mean()
        features['volume_surge'] = features['volume_ratio'] > 2.0

        # 9. Support/resistance strength
        features['support_test'] = self._support_resistance_test(data['low'])
        features['resistance_test'] = self._support_resistance_test(data['high'], mode='resistance')

        print(f"Ultra-features creados: {len(features.columns)} features")
        return features.fillna(method='ffill').fillna(0)

    def _rsi(self, prices, window=14):
        """RSI mejorado"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _support_resistance_test(self, prices, window=20, mode='support'):
        """Test de soporte/resistencia"""
        if mode == 'support':
            levels = prices.rolling(window).min()
            return (prices <= levels * 1.002).rolling(window).sum() / window
        else:
            levels = prices.rolling(window).max()
            return (prices >= levels * 0.998).rolling(window).sum() / window

    def create_precision_sequences(self, features, data):
        """Crea secuencias de máxima precisión"""
        print("Creando secuencias de precisión ultra-alta...")

        # Normalización ultra-robusta
        normalized_features = features.copy()

        for col in features.columns:
            scaler = RobustScaler()
            normalized_features[col] = scaler.fit_transform(features[col].values.reshape(-1, 1)).flatten()
            self.scalers[col] = scaler

        sequences, targets = [], []
        sequence_length = self.config['sequence_length']
        step_size = self.config['step_size']
        thresholds = self.thresholds

        class_counts = {0: 0, 1: 0, 2: 0}
        target_per_class = 800  # Más muestras de alta calidad

        for i in range(sequence_length, len(normalized_features) - 3, step_size):
            seq = normalized_features.iloc[i-sequence_length:i].values

            # Métricas ultra-precisas para clasificación
            future_1 = features.iloc[i+1]['returns_1']
            future_2 = features.iloc[i+2]['returns_1']
            future_3 = features.iloc[i+3]['returns_1']
            future_avg = (future_1 + future_2 + future_3) / 3

            volatility = features.iloc[i]['vol_10']
            trend_strength = features.iloc[i]['trend_strength_10_40']
            rsi_neutral = features.iloc[i]['rsi_neutral']
            bb_position = features.iloc[i]['bb_position']
            volume_ratio = features.iloc[i]['volume_ratio']
            macd_hist = features.iloc[i]['macd_hist']
            macd_momentum = features.iloc[i]['macd_momentum']

            # CLASIFICACIÓN ULTRA-PRECISA

            # HOLD ultra-estricto
            is_perfect_hold = (
                volatility < thresholds['hold_vol_max'] and
                trend_strength < thresholds['hold_trend_max'] and
                rsi_neutral < 0.2 and  # RSI muy neutral
                abs(bb_position) < 0.5 and  # Dentro de bandas medias
                abs(macd_hist) < 0.0003 and  # MACD muy neutral
                abs(future_avg) < abs(thresholds['weak_buy']) * 0.8 and  # Movimiento futuro mínimo
                volume_ratio < 1.5  # Sin volumen anómalo
            )

            # BUY con múltiples confirmaciones
            is_strong_buy = (
                (future_avg >= thresholds['strong_buy']) or
                (future_avg >= thresholds['weak_buy'] and
                 macd_hist > 0.0002 and macd_momentum > 0 and
                 volume_ratio > 1.2 and rsi_neutral > 0.1)
            )

            # SELL con múltiples confirmaciones
            is_strong_sell = (
                (future_avg <= thresholds['strong_sell']) or
                (future_avg <= thresholds['weak_sell'] and
                 macd_hist < -0.0002 and macd_momentum < 0 and
                 volume_ratio > 1.2 and rsi_neutral > 0.1)
            )

            # Clasificación con prioridad HOLD
            if is_perfect_hold:
                target_class = 1
            elif is_strong_buy and not is_strong_sell:
                target_class = 2
            elif is_strong_sell and not is_strong_buy:
                target_class = 0
            else:
                # Clasificación secundaria más conservadora
                if future_avg > thresholds['weak_buy'] * 1.5:
                    target_class = 2
                elif future_avg < thresholds['weak_sell'] * 1.5:
                    target_class = 0
                else:
                    target_class = 1  # HOLD por defecto

            # Balance inteligente ultra-conservador
            if class_counts[target_class] < target_per_class:
                sequences.append(seq)
                targets.append(target_class)
                class_counts[target_class] += 1

            if all(count >= target_per_class * 0.95 for count in class_counts.values()):
                break

        sequences, targets = np.array(sequences), np.array(targets)

        unique, counts = np.unique(targets, return_counts=True)
        class_names = ['SELL', 'HOLD', 'BUY']
        print("Distribución ultra-precisa:")
        for i, class_name in enumerate(class_names):
            if i in unique:
                idx = list(unique).index(i)
                percentage = counts[idx] / len(targets) * 100
                print(f"  {class_name}: {counts[idx]} ({percentage:.1f}%)")

        return sequences, targets

    def build_ultra_model(self, input_shape):
        """Modelo ultra-optimizado para máxima precisión"""
        print("Construyendo modelo ultra-optimizado...")

        inputs = layers.Input(shape=input_shape)

        # Normalization layer
        x = layers.LayerNormalization()(inputs)

        # Multi-scale TCN con residual connections
        # Scale 1: Short-term patterns
        x1 = layers.Conv1D(24, 3, dilation_rate=1, padding='causal', activation='swish')(x)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Dropout(0.05)(x1)

        # Scale 2: Medium-term patterns
        x2 = layers.Conv1D(32, 3, dilation_rate=2, padding='causal', activation='swish')(x1)
        x2 = layers.BatchNormalization()(x2)
        x2 = layers.Dropout(0.1)(x2)

        # Scale 3: Long-term patterns
        x3 = layers.Conv1D(40, 3, dilation_rate=4, padding='causal', activation='swish')(x2)
        x3 = layers.BatchNormalization()(x3)
        x3 = layers.Dropout(0.15)(x3)

        # Scale 4: Very long-term patterns
        x4 = layers.Conv1D(32, 3, dilation_rate=8, padding='causal', activation='swish')(x3)
        x4 = layers.BatchNormalization()(x4)
        x4 = layers.Dropout(0.2)(x4)

        # Multi-pooling strategy
        global_avg = layers.GlobalAveragePooling1D()(x4)
        global_max = layers.GlobalMaxPooling1D()(x4)

        # Last timestep (most recent information)
        last_timestep = layers.Lambda(lambda x: x[:, -1, :])(x4)

        # Combine multi-scale features
        combined = layers.Concatenate()([global_avg, global_max, last_timestep])

        # Dense classification head con regularización
        x = layers.Dense(96, activation='swish')(combined)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Dense(48, activation='swish')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Dense(24, activation='swish')(x)
        x = layers.Dropout(0.2)(x)

        # Output layer con bias balanceado
        outputs = layers.Dense(3, activation='softmax',
                              bias_initializer=tf.constant_initializer([0.33, 0.33, 0.33]))(x)

        model = models.Model(inputs, outputs)

        # Compilación ultra-optimizada
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(
                learning_rate=self.config['learning_rate'],
                beta_1=0.9, beta_2=0.999, epsilon=1e-8
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

def test_ultra_optimized_system():
    """Test del sistema ultra-optimizado final"""
    print("=== TCN ULTRA OPTIMIZADO FINAL ===")
    print("Sistema definitivo para alcanzar métricas trading-ready\n")

    pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    ultra_results = {}

    for pair in pairs:
        print(f"\n{'='*60}")
        print(f"ULTRA OPTIMIZATION {pair}")
        print('='*60)

        tcn = UltraOptimizedTCN(pair_name=pair)

        # Datos ultra-realistas
        data = tcn.generate_ultra_realistic_data(n_samples=tcn.config['n_samples'])

        # Features ultra-optimizados
        features = tcn.create_ultra_features(data)

        # Secuencias de precisión
        sequences, targets = tcn.create_precision_sequences(features, data)

        # ADASYN para balanceo superior
        n_samples, n_timesteps, n_features = sequences.shape
        X_reshaped = sequences.reshape(n_samples, n_timesteps * n_features)

        try:
            adasyn = ADASYN(sampling_strategy='all', random_state=42, n_neighbors=3)
            X_balanced, y_balanced = adasyn.fit_resample(X_reshaped, targets)
            X_balanced = X_balanced.reshape(-1, n_timesteps, n_features)

            print(f"\nADASYN aplicado:")
            unique, counts = np.unique(y_balanced, return_counts=True)
            for i, count in enumerate(counts):
                class_name = ['SELL', 'HOLD', 'BUY'][i]
                pct = count / len(y_balanced) * 100
                print(f"  {class_name}: {count} ({pct:.1f}%)")
        except Exception as e:
            print(f"ADASYN falló: {e}")
            X_balanced, y_balanced = sequences, targets

        # Split optimizado
        split_point = int(0.8 * len(X_balanced))
        X_train, X_test = X_balanced[:split_point], X_balanced[split_point:]
        y_train, y_test = y_balanced[:split_point], y_balanced[split_point:]

        print(f"Train: {len(X_train)}, Test: {len(X_test)}")

        # Modelo ultra-optimizado
        model = tcn.build_ultra_model(X_train.shape[1:])

        # Class weights ultra-balanceados
        unique_classes = np.unique(y_balanced)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_balanced)
        class_weight_dict = dict(zip(unique_classes, class_weights))

        # Callbacks ultra-agresivos
        callbacks_list = [
            callbacks.EarlyStopping(
                patience=25, restore_best_weights=True,
                monitor='val_accuracy', mode='max', verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                factor=0.5, patience=12, min_lr=1e-8,
                monitor='val_accuracy', mode='max', verbose=1
            ),
            callbacks.ModelCheckpoint(
                f'ultra_model_{pair}.h5', save_best_only=True,
                monitor='val_accuracy', mode='max', verbose=0
            )
        ]

        # Entrenamiento ultra-optimizado
        print("Entrenamiento ultra-optimizado...")
        history = model.fit(
            X_train, y_train,
            batch_size=tcn.config['batch_size'],
            epochs=tcn.config['epochs'],
            validation_split=0.25,  # Más validación
            class_weight=class_weight_dict,
            callbacks=callbacks_list,
            verbose=1
        )

        # Evaluación final
        predictions = model.predict(X_test, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)

        class_names = ['SELL', 'HOLD', 'BUY']
        unique_pred, counts_pred = np.unique(pred_classes, return_counts=True)

        print(f"\nResultados ultra-optimizados {pair}:")
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

        # Métricas ultra-precisas
        overall_accuracy = accuracy_score(y_test, pred_classes)
        avg_confidence = np.mean(confidences)
        f1_macro = f1_score(y_test, pred_classes, average='macro')

        # Bias score ultra-preciso
        sell_pct = signal_distribution.get('SELL', 0) / 100
        hold_pct = signal_distribution.get('HOLD', 0) / 100
        buy_pct = signal_distribution.get('BUY', 0) / 100
        target_pct = 1/3
        deviations = abs(sell_pct - target_pct) + abs(hold_pct - target_pct) + abs(buy_pct - target_pct)
        bias_score = 10 * (1 - deviations / 2)

        print(f"\n🏆 MÉTRICAS ULTRA-OPTIMIZADAS:")
        print(f"  Accuracy: {overall_accuracy:.3f}")
        print(f"  Confianza: {avg_confidence:.3f}")
        print(f"  F1 Macro: {f1_macro:.3f}")
        print(f"  Bias Score: {bias_score:.1f}/10")
        print(f"  HOLD detectado: {'✅' if hold_detected else '❌'}")
        print(f"  3 clases: {'✅' if three_classes else '❌'}")

        # Evaluación trading-ready ultra-estricta
        trading_ready = (
            overall_accuracy >= 0.45 and  # Threshold más alto
            avg_confidence >= 0.65 and   # Threshold más alto
            f1_macro >= 0.40 and         # Threshold más alto
            bias_score >= 6.0 and
            hold_detected and three_classes
        )

        print(f"\n🎯 EVALUACIÓN ULTRA TRADING-READY:")
        if trading_ready:
            print(f"🎉 {pair} ¡ULTRA TRADING-READY ALCANZADO!")
            print(f"✅ Todas las métricas ultra-objetivo conseguidas")
            print(f"🚀 Listo para production en Binance")
        else:
            print(f"🔧 {pair} en ultra-optimización:")
            metrics_needed = []
            if overall_accuracy < 0.45: metrics_needed.append(f"Accuracy: {overall_accuracy:.3f} (≥0.45)")
            if avg_confidence < 0.65: metrics_needed.append(f"Confianza: {avg_confidence:.3f} (≥0.65)")
            if f1_macro < 0.40: metrics_needed.append(f"F1: {f1_macro:.3f} (≥0.40)")
            if bias_score < 6.0: metrics_needed.append(f"Bias: {bias_score:.1f} (≥6.0)")
            if not hold_detected: metrics_needed.append("HOLD detection")
            if not three_classes: metrics_needed.append("3 clases prediction")

            for metric in metrics_needed:
                print(f"   • {metric}")

        ultra_results[pair] = {
            'ultra_ready': trading_ready,
            'accuracy': overall_accuracy,
            'confidence': avg_confidence,
            'f1_score': f1_macro,
            'bias_score': bias_score,
            'hold_detected': hold_detected,
            'three_classes': three_classes
        }

    # Resumen ultra-final
    print(f"\n{'='*70}")
    print("🚀 RESUMEN ULTRA-OPTIMIZADO FINAL")
    print('='*70)

    ultra_ready_count = sum(1 for r in ultra_results.values() if r['ultra_ready'])
    success_rate = (ultra_ready_count / len(ultra_results)) * 100

    avg_metrics = {
        'accuracy': np.mean([r['accuracy'] for r in ultra_results.values()]),
        'confidence': np.mean([r['confidence'] for r in ultra_results.values()]),
        'f1_score': np.mean([r['f1_score'] for r in ultra_results.values()]),
        'bias_score': np.mean([r['bias_score'] for r in ultra_results.values()])
    }

    print(f"🎯 ULTRA-READY: {ultra_ready_count}/{len(ultra_results)} pares")
    print(f"📊 MÉTRICAS PROMEDIO ULTRA:")
    print(f"  Accuracy: {avg_metrics['accuracy']:.3f}")
    print(f"  Confianza: {avg_metrics['confidence']:.3f}")
    print(f"  F1 Score: {avg_metrics['f1_score']:.3f}")
    print(f"  Bias Score: {avg_metrics['bias_score']:.1f}/10")

    for pair, result in ultra_results.items():
        status = "🎉 ULTRA-READY" if result['ultra_ready'] else "🔧 OPTIMIZANDO"
        print(f"\n{pair}: {status}")
        print(f"  📊 Acc: {result['accuracy']:.3f} | 🔥 Conf: {result['confidence']:.3f}")
        print(f"  📈 F1: {result['f1_score']:.3f} | 🎯 Bias: {result['bias_score']:.1f}")
        print(f"  ✅ HOLD: {'Sí' if result['hold_detected'] else 'No'} | 🔢 Clases: {'3/3' if result['three_classes'] else 'X/3'}")

    print(f"\n{'='*40}")
    if success_rate >= 67:
        print(f"🎉 SISTEMA ULTRA TRADING-READY: {success_rate:.0f}%")
        print(f"✅ Ultra-optimización COMPLETA")
        print(f"🚀 SIGUIENTE: Integración Binance API")
    else:
        print(f"🔧 ULTRA-OPTIMIZACIÓN: {success_rate:.0f}%")
        print(f"⚡ Refinamiento final en progreso")

    return ultra_results

if __name__ == "__main__":
    test_ultra_optimized_system()
