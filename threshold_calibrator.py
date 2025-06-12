#!/usr/bin/env python3
"""
THRESHOLD CALIBRATOR - Sistema Automático de Calibración de Umbrales
Calibra umbrales específicos por par para optimizar métricas de trading
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

class ThresholdCalibrator:
    """
    Calibrador automático de umbrales para optimización por par
    """
    
    def __init__(self, pair_name="BTCUSDT"):
        self.pair_name = pair_name
        self.best_params = None
        self.calibration_history = []
        
    def generate_market_data(self, n_samples=3000):
        """
        Genera datos de mercado específicos para el par
        """
        np.random.seed(42)
        
        # Parámetros de mercado específicos por par
        market_params = {
            "BTCUSDT": {
                'base_price': 45000,
                'daily_volatility': 0.025,
                'trend_strength': 0.0001,
                'regime_probabilities': [0.25, 0.5, 0.25]  # bear, sideways, bull
            },
            "ETHUSDT": {
                'base_price': 2500,
                'daily_volatility': 0.035,
                'trend_strength': 0.0002,
                'regime_probabilities': [0.3, 0.4, 0.3]
            },
            "BNBUSDT": {
                'base_price': 300,
                'daily_volatility': 0.045,
                'trend_strength': 0.0003,
                'regime_probabilities': [0.35, 0.3, 0.35]
            }
        }
        
        params = market_params.get(self.pair_name, market_params["BTCUSDT"])
        
        # Generar regímenes de mercado
        regimes = np.random.choice([0, 1, 2], n_samples, p=params['regime_probabilities'])
        
        # Generar returns según régimen
        base_returns = np.random.normal(0, params['daily_volatility'], n_samples)
        regime_adjustments = np.where(regimes == 0, -params['trend_strength'],
                            np.where(regimes == 1, 0, params['trend_strength']))
        
        returns = base_returns + regime_adjustments
        price_path = np.cumsum(returns)
        prices = params['base_price'] * np.exp(price_path)
        
        # Generar datos OHLCV
        data = pd.DataFrame({
            'close': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.002, n_samples))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_samples))),
            'volume': np.random.lognormal(8, 0.3, n_samples) * (1 + np.abs(returns) * 20)
        })
        
        data['open'] = data['close'].shift(1).fillna(data['close'].iloc[0])
        
        print(f"Datos generados para calibración de {self.pair_name}")
        print(f"  Rango de precios: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        print(f"  Volatilidad diaria: {data['close'].pct_change().std():.4f}")
        
        return data
    
    def calculate_market_features(self, data):
        """
        Calcula features necesarios para calibración
        """
        features = pd.DataFrame(index=data.index)
        
        # Returns básicos
        features['returns_1'] = data['close'].pct_change()
        features['returns_5'] = data['close'].pct_change(5)
        
        # Volatilidad realizada
        features['volatility_12'] = features['returns_1'].rolling(12).std()
        features['volatility_24'] = features['returns_1'].rolling(24).std()
        features['volatility_48'] = features['returns_1'].rolling(48).std()
        
        # ATR
        features['atr_14'] = self._calculate_atr(data, 14)
        features['atr_21'] = self._calculate_atr(data, 21)
        
        # Volume metrics
        features['volume_sma_20'] = data['volume'].rolling(20).mean()
        features['volume_ratio'] = data['volume'] / features['volume_sma_20']
        
        # Sentiment simulado basado en momentum
        momentum = data['close'] / data['close'].shift(24) - 1
        features['sentiment_score'] = np.tanh(momentum * 10)
        
        # Support/Resistance
        features['support_24'] = data['low'].rolling(24).min()
        features['resistance_24'] = data['high'].rolling(24).max()
        
        return features.fillna(method='ffill').fillna(0)
    
    def _calculate_atr(self, data, period=14):
        """Calcula Average True Range"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        return tr.rolling(period).mean()
    
    def calibrate_thresholds(self, data, features):
        """
        Calibra umbrales automáticamente usando grid search
        """
        print(f"\n=== CALIBRANDO UMBRALES PARA {self.pair_name} ===")
        
        # Definir grid de parámetros para calibración
        param_grid = {
            'volatility_multiplier': [0.3, 0.5, 0.7, 0.9, 1.1, 1.3],
            'atr_multiplier': [0.5, 0.8, 1.0, 1.2, 1.5],
            'sentiment_weight': [0.1, 0.2, 0.3, 0.4, 0.5],
            'volume_threshold': [1.0, 1.2, 1.5, 1.8, 2.0]
        }
        
        best_score = -np.inf
        best_params = None
        calibration_results = []
        
        print(f"Evaluando {len(list(ParameterGrid(param_grid)))} combinaciones de parámetros...")
        
        for i, params in enumerate(ParameterGrid(param_grid)):
            if i % 50 == 0:
                print(f"  Progreso: {i}/{len(list(ParameterGrid(param_grid)))}")
            
            # Generar targets con estos parámetros
            targets = self._generate_targets_with_params(features, params)
            
            # Evaluar calidad de la distribución
            score = self._evaluate_threshold_quality(targets, features)
            
            calibration_results.append({
                'params': params.copy(),
                'score': score,
                'targets': targets
            })
            
            if score > best_score:
                best_score = score
                best_params = params.copy()
        
        self.best_params = best_params
        self.calibration_history = calibration_results
        
        print(f"\n--- MEJORES PARÁMETROS ENCONTRADOS ---")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"Score de calidad: {best_score:.3f}")
        
        # Analizar distribución con mejores parámetros
        best_targets = self._generate_targets_with_params(features, best_params)
        self._analyze_distribution(best_targets)
        
        return best_params
    
    def _generate_targets_with_params(self, features, params):
        """
        Genera targets usando parámetros específicos
        """
        targets = []
        
        for i in range(1, len(features) - 1):
            future_return = features.iloc[i+1]['returns_1']
            current_volatility = features.iloc[i]['volatility_24']
            current_atr = features.iloc[i]['atr_14']
            sentiment = features.iloc[i]['sentiment_score']
            volume_signal = features.iloc[i]['volume_ratio']
            
            # Calcular threshold dinámico
            base_threshold = params['volatility_multiplier'] * current_volatility
            atr_adjustment = params['atr_multiplier'] * current_atr * 0.01
            sentiment_adjustment = params['sentiment_weight'] * abs(sentiment) * 0.1
            volume_adjustment = 0.1 if volume_signal > params['volume_threshold'] else 0
            
            final_threshold = base_threshold + atr_adjustment + sentiment_adjustment + volume_adjustment
            
            # Clasificar
            if future_return > final_threshold:
                targets.append(2)  # BUY
            elif future_return < -final_threshold:
                targets.append(0)  # SELL
            else:
                targets.append(1)  # HOLD
        
        return np.array(targets)
    
    def _evaluate_threshold_quality(self, targets, features):
        """
        Evalúa la calidad de los umbrales basado en métricas de trading
        """
        if len(targets) == 0:
            return -1000
        
        # Distribución de clases
        unique, counts = np.unique(targets, return_counts=True)
        class_proportions = counts / len(targets)
        
        # Penalizar distribuciones muy desequilibradas
        target_proportion = 1/3
        balance_penalty = sum(abs(prop - target_proportion) for prop in class_proportions)
        
        # Penalizar si alguna clase tiene muy pocas muestras
        min_samples_penalty = 0
        for count in counts:
            if count < 0.1 * len(targets):  # Menos del 10%
                min_samples_penalty += 10
        
        # Evaluar calidad de señales
        signal_quality = self._evaluate_signal_quality(targets, features)
        
        # Score combinado
        score = signal_quality - balance_penalty * 10 - min_samples_penalty
        
        return score
    
    def _evaluate_signal_quality(self, targets, features):
        """
        Evalúa la calidad de las señales generadas
        """
        if len(targets) == 0:
            return 0
        
        quality_score = 0
        
        # Evaluar coherencia de señales con momentum
        for i in range(1, len(targets)):
            if i >= len(features) - 1:
                break
                
            momentum = features.iloc[i]['returns_5']
            signal = targets[i]
            
            # Recompensar coherencia
            if signal == 2 and momentum > 0:  # BUY con momentum positivo
                quality_score += 1
            elif signal == 0 and momentum < 0:  # SELL con momentum negativo
                quality_score += 1
            elif signal == 1:  # HOLD es neutral
                quality_score += 0.5
        
        return quality_score / len(targets) if len(targets) > 0 else 0
    
    def _analyze_distribution(self, targets):
        """
        Analiza la distribución de señales
        """
        class_names = ['SELL', 'HOLD', 'BUY']
        unique, counts = np.unique(targets, return_counts=True)
        
        print(f"\n--- DISTRIBUCIÓN DE SEÑALES CALIBRADAS ---")
        for i, class_name in enumerate(class_names):
            if i in unique:
                idx = list(unique).index(i)
                percentage = counts[idx] / len(targets) * 100
                print(f"  {class_name}: {counts[idx]} ({percentage:.1f}%)")
            else:
                print(f"  {class_name}: 0 (0.0%)")
    
    def save_calibration_results(self, filename=None):
        """
        Guarda los resultados de calibración
        """
        if filename is None:
            filename = f"calibration_{self.pair_name}.json"
        
        results = {
            'pair': self.pair_name,
            'best_params': self.best_params,
            'total_combinations_tested': len(self.calibration_history)
        }
        
        print(f"Resultados de calibración guardados en {filename}")
        return results
    
    def get_optimized_config(self):
        """
        Retorna configuración optimizada para usar en ProductionTCNEnsemble
        """
        if self.best_params is None:
            raise ValueError("Debe ejecutar calibrate_thresholds() primero")
        
        # Configuración base específica por par
        base_configs = {
            "BTCUSDT": {
                'sequence_length': 48,
                'step_size': 24,
                'tcn_layers': 6,
                'filters': [32, 64, 128, 128, 64, 32],
                'dropout_rate': 0.35,
                'learning_rate': 2e-4,
                'trend_sensitivity': 0.8,
                'volume_importance': 0.4,
                'price_volatility': 0.6,
            },
            "ETHUSDT": {
                'sequence_length': 36,
                'step_size': 18,
                'tcn_layers': 6,
                'filters': [32, 64, 96, 96, 64, 32],
                'dropout_rate': 0.4,
                'learning_rate': 3e-4,
                'trend_sensitivity': 0.6,
                'volume_importance': 0.3,
                'price_volatility': 0.8,
            },
            "BNBUSDT": {
                'sequence_length': 30,
                'step_size': 15,
                'tcn_layers': 5,
                'filters': [32, 64, 96, 64, 32],
                'dropout_rate': 0.45,
                'learning_rate': 4e-4,
                'trend_sensitivity': 0.5,
                'volume_importance': 0.2,
                'price_volatility': 1.0,
            }
        }
        
        # Combinar con parámetros calibrados
        optimized_config = base_configs.get(self.pair_name, base_configs["BTCUSDT"]).copy()
        optimized_config.update(self.best_params)
        
        return optimized_config

def calibrate_all_pairs():
    """
    Calibra umbrales para todos los pares principales
    """
    print("=== CALIBRADOR AUTOMÁTICO DE UMBRALES ===")
    print("Optimizando parámetros para múltiples pares\n")
    
    pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    calibrated_configs = {}
    
    for pair in pairs:
        print(f"\n{'='*50}")
        print(f"CALIBRANDO {pair}")
        print('='*50)
        
        # Crear calibrador
        calibrator = ThresholdCalibrator(pair_name=pair)
        
        # Generar datos de mercado
        data = calibrator.generate_market_data(n_samples=2000)
        
        # Calcular features
        features = calibrator.calculate_market_features(data)
        
        # Calibrar umbrales
        best_params = calibrator.calibrate_thresholds(data, features)
        
        # Obtener configuración optimizada
        optimized_config = calibrator.get_optimized_config()
        calibrated_configs[pair] = optimized_config
        
        # Guardar resultados
        calibrator.save_calibration_results()
    
    # Resumen final
    print(f"\n{'='*60}")
    print("RESUMEN CALIBRACIÓN - TODOS LOS PARES")
    print('='*60)
    
    for pair, config in calibrated_configs.items():
        print(f"\n{pair} - Parámetros optimizados:")
        print(f"  Volatility Multiplier: {config['volatility_multiplier']}")
        print(f"  ATR Multiplier: {config['atr_multiplier']}")
        print(f"  Sentiment Weight: {config['sentiment_weight']}")
        print(f"  Volume Threshold: {config['volume_threshold']}")
    
    print(f"\n✅ Calibración completada para {len(pairs)} pares")
    print(f"✅ Configuraciones optimizadas generadas")
    print(f"✅ Listo para integrar con ProductionTCNEnsemble")
    
    return calibrated_configs

if __name__ == "__main__":
    calibrate_all_pairs() 