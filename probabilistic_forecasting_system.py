#!/usr/bin/env python3
"""
🎯 SISTEMA DE PREDICCIÓN PROBABILÍSTICA CON DISTRIBUCIÓN CHI-CUADRADO
Utiliza los predictores de 1m, 3m y 5m para generar predicciones con horizonte temporal

Características principales:
- Combina predicciones de múltiples timeframes (1m, 3m, 5m)
- Utiliza distribución chi-cuadrado para modelar incertidumbre temporal
- Genera predicciones con horizonte de minutos en el futuro
- Proporciona intervalos de confianza y probabilidades
- Integra contexto de mercado para ajustar predicciones
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2
from typing import Dict, List, Tuple, Optional, Any
import asyncio
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Importar predictores existentes
from predictor1m_talib import get_ensemble_ready_prediction_talib, ProbabilisticPredictorTalib
from predictor3m_core_optimized import get_ensemble_ready_prediction_core_3m, CoreProbabilisticPredictor3m
from predictor5m_talib import get_ensemble_ready_prediction_5m_talib, ProbabilisticPredictor5mTalib
from tcn_ensemble_predictor import TCNEnsemblePredictor

class ProbabilisticForecastingSystem:
    """
    🎯 Sistema de predicción probabilística con distribución chi-cuadrado
    
    Utiliza los predictores de 1m, 3m y 5m para generar predicciones con:
    - Horizonte temporal (minutos en el futuro)
    - Intervalos de confianza
    - Probabilidades de diferentes escenarios
    - Ajuste basado en contexto de mercado
    """
    
    def __init__(self):
        self.timeframe_weights = {
            '1m': 0.40,  # Mayor peso para predicciones a corto plazo
            '3m': 0.35,  # Peso medio para tendencias intermedias
            '5m': 0.25   # Menor peso para tendencias a largo plazo
        }
        
        # Parámetros de la distribución chi-cuadrado
        self.chi_square_params = {
            'df': 3,  # Grados de libertad (basado en 3 timeframes)
            'scale': 1.0,  # Escala de la distribución
            'location': 0.0  # Ubicación de la distribución
        }
        
        # Horizonte de predicción en minutos
        self.prediction_horizons = [1, 3, 5, 10, 15, 30, 60]
        
        # Contexto de mercado para ajustar predicciones
        self.market_context_weights = {
            'low_volatility': 1.2,    # Aumentar confianza en mercados tranquilos
            'normal_volatility': 1.0,  # Peso normal
            'high_volatility': 0.8,    # Reducir confianza en mercados volátiles
            'extreme_volatility': 0.6  # Reducir significativamente en crisis
        }
        
        # Instanciar predictores y otros componentes una sola vez
        self.tcn_predictor = TCNEnsemblePredictor()
        
        print("🎯 Sistema de Predicción Probabilística inicializado")
        print(f"   📊 Pesos por timeframe: {self.timeframe_weights}")
        print(f"   📈 Horizonte de predicción: {self.prediction_horizons} minutos")
        print(f"   🎲 Distribución chi-cuadrado: df={self.chi_square_params['df']}")
    
    def get_timeframe_predictions(self, symbol: str) -> Dict[str, Dict]:
        """
        Obtener predicciones de todos los timeframes disponibles
        
        Args:
            symbol: Símbolo del par de trading (ej: 'BTCUSDT')
            
        Returns:
            Dict con predicciones de cada timeframe
        """
        predictions = {}
        
        print(f"🔮 Obteniendo predicciones para {symbol}...")
        
        # Predicción 1m
        try:
            pred_1m = get_ensemble_ready_prediction_talib(symbol)
            if pred_1m:
                predictions['1m'] = pred_1m
                print(f"   ✅ 1m: {pred_1m.get('primary_signal', 'N/A')} (conf: {pred_1m.get('confidence', 0)*100:.1f}%)")
            else:
                print(f"   ❌ 1m: Sin predicción disponible")
        except Exception as e:
            print(f"   ⚠️ 1m: Error - {e}")
        
        # Predicción 3m
        try:
            pred_3m = get_ensemble_ready_prediction_core_3m(symbol)
            if pred_3m:
                predictions['3m'] = pred_3m
                print(f"   ✅ 3m: {pred_3m.get('primary_signal', 'N/A')} (conf: {pred_3m.get('confidence', 0)*100:.1f}%)")
            else:
                print(f"   ❌ 3m: Sin predicción disponible")
        except Exception as e:
            print(f"   ⚠️ 3m: Error - {e}")
        
        # Predicción 5m
        try:
            pred_5m = get_ensemble_ready_prediction_5m_talib(symbol)
            if pred_5m:
                predictions['5m'] = pred_5m
                print(f"   ✅ 5m: {pred_5m.get('primary_signal', 'N/A')} (conf: {pred_5m.get('confidence', 0)*100:.1f}%)")
            else:
                print(f"   ❌ 5m: Sin predicción disponible")
        except Exception as e:
            print(f"   ⚠️ 5m: Error - {e}")
        
        return predictions
    
    def calculate_ensemble_probabilities(self, predictions: Dict[str, Dict]) -> Dict[str, float]:
        """
        Calcular probabilidades del ensemble basado en predicciones de múltiples timeframes
        
        Args:
            predictions: Dict con predicciones de cada timeframe
            
        Returns:
            Dict con probabilidades BUY, HOLD, SELL
        """
        if not predictions:
            return {'BUY': 0.33, 'HOLD': 0.34, 'SELL': 0.33}
        
        # Extraer probabilidades de cada timeframe
        timeframe_probs = {}
        for tf, pred in predictions.items():
            if 'probabilities' in pred:
                timeframe_probs[tf] = pred['probabilities']
            else:
                # Convertir señal a probabilidades si no están disponibles
                signal = pred.get('primary_signal', 'HOLD')
                confidence = pred.get('confidence', 0.5)
                
                if signal == 'BUY':
                    timeframe_probs[tf] = {
                        'BUY': confidence,
                        'HOLD': (1 - confidence) * 0.5,
                        'SELL': (1 - confidence) * 0.5
                    }
                elif signal == 'SELL':
                    timeframe_probs[tf] = {
                        'BUY': (1 - confidence) * 0.5,
                        'HOLD': (1 - confidence) * 0.5,
                        'SELL': confidence
                    }
                else:  # HOLD
                    timeframe_probs[tf] = {
                        'BUY': (1 - confidence) * 0.3,
                        'HOLD': confidence,
                        'SELL': (1 - confidence) * 0.3
                    }
        
        # Calcular probabilidades ponderadas
        ensemble_probs = {'BUY': 0.0, 'HOLD': 0.0, 'SELL': 0.0}
        total_weight = 0.0
        
        for tf, probs in timeframe_probs.items():
            weight = self.timeframe_weights.get(tf, 0.0)
            total_weight += weight
            
            for signal, prob in probs.items():
                ensemble_probs[signal] += prob * weight
        
        # Normalizar probabilidades
        if total_weight > 0:
            for signal in ensemble_probs:
                ensemble_probs[signal] /= total_weight
        
        # Asegurar que sumen 1.0
        total_prob = sum(ensemble_probs.values())
        if total_prob > 0:
            for signal in ensemble_probs:
                ensemble_probs[signal] /= total_prob
        
        return ensemble_probs
    
    def calculate_chi_square_uncertainty(self, horizon_minutes: int, market_context: str = 'normal_volatility') -> float:
        """
        Calcular incertidumbre usando distribución chi-cuadrado con reducción para primeros 10 minutos
        
        Args:
            horizon_minutes: Horizonte de predicción en minutos
            market_context: Contexto de mercado para ajustar incertidumbre
            
        Returns:
            Factor de incertidumbre (0.0 a 1.0)
        """
        # Reducir incertidumbre para los primeros 10 minutos
        if horizon_minutes <= 10:
            # Aplicar factor de reducción progresiva para primeros 10 minutos
            reduction_factor = 1.0 - (horizon_minutes / 10.0) * 0.4  # Reducción del 40% máximo
            base_uncertainty = 0.15 + (horizon_minutes / 10.0) * 0.15  # Base de 15% a 30%
        else:
            # Para horizontes mayores a 10 minutos, usar cálculo original
            reduction_factor = 1.0
            base_uncertainty = 0.3 + (horizon_minutes / 60.0) * 0.4  # Base de 30% a 70%
        
        # Ajustar grados de libertad basado en horizonte temporal (más conservador para primeros 10m)
        if horizon_minutes <= 10:
            df = self.chi_square_params['df'] + (horizon_minutes / 20.0)  # Crecimiento más lento
        else:
            df = self.chi_square_params['df'] + (horizon_minutes / 10.0)  # Crecimiento normal
        
        # Calcular percentil de la distribución chi-cuadrado
        # Usar percentil 70 para primeros 10m (más conservador), 75 para el resto
        percentile = 0.70 if horizon_minutes <= 10 else 0.75
        chi_value = chi2.ppf(percentile, df)
        
        # Normalizar a rango [0, 1] con base ajustada
        uncertainty = min(1.0, base_uncertainty + (chi_value / 15.0))
        
        # Aplicar factor de reducción para primeros 10 minutos
        uncertainty *= reduction_factor
        
        # Ajustar por contexto de mercado
        context_weight = self.market_context_weights.get(market_context, 1.0)
        uncertainty *= context_weight
        
        return uncertainty
    
    def estimate_price_movement(self, symbol: str, current_price: float, ensemble_probs: Dict[str, float], 
                              horizon_minutes: int, market_context: str = 'normal_volatility') -> Dict[str, float]:
        """
        Estimar movimiento de precio basado en probabilidades del ensemble
        
        Args:
            symbol: Símbolo del par de trading
            current_price: Precio actual
            ensemble_probs: Probabilidades del ensemble (BUY, HOLD, SELL)
            horizon_minutes: Horizonte temporal en minutos
            market_context: Contexto de mercado
            
        Returns:
            Dict con estimaciones de precio
        """
        # Obtener volatilidad histórica para el símbolo
        volatility = self._get_historical_volatility(symbol, horizon_minutes)
        
        # Calcular incertidumbre temporal
        uncertainty = self.calculate_chi_square_uncertainty(horizon_minutes, market_context)
        
        # Ajustar volatilidad por incertidumbre temporal
        adjusted_volatility = volatility * (1 + uncertainty * 0.5)
        
        # Calcular movimiento esperado basado en probabilidades
        buy_prob = ensemble_probs.get('BUY', 0.33)
        sell_prob = ensemble_probs.get('SELL', 0.33)
        hold_prob = ensemble_probs.get('HOLD', 0.34)
        
        # Calcular dirección esperada (-1 a 1)
        expected_direction = (buy_prob - sell_prob) / (buy_prob + sell_prob + hold_prob)
        
        # Calcular magnitud del movimiento basada en volatilidad y confianza
        confidence = max(buy_prob, sell_prob, hold_prob)
        movement_magnitude = adjusted_volatility * confidence * abs(expected_direction)
        
        # Calcular precios estimados
        price_change_percent = expected_direction * movement_magnitude * 100
        
        # Precio esperado
        expected_price = current_price * (1 + price_change_percent / 100)
        
        # Calcular intervalos de confianza usando distribución normal
        # 68% de confianza (1 desviación estándar)
        std_dev_68 = adjusted_volatility * current_price * 0.01
        price_68_lower = expected_price - std_dev_68
        price_68_upper = expected_price + std_dev_68
        
        # 95% de confianza (2 desviaciones estándar)
        std_dev_95 = adjusted_volatility * current_price * 0.02
        price_95_lower = expected_price - std_dev_95
        price_95_upper = expected_price + std_dev_95
        
        return {
            'current_price': current_price,
            'expected_price': expected_price,
            'price_change_percent': price_change_percent,
            'price_change_absolute': expected_price - current_price,
            'volatility': adjusted_volatility,
            'confidence_intervals': {
                '68%': {
                    'lower': max(0, price_68_lower),
                    'upper': price_68_upper,
                    'center': expected_price
                },
                '95%': {
                    'lower': max(0, price_95_lower),
                    'upper': price_95_upper,
                    'center': expected_price
                }
            },
            'movement_analysis': {
                'direction': 'BULLISH' if expected_direction > 0.1 else 'BEARISH' if expected_direction < -0.1 else 'SIDEWAYS',
                'strength': abs(expected_direction),
                'confidence': confidence,
                'uncertainty': uncertainty
            }
        }
    
    def _get_historical_volatility(self, symbol: str, horizon_minutes: int) -> float:
        """
        Obtener volatilidad histórica para el símbolo
        
        Args:
            symbol: Símbolo del par de trading
            horizon_minutes: Horizonte temporal en minutos
            
        Returns:
            Volatilidad histórica (porcentaje)
        """
        # Volatilidades típicas por símbolo (en % por minuto)
        volatility_map = {
            'BTCUSDT': 0.15,   # Bitcoin: ~15% anual
            'ETHUSDT': 0.20,   # Ethereum: ~20% anual
            'BNBUSDT': 0.18,   # BNB: ~18% anual
            'ADAUSDT': 0.25,   # Cardano: ~25% anual
            'DOTUSDT': 0.22,   # Polkadot: ~22% anual
            'POLUSDT': 0.25,   # Polygon: ~25% anual
            'LTCUSDT': 0.20,   # Litecoin: ~20% anual
            'XRPUSDT': 0.24,   # XRP: ~24% anual
        }
        
        # Obtener volatilidad base
        base_volatility = volatility_map.get(symbol, 0.20)  # Default 20%
        
        # Ajustar por horizonte temporal (volatilidad aumenta con el tiempo)
        time_factor = 1 + (horizon_minutes / 60.0) * 0.1  # 10% más por hora
        
        return base_volatility * time_factor
    
    def generate_temporal_predictions(self, symbol: str, ensemble_probs: Dict[str, float], 
                                    market_context: str = 'normal_volatility', current_price: float = None) -> Dict[int, Dict]:
        """
        Generar predicciones para diferentes horizontes temporales
        
        Args:
            symbol: Símbolo del par de trading
            ensemble_probs: Probabilidades del ensemble
            market_context: Contexto de mercado
            current_price: Precio actual (opcional)
            
        Returns:
            Dict con predicciones para cada horizonte temporal
        """
        temporal_predictions = {}
        
        print(f"⏰ Generando predicciones temporales para {symbol}...")
        
        for horizon in self.prediction_horizons:
            # Calcular incertidumbre para este horizonte
            uncertainty = self.calculate_chi_square_uncertainty(horizon, market_context)
            
            # Ajustar probabilidades basado en incertidumbre temporal
            adjusted_probs = {}
            for signal, prob in ensemble_probs.items():
                # Reducir confianza con el tiempo (menos reducción para primeros 10 minutos)
                if horizon <= 10:
                    # Reducción más suave para primeros 10 minutos
                    reduction_factor = 0.15  # Solo 15% de reducción
                else:
                    # Reducción normal para horizontes mayores
                    reduction_factor = 0.3   # 30% de reducción
                
                adjusted_prob = prob * (1 - uncertainty * reduction_factor)
                adjusted_probs[signal] = max(0.0, min(1.0, adjusted_prob))
            
            # Normalizar probabilidades ajustadas
            total_adj_prob = sum(adjusted_probs.values())
            if total_adj_prob > 0:
                for signal in adjusted_probs:
                    adjusted_probs[signal] /= total_adj_prob
            
            # Determinar señal principal
            primary_signal = max(adjusted_probs, key=adjusted_probs.get)
            confidence = adjusted_probs[primary_signal]
            
            # Calcular intervalo de confianza
            confidence_interval = self.calculate_confidence_interval(adjusted_probs, uncertainty)
            
            # Estimar movimiento de precio si se proporciona precio actual
            price_estimation = None
            if current_price is not None:
                price_estimation = self.estimate_price_movement(
                    symbol, current_price, adjusted_probs, horizon, market_context
                )
            
            temporal_predictions[horizon] = {
                'primary_signal': primary_signal,
                'confidence': confidence,
                'probabilities': adjusted_probs,
                'uncertainty': uncertainty,
                'confidence_interval': confidence_interval,
                'price_estimation': price_estimation,
                'horizon_minutes': horizon,
                'timestamp': datetime.now()
            }
            
            # Mostrar información de precio si está disponible
            if price_estimation:
                price_change = price_estimation['price_change_percent']
                expected_price = price_estimation['expected_price']
                print(f"   📊 {horizon:2d}m: {primary_signal} (conf: {confidence*100:.1f}%, incert: {uncertainty*100:.1f}%) | Precio: ${expected_price:.4f} ({price_change:+.2f}%)")
            else:
                print(f"   📊 {horizon:2d}m: {primary_signal} (conf: {confidence*100:.1f}%, incert: {uncertainty*100:.1f}%)")
        
        return temporal_predictions
    
    def calculate_confidence_interval(self, probabilities: Dict[str, float], uncertainty: float) -> Dict[str, float]:
        """
        Calcular intervalo de confianza para las probabilidades
        
        Args:
            probabilities: Probabilidades de cada señal
            uncertainty: Factor de incertidumbre
            
        Returns:
            Dict con intervalos de confianza
        """
        confidence_interval = {}
        
        for signal, prob in probabilities.items():
            # Calcular margen de error basado en incertidumbre
            margin_error = prob * uncertainty * 0.5
            
            confidence_interval[signal] = {
                'lower': max(0.0, prob - margin_error),
                'upper': min(1.0, prob + margin_error),
                'center': prob
            }
        
        return confidence_interval
    
    async def get_market_context(self, symbol: str) -> str:
        """
        Obtener contexto de mercado para ajustar predicciones (asíncrono)
        
        Args:
            symbol: Símbolo del par de trading
            
        Returns:
            Contexto de mercado ('low_volatility', 'normal_volatility', etc.)
        """
        try:
            # Usar la instancia de TCNEnsemblePredictor del sistema
            market_data = await self.tcn_predictor.get_market_data(symbol, '1m', hours=8)
            
            if market_data is not None and len(market_data) > 0:
                # Detectar contexto de mercado
                context = self.tcn_predictor.detect_market_context(symbol, market_data)
                volatility_regime = context.get('volatility_regime', 'normal_volatility')
                return volatility_regime
            else:
                return 'normal_volatility'
                
        except Exception as e:
            print(f"⚠️ Error obteniendo contexto de mercado: {e}")
            return 'normal_volatility'
    
    async def generate_comprehensive_forecast(self, symbol: str) -> Dict[str, Any]:
        """
        Generar predicción completa con horizonte temporal (asíncrono)
        
        Args:
            symbol: Símbolo del par de trading
            
        Returns:
            Dict con predicción completa
        """
        print(f"\n🎯 GENERANDO PREDICCIÓN PROBABILÍSTICA PARA {symbol}")
        print("=" * 60)
        
        # Obtener predicciones de timeframes
        timeframe_predictions = self.get_timeframe_predictions(symbol)
        
        if not timeframe_predictions:
            print("❌ No se pudieron obtener predicciones de ningún timeframe")
            return None
        
        # Calcular probabilidades del ensemble
        ensemble_probs = self.calculate_ensemble_probabilities(timeframe_predictions)
        
        print(f"\n📊 PROBABILIDADES DEL ENSEMBLE:")
        for signal, prob in ensemble_probs.items():
            print(f"   {signal}: {prob*100:.1f}%")
        
        # Obtener contexto de mercado y precio actual
        market_context = await self.get_market_context(symbol)
        print(f"\n🌍 CONTEXTO DE MERCADO: {market_context}")
        
        # Obtener precio actual
        current_price = await self._get_current_price(symbol)
        if current_price:
            print(f"💰 PRECIO ACTUAL: ${current_price:.4f}")
        else:
            print("⚠️ No se pudo obtener el precio actual")
        
        # Generar predicciones temporales con estimación de precio
        temporal_predictions = self.generate_temporal_predictions(
            symbol, ensemble_probs, market_context, current_price
        )
        
        # Crear resultado final
        result = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'current_price': current_price,
            'market_context': market_context,
            'ensemble_probabilities': ensemble_probs,
            'timeframe_predictions': timeframe_predictions,
            'temporal_predictions': temporal_predictions,
            'primary_signal': max(ensemble_probs, key=ensemble_probs.get),
            'confidence': ensemble_probs[max(ensemble_probs, key=ensemble_probs.get)],
            'uncertainty_factors': {
                'chi_square_df': self.chi_square_params['df'],
                'market_context_weight': self.market_context_weights.get(market_context, 1.0)
            }
        }
        
        print(f"\n🎯 PREDICCIÓN FINAL:")
        print(f"   📈 Señal principal: {result['primary_signal']}")
        print(f"   🎯 Confianza: {result['confidence']*100:.1f}%")
        print(f"   ⏰ Horizonte recomendado: {self.get_recommended_horizon(temporal_predictions)} minutos")
        
        # Mostrar estimación de precio si está disponible
        if current_price and temporal_predictions:
            recommended_horizon = self.get_recommended_horizon(temporal_predictions)
            if recommended_horizon in temporal_predictions:
                price_est = temporal_predictions[recommended_horizon].get('price_estimation')
                if price_est:
                    print(f"   💰 Precio esperado ({recommended_horizon}m): ${price_est['expected_price']:.4f} ({price_est['price_change_percent']:+.2f}%)")
        
        return result
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """
        Obtener precio actual del símbolo
        
        Args:
            symbol: Símbolo del par de trading
            
        Returns:
            Precio actual o None si no se puede obtener
        """
        try:
            # Usar la instancia de TCNEnsemblePredictor para obtener datos de mercado
            market_data = await self.tcn_predictor.get_market_data(symbol, '1m', hours=1)
            
            if market_data is not None and len(market_data) > 0:
                # Obtener el precio de cierre más reciente
                current_price = market_data['close'].iloc[-1]
                return float(current_price)
            else:
                return None
                
        except Exception as e:
            print(f"⚠️ Error obteniendo precio actual: {e}")
            return None
    
    def get_recommended_horizon(self, temporal_predictions: Dict[int, Dict]) -> int:
        """
        Obtener horizonte temporal recomendado basado en confianza
        
        Args:
            temporal_predictions: Predicciones temporales
            
        Returns:
            Horizonte recomendado en minutos
        """
        best_horizon = 1
        best_confidence = 0.0
        
        for horizon, pred in temporal_predictions.items():
            confidence = pred['confidence']
            uncertainty = pred['uncertainty']
            
            # Calcular score combinado (confianza alta, incertidumbre baja)
            score = confidence * (1 - uncertainty)
            
            if score > best_confidence:
                best_confidence = score
                best_horizon = horizon
        
        return best_horizon
    
    def print_forecast_summary(self, forecast: Dict[str, Any]):
        """
        Imprimir resumen de la predicción
        
        Args:
            forecast: Resultado de generate_comprehensive_forecast
        """
        if not forecast:
            print("❌ No hay predicción disponible")
            return
        
        print(f"\n📊 RESUMEN DE PREDICCIÓN PARA {forecast['symbol']}")
        print("=" * 50)
        
        print(f"🕐 Timestamp: {forecast['timestamp']}")
        print(f"🌍 Contexto: {forecast['market_context']}")
        print(f"📈 Señal principal: {forecast['primary_signal']}")
        print(f"🎯 Confianza: {forecast['confidence']*100:.1f}%")
        
        if forecast.get('current_price'):
            print(f"💰 Precio actual: ${forecast['current_price']:.4f}")
        
        print(f"\n⏰ PREDICCIONES TEMPORALES:")
        for horizon, pred in forecast['temporal_predictions'].items():
            price_info = ""
            if pred.get('price_estimation'):
                price_est = pred['price_estimation']
                price_info = f" | Precio: ${price_est['expected_price']:.4f} ({price_est['price_change_percent']:+.2f}%)"
            
            print(f"   {horizon:2d}m: {pred['primary_signal']} (conf: {pred['confidence']*100:.1f}%, incert: {pred['uncertainty']*100:.1f}%){price_info}")
        
        print(f"\n📊 PROBABILIDADES DEL ENSEMBLE:")
        for signal, prob in forecast['ensemble_probabilities'].items():
            print(f"   {signal}: {prob*100:.1f}%")
        
        print(f"\n🎯 HORIZONTE RECOMENDADO: {self.get_recommended_horizon(forecast['temporal_predictions'])} minutos")
        
        # Mostrar análisis de intervalos de confianza de precio
        if forecast.get('current_price') and forecast['temporal_predictions']:
            recommended_horizon = self.get_recommended_horizon(forecast['temporal_predictions'])
            if recommended_horizon in forecast['temporal_predictions']:
                price_est = forecast['temporal_predictions'][recommended_horizon].get('price_estimation')
                if price_est:
                    print(f"\n💰 ANÁLISIS DE PRECIO ({recommended_horizon}m):")
                    print(f"   📈 Precio esperado: ${price_est['expected_price']:.4f}")
                    print(f"   📊 Cambio: {price_est['price_change_percent']:+.2f}% (${price_est['price_change_absolute']:+.4f})")
                    print(f"   🎯 Volatilidad: {price_est['volatility']:.2f}%")
                    print(f"   📉 Intervalo 68%: ${price_est['confidence_intervals']['68%']['lower']:.4f} - ${price_est['confidence_intervals']['68%']['upper']:.4f}")
                    print(f"   📉 Intervalo 95%: ${price_est['confidence_intervals']['95%']['lower']:.4f} - ${price_est['confidence_intervals']['95%']['upper']:.4f}")
                    print(f"   🎯 Dirección: {price_est['movement_analysis']['direction']} (fuerza: {price_est['movement_analysis']['strength']:.2f})")

# Función de conveniencia para uso rápido
async def generate_probabilistic_forecast(symbol: str) -> Dict[str, Any]:
    """
    Función de conveniencia para generar predicción probabilística (asíncrona).
    Nota: Para uso en producción, se recomienda instanciar el sistema una vez y reutilizarlo.
    
    Args:
        symbol: Símbolo del par de trading
        
    Returns:
        Dict con predicción completa
    """
    system = ProbabilisticForecastingSystem()
    forecast = await system.generate_comprehensive_forecast(symbol)
    
    if forecast:
        system.print_forecast_summary(forecast)
    
    return forecast

# Ejemplo de uso
async def main():
    """Función principal para ejecutar el ejemplo de forma asíncrona."""
    print("🚀 Iniciando ejemplo de predicción probabilística...")
    
    # Se recomienda instanciar el sistema una vez y reutilizarlo en una aplicación real
    system = ProbabilisticForecastingSystem()
    
    # Generar predicción para BTCUSDT
    forecast = await system.generate_comprehensive_forecast('BTCUSDT')
    
    if forecast:
        # El resumen ya se imprime dentro de generate_comprehensive_forecast
        print(f"\n✅ Predicción generada exitosamente para {forecast['symbol']}")
    else:
        print("❌ No se pudo generar la predicción")

if __name__ == "__main__":
    # Ejecutar la función principal de forma asíncrona
    asyncio.run(main())