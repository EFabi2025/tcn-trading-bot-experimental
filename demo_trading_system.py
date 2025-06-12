#!/usr/bin/env python3
"""
DEMO TRADING SYSTEM - Demostración del sistema funcionando
Simula el funcionamiento completo del sistema de trading
"""

import asyncio
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Simular datos de mercado en tiempo real
class MarketDataSimulator:
    """Simulador de datos de mercado"""
    
    def __init__(self):
        self.pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        self.current_prices = {
            "BTCUSDT": 95000.0,
            "ETHUSDT": 3500.0,
            "BNBUSDT": 650.0
        }
        self.price_history = {pair: [] for pair in self.pairs}
    
    def generate_price_movement(self, pair: str) -> float:
        """Generar movimiento realista de precio"""
        base_vol = {
            "BTCUSDT": 0.02,  # 2% volatilidad
            "ETHUSDT": 0.025, # 2.5% volatilidad
            "BNBUSDT": 0.03   # 3% volatilidad
        }
        
        # Generar cambio de precio con tendencia
        random_factor = np.random.normal(0, base_vol[pair])
        trend_factor = np.random.choice([-0.001, 0, 0.001], p=[0.3, 0.4, 0.3])
        
        change = random_factor + trend_factor
        new_price = self.current_prices[pair] * (1 + change)
        
        self.current_prices[pair] = new_price
        self.price_history[pair].append({
            'timestamp': datetime.now(),
            'price': new_price,
            'change': change
        })
        
        return new_price
    
    def get_market_data(self, pair: str) -> Dict:
        """Obtener datos de mercado simulados"""
        price = self.generate_price_movement(pair)
        
        # Simular datos de vela
        high = price * np.random.uniform(1.0, 1.005)
        low = price * np.random.uniform(0.995, 1.0)
        volume = np.random.uniform(1000, 5000)
        
        return {
            'symbol': pair,
            'price': price,
            'high': high,
            'low': low,
            'volume': volume,
            'timestamp': datetime.now()
        }

class TradingDecisionEngine:
    """Motor de decisiones de trading simulado"""
    
    def __init__(self):
        # Simular el comportamiento del TCN real
        self.model_performance = {
            "BTCUSDT": {"accuracy": 0.738, "confidence": 0.901},
            "ETHUSDT": {"accuracy": 0.628, "confidence": 0.859},
            "BNBUSDT": {"accuracy": 0.629, "confidence": 0.843}
        }
        
        self.prediction_history = []
    
    def analyze_market_conditions(self, market_data: Dict) -> Dict:
        """Analizar condiciones de mercado"""
        price = market_data['price']
        
        # Simular análisis técnico
        volatility = np.random.uniform(0.01, 0.05)
        trend_strength = np.random.uniform(0.2, 0.8)
        rsi = np.random.uniform(30, 70)
        
        return {
            'volatility': volatility,
            'trend_strength': trend_strength,
            'rsi': rsi,
            'support_level': price * 0.98,
            'resistance_level': price * 1.02
        }
    
    def make_trading_decision(self, pair: str, market_data: Dict, analysis: Dict) -> Dict:
        """Hacer decisión de trading"""
        
        # Simular predicción del TCN
        signals = ['SELL', 'HOLD', 'BUY']
        
        # Usar lógica realista basada en análisis
        if analysis['rsi'] < 35 and analysis['trend_strength'] > 0.6:
            # Sobreventa con tendencia fuerte = BUY
            signal = 'BUY'
            confidence = np.random.uniform(0.75, 0.95)
        elif analysis['rsi'] > 65 and analysis['trend_strength'] > 0.6:
            # Sobrecompra con tendencia fuerte = SELL
            signal = 'SELL'
            confidence = np.random.uniform(0.75, 0.95)
        elif analysis['volatility'] < 0.02:
            # Baja volatilidad = HOLD
            signal = 'HOLD'
            confidence = np.random.uniform(0.60, 0.85)
        else:
            # Decision aleatoria con menor confianza
            signal = np.random.choice(signals)
            confidence = np.random.uniform(0.50, 0.75)
        
        prediction = {
            'pair': pair,
            'signal': signal,
            'confidence': confidence,
            'model_accuracy': self.model_performance[pair]['accuracy'],
            'probabilities': {
                'SELL': np.random.uniform(0.2, 0.4),
                'HOLD': np.random.uniform(0.1, 0.3),
                'BUY': np.random.uniform(0.3, 0.5)
            },
            'market_analysis': analysis,
            'timestamp': datetime.now()
        }
        
        # Ajustar probabilidades según señal
        if signal == 'BUY':
            prediction['probabilities']['BUY'] = confidence
            prediction['probabilities']['SELL'] = (1 - confidence) * 0.6
            prediction['probabilities']['HOLD'] = (1 - confidence) * 0.4
        elif signal == 'SELL':
            prediction['probabilities']['SELL'] = confidence
            prediction['probabilities']['BUY'] = (1 - confidence) * 0.6
            prediction['probabilities']['HOLD'] = (1 - confidence) * 0.4
        else:  # HOLD
            prediction['probabilities']['HOLD'] = confidence
            prediction['probabilities']['BUY'] = (1 - confidence) * 0.5
            prediction['probabilities']['SELL'] = (1 - confidence) * 0.5
        
        self.prediction_history.append(prediction)
        return prediction

class RiskManagerDemo:
    """Gestor de riesgo para demo"""
    
    def __init__(self):
        self.max_position_size = 0.05  # 5% del balance
        self.min_confidence = 0.75
        self.daily_trades = 0
        self.max_daily_trades = 10
        self.balance = 10000.0  # $10k demo
        self.open_positions = {}
        
    def can_trade(self, prediction: Dict) -> bool:
        """Verificar si se puede tradear"""
        pair = prediction['pair']
        signal = prediction['signal']
        confidence = prediction['confidence']
        
        # Verificar límites
        if self.daily_trades >= self.max_daily_trades:
            return False
        
        if confidence < self.min_confidence:
            return False
        
        if signal == 'HOLD':
            return False
        
        if pair in self.open_positions:
            return False
            
        return True
    
    def calculate_position_size(self, price: float) -> float:
        """Calcular tamaño de posición"""
        max_value = self.balance * self.max_position_size
        return max_value / price
    
    def execute_trade(self, prediction: Dict, market_data: Dict) -> Dict:
        """Ejecutar trade simulado"""
        pair = prediction['pair']
        signal = prediction['signal']
        price = market_data['price']
        
        position_size = self.calculate_position_size(price)
        trade_value = position_size * price
        
        trade = {
            'pair': pair,
            'signal': signal,
            'price': price,
            'quantity': position_size,
            'value': trade_value,
            'confidence': prediction['confidence'],
            'timestamp': datetime.now(),
            'status': 'EXECUTED'
        }
        
        self.daily_trades += 1
        self.open_positions[pair] = trade
        
        print(f"""
🎯 TRADE EJECUTADO:
Par: {pair}
Señal: {signal}
Confianza: {prediction['confidence']:.3f}
Precio: ${price:,.2f}
Cantidad: {position_size:.6f}
Valor: ${trade_value:.2f}
Accuracy del modelo: {prediction['model_accuracy']:.3f}
""")
        
        return trade

class TradingSystemDemo:
    """Sistema de trading completo para demo"""
    
    def __init__(self):
        self.market_simulator = MarketDataSimulator()
        self.decision_engine = TradingDecisionEngine()
        self.risk_manager = RiskManagerDemo()
        self.running = False
        
    async def run_demo(self, duration_minutes: int = 5):
        """Ejecutar demo del sistema"""
        print("🚀 INICIANDO DEMO DEL SISTEMA DE TRADING TCN")
        print("="*60)
        print(f"⏱️  Duración: {duration_minutes} minutos")
        print(f"💰 Balance inicial: ${self.risk_manager.balance:,.2f}")
        print(f"📊 Pares monitoreados: {', '.join(self.market_simulator.pairs)}")
        print(f"🎯 Confianza mínima: {self.risk_manager.min_confidence:.1%}")
        print("="*60)
        
        self.running = True
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        cycle = 0
        
        while self.running and datetime.now() < end_time:
            cycle += 1
            print(f"\\n🔄 CICLO {cycle} - {datetime.now().strftime('%H:%M:%S')}")
            print("-" * 50)
            
            # Procesar cada par
            for pair in self.market_simulator.pairs:
                await self.process_pair(pair)
            
            # Mostrar resumen del ciclo
            await self.show_cycle_summary()
            
            # Esperar antes del siguiente ciclo
            await asyncio.sleep(15)  # 15 segundos entre ciclos
        
        await self.show_final_summary()
    
    async def process_pair(self, pair: str):
        """Procesar un par específico"""
        try:
            # Obtener datos de mercado
            market_data = self.market_simulator.get_market_data(pair)
            
            # Analizar condiciones
            analysis = self.decision_engine.analyze_market_conditions(market_data)
            
            # Hacer predicción
            prediction = self.decision_engine.make_trading_decision(pair, market_data, analysis)
            
            # Mostrar predicción
            print(f"📈 {pair}: {prediction['signal']} (conf: {prediction['confidence']:.3f}) - ${market_data['price']:,.2f}")
            
            # Verificar si se puede tradear
            if self.risk_manager.can_trade(prediction):
                trade = self.risk_manager.execute_trade(prediction, market_data)
                return trade
            else:
                if prediction['confidence'] < self.risk_manager.min_confidence:
                    print(f"   ⚠️  Confianza insuficiente: {prediction['confidence']:.3f}")
                elif prediction['signal'] == 'HOLD':
                    print(f"   💤 Señal HOLD - Sin acción")
                else:
                    print(f"   ⏸️  No se puede tradear: Límites de riesgo")
        
        except Exception as e:
            print(f"❌ Error procesando {pair}: {e}")
        
        return None
    
    async def show_cycle_summary(self):
        """Mostrar resumen del ciclo"""
        total_predictions = len(self.decision_engine.prediction_history)
        if total_predictions > 0:
            recent_predictions = self.decision_engine.prediction_history[-3:]  # Últimas 3
            avg_confidence = np.mean([p['confidence'] for p in recent_predictions])
            
            signals_count = {}
            for p in recent_predictions:
                signals_count[p['signal']] = signals_count.get(p['signal'], 0) + 1
            
            print(f"📊 Resumen ciclo:")
            print(f"   Confianza promedio: {avg_confidence:.3f}")
            print(f"   Señales: {signals_count}")
            print(f"   Trades diarios: {self.risk_manager.daily_trades}/{self.risk_manager.max_daily_trades}")
            print(f"   Posiciones abiertas: {len(self.risk_manager.open_positions)}")
    
    async def show_final_summary(self):
        """Mostrar resumen final"""
        print("\\n" + "="*60)
        print("🎉 DEMO COMPLETADO - RESUMEN FINAL")
        print("="*60)
        
        total_predictions = len(self.decision_engine.prediction_history)
        
        if total_predictions > 0:
            # Estadísticas de predicciones
            confidences = [p['confidence'] for p in self.decision_engine.prediction_history]
            signals = [p['signal'] for p in self.decision_engine.prediction_history]
            
            signal_counts = {}
            for signal in signals:
                signal_counts[signal] = signal_counts.get(signal, 0) + 1
            
            print(f"📊 ESTADÍSTICAS:")
            print(f"   Total predicciones: {total_predictions}")
            print(f"   Confianza promedio: {np.mean(confidences):.3f}")
            print(f"   Confianza máxima: {np.max(confidences):.3f}")
            print(f"   Distribución señales: {signal_counts}")
            print(f"   Trades ejecutados: {self.risk_manager.daily_trades}")
            print(f"   Posiciones abiertas: {len(self.risk_manager.open_positions)}")
            
            print(f"\\n💰 GESTIÓN DE RIESGO:")
            print(f"   Balance inicial: ${self.risk_manager.balance:,.2f}")
            print(f"   Límite por posición: {self.risk_manager.max_position_size:.1%}")
            print(f"   Confianza mínima: {self.risk_manager.min_confidence:.1%}")
            
            print(f"\\n🎯 RENDIMIENTO DEL MODELO:")
            for pair in self.market_simulator.pairs:
                perf = self.decision_engine.model_performance[pair]
                print(f"   {pair}: Accuracy {perf['accuracy']:.3f}, Conf. base {perf['confidence']:.3f}")
        
        print("\\n✅ Demo completado exitosamente!")
        print("🚀 Sistema listo para integración real con Binance")
        
    async def stop(self):
        """Detener demo"""
        self.running = False

async def main():
    """Función principal del demo"""
    print("🎯 BINANCE TCN TRADING SYSTEM - DEMO")
    print("Demostración del sistema completo funcionando")
    print()
    
    # Crear y ejecutar demo
    demo_system = TradingSystemDemo()
    
    try:
        await demo_system.run_demo(duration_minutes=3)  # Demo de 3 minutos
    except KeyboardInterrupt:
        print("\\n⏹️  Demo interrumpido por usuario")
        await demo_system.stop()

if __name__ == "__main__":
    asyncio.run(main()) 