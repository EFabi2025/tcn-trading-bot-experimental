#!/usr/bin/env python3
"""
BACKTESTING SYSTEM - Sistema de backtesting histórico
Validación del rendimiento del modelo TCN con datos históricos reales
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
from final_real_binance_predictor import OptimizedBinanceData, OptimizedTCNPredictor, CompatibleFeatureEngine
import warnings
warnings.filterwarnings('ignore')

class HistoricalDataProvider:
    """Proveedor de datos históricos de Binance"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com"
    
    async def get_historical_klines(self, symbol: str, interval: str, start_time: datetime, end_time: datetime) -> List[dict]:
        """Obtener datos históricos por rango de fechas"""
        
        all_klines = []
        current_start = int(start_time.timestamp() * 1000)
        end_timestamp = int(end_time.timestamp() * 1000)
        
        async with aiohttp.ClientSession() as session:
            while current_start < end_timestamp:
                try:
                    url = f"{self.base_url}/api/v3/klines"
                    params = {
                        "symbol": symbol,
                        "interval": interval,
                        "startTime": current_start,
                        "limit": 1000
                    }
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            if not data:
                                break
                            
                            klines_batch = [{
                                'timestamp': int(item[0]),
                                'open': float(item[1]),
                                'high': float(item[2]),
                                'low': float(item[3]),
                                'close': float(item[4]),
                                'volume': float(item[5])
                            } for item in data]
                            
                            all_klines.extend(klines_batch)
                            
                            # Actualizar para siguiente batch
                            current_start = int(data[-1][0]) + 1
                            
                            print(f"  📊 Descargados {len(all_klines)} velas para {symbol}")
                            
                            # Rate limiting
                            await asyncio.sleep(0.1)
                        else:
                            print(f"❌ Error API: {response.status}")
                            break
                            
                except Exception as e:
                    print(f"❌ Error descargando datos: {e}")
                    break
        
        return all_klines

class BacktestEngine:
    """Motor de backtesting"""
    
    def __init__(self):
        self.predictor = OptimizedTCNPredictor()
        self.feature_engine = CompatibleFeatureEngine()
        
        # Configuración de trading
        self.initial_balance = 10000.0  # USDT
        self.commission_rate = 0.001    # 0.1% comisión
        self.position_size = 0.95       # 95% del balance
        
    def run_backtest(self, symbol: str, historical_data: List[dict], 
                    start_date: datetime, end_date: datetime) -> Dict:
        """Ejecutar backtesting completo"""
        
        print(f"\\n🔄 EJECUTANDO BACKTESTING: {symbol}")
        print(f"📅 Período: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
        print(f"📊 Datos: {len(historical_data)} velas")
        
        # Convertir a DataFrame
        df = pd.DataFrame(historical_data)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        df = df.sort_index()
        
        # Crear features
        print("🔧 Creando features técnicos...")
        features = self.feature_engine.create_exact_features(historical_data)
        
        if features.empty or len(features) < 100:
            print("❌ Datos insuficientes para backtesting")
            return {}
        
        # Normalizar features
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features.values)
        
        # Inicializar portfolio
        portfolio = {
            'balance_usdt': self.initial_balance,
            'balance_crypto': 0.0,
            'total_value': self.initial_balance,
            'position': 'USDT',  # USDT, CRYPTO
            'trades': [],
            'daily_values': [],
            'signals': []
        }
        
        # Simular trading día por día
        print("📈 Simulando trading...")
        
        # Empezar desde índice 100 para tener suficiente histórico
        start_idx = 100
        prediction_points = []
        
        for i in range(start_idx, len(features_scaled) - 1):
            current_time = features.index[i]
            current_price = df.loc[current_time, 'close']
            next_price = df.iloc[i + 1]['close']
            
            # Preparar secuencia para predicción (50, 21)
            if i >= 50:
                sequence = features_scaled[i-50:i, :21]
                sequence = np.expand_dims(sequence, axis=0)
                
                try:
                    # Predicción
                    prediction = self.predictor.models[symbol].predict(sequence, verbose=0)
                    probabilities = prediction[0]
                    
                    predicted_class = np.argmax(probabilities)
                    confidence = float(np.max(probabilities))
                    
                    class_names = ['SELL', 'HOLD', 'BUY']
                    signal = class_names[predicted_class]
                    
                    # Guardar señal
                    portfolio['signals'].append({
                        'timestamp': current_time,
                        'signal': signal,
                        'confidence': confidence,
                        'price': current_price,
                        'probabilities': {
                            'SELL': float(probabilities[0]),
                            'HOLD': float(probabilities[1]),
                            'BUY': float(probabilities[2])
                        }
                    })
                    
                    # Ejecutar trade si hay cambio de señal
                    self._execute_trade_signal(portfolio, signal, confidence, current_price, current_time)
                    
                    # Actualizar valor del portfolio
                    if portfolio['position'] == 'USDT':
                        portfolio['total_value'] = portfolio['balance_usdt']
                    else:
                        portfolio['total_value'] = portfolio['balance_crypto'] * current_price
                    
                    # Guardar valor diario
                    if i % 1440 == 0:  # Cada día (1440 minutos)
                        portfolio['daily_values'].append({
                            'date': current_time,
                            'total_value': portfolio['total_value'],
                            'price': current_price
                        })
                    
                    prediction_points.append({
                        'timestamp': current_time,
                        'actual_price': current_price,
                        'next_price': next_price,
                        'signal': signal,
                        'confidence': confidence,
                        'return_1h': (next_price - current_price) / current_price
                    })
                    
                except Exception as e:
                    continue
        
        # Calcular métricas de rendimiento
        metrics = self._calculate_performance_metrics(portfolio, prediction_points, symbol)
        
        return {
            'symbol': symbol,
            'period': f"{start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}",
            'portfolio': portfolio,
            'metrics': metrics,
            'predictions': prediction_points
        }
    
    def _execute_trade_signal(self, portfolio: Dict, signal: str, confidence: float, 
                             price: float, timestamp: datetime):
        """Ejecutar señal de trading"""
        
        min_confidence = 0.6  # Confianza mínima para trading
        
        if confidence < min_confidence:
            return
        
        current_position = portfolio['position']
        
        # BUY: Convertir USDT a crypto
        if signal == 'BUY' and current_position == 'USDT' and portfolio['balance_usdt'] > 100:
            amount_to_spend = portfolio['balance_usdt'] * self.position_size
            commission = amount_to_spend * self.commission_rate
            crypto_amount = (amount_to_spend - commission) / price
            
            portfolio['balance_usdt'] -= amount_to_spend
            portfolio['balance_crypto'] = crypto_amount
            portfolio['position'] = 'CRYPTO'
            
            portfolio['trades'].append({
                'timestamp': timestamp,
                'type': 'BUY',
                'price': price,
                'amount': crypto_amount,
                'usdt_amount': amount_to_spend,
                'commission': commission,
                'confidence': confidence
            })
        
        # SELL: Convertir crypto a USDT
        elif signal == 'SELL' and current_position == 'CRYPTO' and portfolio['balance_crypto'] > 0:
            usdt_received = portfolio['balance_crypto'] * price
            commission = usdt_received * self.commission_rate
            final_usdt = usdt_received - commission
            
            portfolio['balance_usdt'] = final_usdt
            portfolio['balance_crypto'] = 0.0
            portfolio['position'] = 'USDT'
            
            portfolio['trades'].append({
                'timestamp': timestamp,
                'type': 'SELL',
                'price': price,
                'amount': portfolio['balance_crypto'],
                'usdt_amount': final_usdt,
                'commission': commission,
                'confidence': confidence
            })
    
    def _calculate_performance_metrics(self, portfolio: Dict, predictions: List[Dict], symbol: str) -> Dict:
        """Calcular métricas de rendimiento"""
        
        if not portfolio['daily_values'] or not predictions:
            return {}
        
        # Rendimiento del portfolio
        initial_value = self.initial_balance
        final_value = portfolio['total_value']
        total_return = (final_value - initial_value) / initial_value * 100
        
        # Rendimiento del activo (Buy & Hold)
        first_price = predictions[0]['actual_price']
        last_price = predictions[-1]['actual_price']
        buy_hold_return = (last_price - first_price) / first_price * 100
        
        # Número de trades
        trades = portfolio['trades']
        num_trades = len(trades)
        
        # Trades ganadores/perdedores
        winning_trades = 0
        losing_trades = 0
        total_profit = 0
        
        buy_trades = [t for t in trades if t['type'] == 'BUY']
        sell_trades = [t for t in trades if t['type'] == 'SELL']
        
        for i in range(min(len(buy_trades), len(sell_trades))):
            buy_price = buy_trades[i]['price']
            sell_price = sell_trades[i]['price']
            profit_pct = (sell_price - buy_price) / buy_price * 100
            
            if profit_pct > 0:
                winning_trades += 1
            else:
                losing_trades += 1
            
            total_profit += profit_pct
        
        # Win rate
        win_rate = (winning_trades / max(winning_trades + losing_trades, 1)) * 100
        
        # Sharpe ratio (simplificado)
        if portfolio['daily_values']:
            daily_returns = []
            for i in range(1, len(portfolio['daily_values'])):
                prev_val = portfolio['daily_values'][i-1]['total_value']
                curr_val = portfolio['daily_values'][i]['total_value']
                daily_ret = (curr_val - prev_val) / prev_val
                daily_returns.append(daily_ret)
            
            if daily_returns:
                sharpe_ratio = np.mean(daily_returns) / max(np.std(daily_returns), 0.001) * np.sqrt(365)
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Precisión de señales
        correct_predictions = 0
        total_predictions = 0
        
        for pred in predictions:
            actual_return = pred['return_1h']
            signal = pred['signal']
            
            # Definir "correcto" basado en la dirección
            if signal == 'BUY' and actual_return > 0.001:  # >0.1%
                correct_predictions += 1
            elif signal == 'SELL' and actual_return < -0.001:  # <-0.1%
                correct_predictions += 1
            elif signal == 'HOLD' and abs(actual_return) <= 0.001:  # ±0.1%
                correct_predictions += 1
            
            total_predictions += 1
        
        prediction_accuracy = (correct_predictions / max(total_predictions, 1)) * 100
        
        return {
            'total_return_pct': round(total_return, 2),
            'buy_hold_return_pct': round(buy_hold_return, 2),
            'excess_return_pct': round(total_return - buy_hold_return, 2),
            'final_balance': round(final_value, 2),
            'num_trades': num_trades,
            'win_rate_pct': round(win_rate, 2),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'sharpe_ratio': round(sharpe_ratio, 3),
            'prediction_accuracy_pct': round(prediction_accuracy, 2),
            'total_commission_usdt': round(sum(t.get('commission', 0) for t in trades), 2)
        }

class BacktestReporter:
    """Generador de reportes de backtesting"""
    
    def generate_report(self, backtest_results: Dict):
        """Generar reporte completo"""
        
        if not backtest_results:
            print("❌ No hay resultados para reportar")
            return
        
        symbol = backtest_results['symbol']
        metrics = backtest_results['metrics']
        portfolio = backtest_results['portfolio']
        
        print("\\n" + "="*60)
        print(f"📊 REPORTE DE BACKTESTING - {symbol}")
        print("="*60)
        print(f"📅 Período: {backtest_results['period']}")
        print()
        
        # Rendimiento
        print("💰 RENDIMIENTO:")
        print(f"  Retorno Total: {metrics['total_return_pct']:+.2f}%")
        print(f"  Buy & Hold: {metrics['buy_hold_return_pct']:+.2f}%")
        print(f"  Exceso de Retorno: {metrics['excess_return_pct']:+.2f}%")
        print(f"  Balance Final: ${metrics['final_balance']:,.2f}")
        print()
        
        # Trading
        print("📈 ESTADÍSTICAS DE TRADING:")
        print(f"  Número de Trades: {metrics['num_trades']}")
        print(f"  Trades Ganadores: {metrics['winning_trades']}")
        print(f"  Trades Perdedores: {metrics['losing_trades']}")
        print(f"  Win Rate: {metrics['win_rate_pct']:.1f}%")
        print(f"  Comisiones Totales: ${metrics['total_commission_usdt']:.2f}")
        print()
        
        # Métricas de Riesgo
        print("⚡ MÉTRICAS DE RIESGO:")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"  Precisión Predicciones: {metrics['prediction_accuracy_pct']:.1f}%")
        print()
        
        # Evaluación
        self._evaluate_performance(metrics)
        
        # Resumen de trades
        if portfolio['trades']:
            print("📋 ÚLTIMOS 5 TRADES:")
            recent_trades = portfolio['trades'][-5:]
            for trade in recent_trades:
                print(f"  {trade['timestamp'].strftime('%Y-%m-%d %H:%M')} - "
                      f"{trade['type']} @ ${trade['price']:.2f} "
                      f"(Confianza: {trade['confidence']:.1%})")
    
    def _evaluate_performance(self, metrics: Dict):
        """Evaluar rendimiento general"""
        
        score = 0
        evaluation = []
        
        # Factor retorno
        excess_return = metrics['excess_return_pct']
        if excess_return > 10:
            score += 3
            evaluation.append("🟢 Excelente exceso de retorno")
        elif excess_return > 5:
            score += 2
            evaluation.append("🟡 Buen exceso de retorno")
        elif excess_return > 0:
            score += 1
            evaluation.append("🟠 Retorno positivo")
        else:
            evaluation.append("🔴 Retorno negativo")
        
        # Factor win rate
        win_rate = metrics['win_rate_pct']
        if win_rate > 60:
            score += 2
            evaluation.append("🟢 Excelente win rate")
        elif win_rate > 50:
            score += 1
            evaluation.append("🟡 Win rate positivo")
        else:
            evaluation.append("🔴 Win rate bajo")
        
        # Factor Sharpe
        sharpe = metrics['sharpe_ratio']
        if sharpe > 1.5:
            score += 2
            evaluation.append("🟢 Excelente Sharpe ratio")
        elif sharpe > 1.0:
            score += 1
            evaluation.append("🟡 Buen Sharpe ratio")
        else:
            evaluation.append("🔴 Sharpe ratio bajo")
        
        # Factor precisión
        accuracy = metrics['prediction_accuracy_pct']
        if accuracy > 60:
            score += 2
            evaluation.append("🟢 Alta precisión de predicciones")
        elif accuracy > 50:
            score += 1
            evaluation.append("🟡 Precisión aceptable")
        else:
            evaluation.append("🔴 Baja precisión")
        
        # Evaluación final
        if score >= 7:
            final_grade = "🏆 EXCELENTE"
        elif score >= 5:
            final_grade = "🥈 BUENO"
        elif score >= 3:
            final_grade = "🥉 ACEPTABLE"
        else:
            final_grade = "❌ NECESITA MEJORA"
        
        print("🎯 EVALUACIÓN DE RENDIMIENTO:")
        for eval_item in evaluation:
            print(f"  {eval_item}")
        print(f"\\n🏅 CALIFICACIÓN FINAL: {final_grade} (Score: {score}/9)")

class BacktestingManager:
    """Gestor principal de backtesting"""
    
    def __init__(self):
        self.data_provider = HistoricalDataProvider()
        self.engine = BacktestEngine()
        self.reporter = BacktestReporter()
    
    async def run_comprehensive_backtest(self, days_back: int = 30):
        """Ejecutar backtesting completo"""
        
        print("🎯 SISTEMA DE BACKTESTING HISTÓRICO")
        print("Validación del modelo TCN con datos reales")
        print()
        
        # Configurar período
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        
        print(f"📅 Período de prueba: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
        print(f"📊 Pares a analizar: {', '.join(pairs)}")
        print()
        
        all_results = {}
        
        for pair in pairs:
            print(f"\\n🔄 Iniciando backtesting para {pair}...")
            
            try:
                # Descargar datos históricos
                print(f"📡 Descargando datos históricos...")
                historical_data = await self.data_provider.get_historical_klines(
                    pair, "1m", start_date, end_date
                )
                
                if len(historical_data) < 1000:
                    print(f"⚠️  Datos insuficientes para {pair} ({len(historical_data)} velas)")
                    continue
                
                # Ejecutar backtesting
                results = self.engine.run_backtest(pair, historical_data, start_date, end_date)
                
                if results:
                    all_results[pair] = results
                    
                    # Generar reporte individual
                    self.reporter.generate_report(results)
                else:
                    print(f"❌ Error en backtesting de {pair}")
                    
            except Exception as e:
                print(f"❌ Error procesando {pair}: {e}")
        
        # Reporte consolidado
        if all_results:
            self._generate_consolidated_report(all_results)
        else:
            print("❌ No se generaron resultados de backtesting")
    
    def _generate_consolidated_report(self, all_results: Dict):
        """Generar reporte consolidado"""
        
        print("\\n" + "="*60)
        print("📊 REPORTE CONSOLIDADO - TODOS LOS PARES")
        print("="*60)
        
        total_pairs = len(all_results)
        successful_pairs = 0
        total_return = 0
        total_buy_hold = 0
        avg_accuracy = 0
        avg_win_rate = 0
        
        for pair, results in all_results.items():
            metrics = results['metrics']
            
            if metrics['total_return_pct'] > metrics['buy_hold_return_pct']:
                successful_pairs += 1
            
            total_return += metrics['total_return_pct']
            total_buy_hold += metrics['buy_hold_return_pct']
            avg_accuracy += metrics['prediction_accuracy_pct']
            avg_win_rate += metrics['win_rate_pct']
        
        print(f"📊 Pares analizados: {total_pairs}")
        print(f"🎯 Pares exitosos: {successful_pairs}/{total_pairs} ({successful_pairs/total_pairs*100:.1f}%)")
        print(f"💰 Retorno promedio: {total_return/total_pairs:.2f}%")
        print(f"📈 Buy & Hold promedio: {total_buy_hold/total_pairs:.2f}%")
        print(f"⚡ Exceso promedio: {(total_return-total_buy_hold)/total_pairs:.2f}%")
        print(f"🎯 Precisión promedio: {avg_accuracy/total_pairs:.1f}%")
        print(f"🏆 Win rate promedio: {avg_win_rate/total_pairs:.1f}%")
        print()
        
        # Ranking de rendimiento
        print("🏆 RANKING DE RENDIMIENTO:")
        sorted_pairs = sorted(all_results.items(), 
                            key=lambda x: x[1]['metrics']['excess_return_pct'], 
                            reverse=True)
        
        for i, (pair, results) in enumerate(sorted_pairs, 1):
            excess = results['metrics']['excess_return_pct']
            accuracy = results['metrics']['prediction_accuracy_pct']
            print(f"  {i}. {pair}: {excess:+.2f}% exceso, {accuracy:.1f}% precisión")
        
        print("\\n✅ Backtesting completo finalizado")

async def main():
    """Función principal"""
    manager = BacktestingManager()
    await manager.run_comprehensive_backtest(days_back=7)  # 7 días para demo

if __name__ == "__main__":
    asyncio.run(main()) 