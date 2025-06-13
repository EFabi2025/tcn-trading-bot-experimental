#!/usr/bin/env python3
"""
📊 ANALIZADOR DE RETORNOS REALES
Analiza la distribución real de retornos para establecer thresholds apropiados
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

class RealReturnsAnalyzer:
    """📊 Analizador de retornos reales de mercado"""

    def __init__(self):
        self.pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        self.prediction_horizons = [6, 12, 24]  # 6min, 12min, 24min

    async def get_market_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """📊 Obtener datos de mercado"""

        print(f"📊 Obteniendo {days} días de datos para {symbol}...")

        base_url = "https://api.binance.com"
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        async with aiohttp.ClientSession() as session:
            url = f"{base_url}/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': '1m',
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
                        break

                await asyncio.sleep(0.1)

        # Convertir a DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp').sort_index()

        print(f"✅ Obtenidos {len(df)} registros")
        return df

    def analyze_returns_distribution(self, df: pd.DataFrame, symbol: str) -> dict:
        """📊 Analizar distribución de retornos para diferentes horizontes"""

        print(f"\n📊 ANALIZANDO DISTRIBUCIÓN DE RETORNOS: {symbol}")
        print("=" * 60)

        close_prices = df['close'].values
        analysis = {}

        for horizon in self.prediction_horizons:
            print(f"\n🔍 Horizonte de predicción: {horizon} minutos")

            # Calcular retornos futuros
            returns = []
            for i in range(len(close_prices) - horizon):
                current_price = close_prices[i]
                future_price = close_prices[i + horizon]
                return_pct = (future_price - current_price) / current_price
                returns.append(return_pct)

            returns = np.array(returns)

            # Estadísticas básicas
            stats = {
                'mean': np.mean(returns),
                'std': np.std(returns),
                'min': np.min(returns),
                'max': np.max(returns),
                'count': len(returns)
            }

            # Percentiles clave
            percentiles = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
            pct_values = np.percentile(returns, percentiles)

            print(f"📈 Estadísticas básicas:")
            print(f"   Media: {stats['mean']:.4f} ({stats['mean']*100:.2f}%)")
            print(f"   Std: {stats['std']:.4f} ({stats['std']*100:.2f}%)")
            print(f"   Min: {stats['min']:.4f} ({stats['min']*100:.2f}%)")
            print(f"   Max: {stats['max']:.4f} ({stats['max']*100:.2f}%)")
            print(f"   Observaciones: {stats['count']:,}")

            print(f"\n📊 Percentiles de retornos:")
            for pct, value in zip(percentiles, pct_values):
                print(f"   P{pct:2d}: {value:.4f} ({value*100:+.2f}%)")

            # Proponer thresholds balanceados
            # Objetivo: ~30% SELL, ~40% HOLD, ~30% BUY
            sell_threshold_strong = np.percentile(returns, 15)  # 15% más bajo = SELL
            sell_threshold_weak = np.percentile(returns, 30)   # 30% más bajo = SELL débil
            buy_threshold_weak = np.percentile(returns, 70)    # 70% más alto = BUY débil
            buy_threshold_strong = np.percentile(returns, 85)  # 85% más alto = BUY

            proposed_thresholds = {
                'strong_sell': sell_threshold_strong,
                'weak_sell': sell_threshold_weak,
                'weak_buy': buy_threshold_weak,
                'strong_buy': buy_threshold_strong
            }

            print(f"\n🎯 THRESHOLDS PROPUESTOS (distribución balanceada):")
            print(f"   Strong SELL: {proposed_thresholds['strong_sell']:.4f} ({proposed_thresholds['strong_sell']*100:+.2f}%)")
            print(f"   Weak SELL:   {proposed_thresholds['weak_sell']:.4f} ({proposed_thresholds['weak_sell']*100:+.2f}%)")
            print(f"   Weak BUY:    {proposed_thresholds['weak_buy']:.4f} ({proposed_thresholds['weak_buy']*100:+.2f}%)")
            print(f"   Strong BUY:  {proposed_thresholds['strong_buy']:.4f} ({proposed_thresholds['strong_buy']*100:+.2f}%)")

            # Simular distribución con estos thresholds
            simulated_labels = []
            for ret in returns:
                if ret <= proposed_thresholds['strong_sell']:
                    simulated_labels.append('SELL')
                elif ret <= proposed_thresholds['weak_sell']:
                    simulated_labels.append('WEAK_SELL')
                elif ret >= proposed_thresholds['strong_buy']:
                    simulated_labels.append('BUY')
                elif ret >= proposed_thresholds['weak_buy']:
                    simulated_labels.append('WEAK_BUY')
                else:
                    simulated_labels.append('HOLD')

            label_counts = pd.Series(simulated_labels).value_counts()
            total = len(simulated_labels)

            print(f"\n📊 DISTRIBUCIÓN SIMULADA:")
            for label, count in label_counts.items():
                pct = count / total * 100
                print(f"   {label}: {count} ({pct:.1f}%)")

            # Agrupar para distribución final
            sell_total = label_counts.get('SELL', 0) + label_counts.get('WEAK_SELL', 0)
            buy_total = label_counts.get('BUY', 0) + label_counts.get('WEAK_BUY', 0)
            hold_total = label_counts.get('HOLD', 0)

            print(f"\n🎯 DISTRIBUCIÓN FINAL AGRUPADA:")
            print(f"   SELL: {sell_total} ({sell_total/total*100:.1f}%)")
            print(f"   HOLD: {hold_total} ({hold_total/total*100:.1f}%)")
            print(f"   BUY:  {buy_total} ({buy_total/total*100:.1f}%)")

            analysis[f'{horizon}min'] = {
                'stats': stats,
                'percentiles': dict(zip(percentiles, pct_values)),
                'proposed_thresholds': proposed_thresholds,
                'simulated_distribution': {
                    'SELL': sell_total/total,
                    'HOLD': hold_total/total,
                    'BUY': buy_total/total
                }
            }

        return analysis

    def recommend_optimal_thresholds(self, analysis: dict, symbol: str) -> dict:
        """🎯 Recomendar thresholds óptimos basados en análisis"""

        print(f"\n🎯 RECOMENDACIONES PARA {symbol}")
        print("=" * 50)

        # Usar horizonte de 12 minutos como base (balance entre ruido y señal)
        base_analysis = analysis['12min']
        thresholds = base_analysis['proposed_thresholds']

        # Ajustar ligeramente para mejor balance
        # Objetivo: 25-35% SELL, 30-50% HOLD, 25-35% BUY
        distribution = base_analysis['simulated_distribution']

        print(f"📊 Distribución actual con thresholds propuestos:")
        print(f"   SELL: {distribution['SELL']*100:.1f}%")
        print(f"   HOLD: {distribution['HOLD']*100:.1f}%")
        print(f"   BUY:  {distribution['BUY']*100:.1f}%")

        # Ajustar si es necesario
        optimal_thresholds = thresholds.copy()

        if distribution['HOLD'] > 0.6:  # Demasiado HOLD
            print("⚠️ Demasiado HOLD, expandiendo thresholds...")
            optimal_thresholds['weak_sell'] *= 0.8
            optimal_thresholds['weak_buy'] *= 0.8
        elif distribution['HOLD'] < 0.3:  # Muy poco HOLD
            print("⚠️ Muy poco HOLD, contrayendo thresholds...")
            optimal_thresholds['weak_sell'] *= 1.2
            optimal_thresholds['weak_buy'] *= 1.2

        print(f"\n✅ THRESHOLDS ÓPTIMOS RECOMENDADOS:")
        print(f"   Strong SELL: {optimal_thresholds['strong_sell']:.4f} ({optimal_thresholds['strong_sell']*100:+.2f}%)")
        print(f"   Weak SELL:   {optimal_thresholds['weak_sell']:.4f} ({optimal_thresholds['weak_sell']*100:+.2f}%)")
        print(f"   Weak BUY:    {optimal_thresholds['weak_buy']:.4f} ({optimal_thresholds['weak_buy']*100:+.2f}%)")
        print(f"   Strong BUY:  {optimal_thresholds['strong_buy']:.4f} ({optimal_thresholds['strong_buy']*100:+.2f}%)")

        return optimal_thresholds

    async def analyze_all_pairs(self) -> dict:
        """📊 Analizar todos los pares y generar recomendaciones"""

        print("📊 ANÁLISIS COMPLETO DE RETORNOS REALES")
        print("=" * 80)
        print("🎯 Objetivo: Establecer thresholds para distribución balanceada")
        print("🎯 Meta: ~30% SELL, ~40% HOLD, ~30% BUY")
        print("=" * 80)

        all_recommendations = {}

        for symbol in self.pairs:
            try:
                # Obtener datos
                df = await self.get_market_data(symbol, days=30)

                # Analizar distribución
                analysis = self.analyze_returns_distribution(df, symbol)

                # Generar recomendaciones
                optimal_thresholds = self.recommend_optimal_thresholds(analysis, symbol)

                all_recommendations[symbol] = optimal_thresholds

            except Exception as e:
                print(f"❌ Error analizando {symbol}: {e}")

        # Resumen final
        print(f"\n🎯 RESUMEN DE THRESHOLDS RECOMENDADOS")
        print("=" * 80)

        for symbol, thresholds in all_recommendations.items():
            print(f"\n{symbol}:")
            print(f"  'strong_sell': {thresholds['strong_sell']:.4f},  # {thresholds['strong_sell']*100:+.2f}%")
            print(f"  'weak_sell':   {thresholds['weak_sell']:.4f},   # {thresholds['weak_sell']*100:+.2f}%")
            print(f"  'weak_buy':    {thresholds['weak_buy']:.4f},    # {thresholds['weak_buy']*100:+.2f}%")
            print(f"  'strong_buy':  {thresholds['strong_buy']:.4f}   # {thresholds['strong_buy']*100:+.2f}%")

        return all_recommendations

async def main():
    """📊 Ejecutar análisis completo"""
    analyzer = RealReturnsAnalyzer()
    recommendations = await analyzer.analyze_all_pairs()

    print(f"\n✅ Análisis completado. Usa estos thresholds en el entrenador definitivo.")

if __name__ == "__main__":
    asyncio.run(main())
