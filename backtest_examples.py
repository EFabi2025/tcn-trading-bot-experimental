#!/usr/bin/env python3
"""
🎯 EJEMPLOS DE USO DEL SISTEMA DE BACKTEST
Ejemplos prácticos para evaluar predictores técnicos de 1m, 3m y 5m

EJEMPLOS INCLUIDOS:
1. Backtest básico con un predictor
2. Comparación de múltiples timeframes
3. Comparación de estrategias
4. Optimización de parámetros
5. Análisis de riesgo
6. Reportes avanzados
"""

import asyncio
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# === IMPORTS DEL SISTEMA ===
from backtest_system import (
    BacktestConfig, BacktestResults, TechnicalPredictorBacktest,
    BacktestVisualizer, run_single_backtest, run_multiple_backtests,
    compare_backtests
)
from backtest_config_manager import (
    BacktestConfigManager, StrategyType, create_quick_config,
    create_comparison_configs
)

# === CONFIGURACIÓN DE VISUALIZACIÓN ===
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BacktestExamples:
    """Clase con ejemplos de uso del sistema de backtest"""
    
    def __init__(self):
        self.config_manager = BacktestConfigManager()
        self.results_dir = "backtest_results"
        self.ensure_results_dir()
    
    def ensure_results_dir(self):
        """Crear directorio de resultados si no existe"""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            print(f"📁 Directorio de resultados creado: {self.results_dir}")
    
    async def example_1_basic_backtest(self):
        """Ejemplo 1: Backtest básico con un predictor"""
        print("\n" + "="*60)
        print("🎯 EJEMPLO 1: BACKTEST BÁSICO")
        print("="*60)
        
        # Crear configuración básica
        config = create_quick_config(
            symbol='BTCUSDT',
            timeframe='1m',
            strategy='moderate',
            days=7  # 7 días para testing rápido
        )
        
        print(f"📊 Configuración: {config.symbol} {config.timeframe}")
        print(f"📅 Período: {config.start_date} - {config.end_date}")
        print(f"💰 Balance inicial: ${config.initial_balance:,.2f}")
        
        # Ejecutar backtest
        results = await run_single_backtest(config)
        
        # Mostrar resultados básicos
        print(f"\n📈 RESULTADOS:")
        print(f"  Total Trades: {results.total_trades}")
        print(f"  Win Rate: {results.win_rate:.2f}%")
        print(f"  Retorno Total: {results.total_return:.2f}%")
        print(f"  Max Drawdown: {results.max_drawdown:.2f}%")
        print(f"  Sharpe Ratio: {results.sharpe_ratio:.2f}")
        
        # Visualizar
        visualizer = BacktestVisualizer(results)
        visualizer.plot_equity_curve(f"{self.results_dir}/example1_equity_curve.png")
        visualizer.plot_trade_distribution(f"{self.results_dir}/example1_trade_distribution.png")
        
        # Generar reporte
        report = visualizer.generate_report(f"{self.results_dir}/example1_report.md")
        
        return results
    
    async def example_2_timeframe_comparison(self):
        """Ejemplo 2: Comparación de timeframes"""
        print("\n" + "="*60)
        print("🎯 EJEMPLO 2: COMPARACIÓN DE TIMEFRAMES")
        print("="*60)
        
        # Crear configuraciones para diferentes timeframes
        timeframes = ['1m', '3m', '5m']
        configs = []
        
        for tf in timeframes:
            config = create_quick_config(
                symbol='BTCUSDT',
                timeframe=tf,
                strategy='moderate',
                days=14
            )
            configs.append(config)
        
        print(f"📊 Comparando timeframes: {timeframes}")
        
        # Ejecutar backtests
        results_dict = await run_multiple_backtests(configs)
        
        # Comparar resultados
        comparison_df = compare_backtests(results_dict)
        print("\n📈 COMPARACIÓN DE RESULTADOS:")
        print(comparison_df.to_string(index=False))
        
        # Guardar comparación
        comparison_df.to_csv(f"{self.results_dir}/example2_timeframe_comparison.csv", index=False)
        
        # Visualizar comparación
        self._plot_comparison_chart(comparison_df, "Timeframe", f"{self.results_dir}/example2_timeframe_comparison.png")
        
        return results_dict, comparison_df
    
    async def example_3_strategy_comparison(self):
        """Ejemplo 3: Comparación de estrategias"""
        print("\n" + "="*60)
        print("🎯 EJEMPLO 3: COMPARACIÓN DE ESTRATEGIAS")
        print("="*60)
        
        # Crear configuraciones para diferentes estrategias
        strategies = ['conservative', 'moderate', 'aggressive', 'scalping']
        configs = []
        
        for strategy in strategies:
            config = create_quick_config(
                symbol='BTCUSDT',
                timeframe='1m',
                strategy=strategy,
                days=21
            )
            configs.append(config)
        
        print(f"📊 Comparando estrategias: {strategies}")
        
        # Ejecutar backtests
        results_dict = await run_multiple_backtests(configs)
        
        # Comparar resultados
        comparison_df = compare_backtests(results_dict)
        print("\n📈 COMPARACIÓN DE ESTRATEGIAS:")
        print(comparison_df.to_string(index=False))
        
        # Guardar comparación
        comparison_df.to_csv(f"{self.results_dir}/example3_strategy_comparison.csv", index=False)
        
        # Visualizar comparación
        self._plot_comparison_chart(comparison_df, "Strategy", f"{self.results_dir}/example3_strategy_comparison.png")
        
        return results_dict, comparison_df
    
    async def example_4_parameter_optimization(self):
        """Ejemplo 4: Optimización de parámetros"""
        print("\n" + "="*60)
        print("🎯 EJEMPLO 4: OPTIMIZACIÓN DE PARÁMETROS")
        print("="*60)
        
        # Configuración base
        base_config = create_quick_config(
            symbol='BTCUSDT',
            timeframe='1m',
            strategy='moderate',
            days=14
        )
        
        # Crear variaciones de parámetros
        param_variations = {
            'position_size_pct': [0.05, 0.1, 0.15, 0.2],
            'stop_loss_pct': [0.01, 0.02, 0.03],
            'take_profit_pct': [0.02, 0.04, 0.06]
        }
        
        configs = self.config_manager.generate_config_variations(base_config, param_variations)
        
        print(f"📊 Optimizando {len(configs)} configuraciones...")
        
        # Ejecutar backtests (limitado para demo)
        limited_configs = configs[:20]  # Limitar para demo
        results_dict = await run_multiple_backtests(limited_configs)
        
        # Encontrar mejor configuración
        best_config = None
        best_sharpe = -float('inf')
        
        for name, result in results_dict.items():
            if result.sharpe_ratio > best_sharpe:
                best_sharpe = result.sharpe_ratio
                best_config = name
        
        print(f"\n🏆 MEJOR CONFIGURACIÓN:")
        print(f"  Configuración: {best_config}")
        print(f"  Sharpe Ratio: {best_sharpe:.2f}")
        
        # Guardar resultados de optimización
        optimization_results = []
        for name, result in results_dict.items():
            optimization_results.append({
                'config': name,
                'sharpe_ratio': result.sharpe_ratio,
                'total_return': result.total_return,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate
            })
        
        opt_df = pd.DataFrame(optimization_results)
        opt_df = opt_df.sort_values('sharpe_ratio', ascending=False)
        opt_df.to_csv(f"{self.results_dir}/example4_optimization_results.csv", index=False)
        
        # Visualizar optimización
        self._plot_optimization_results(opt_df, f"{self.results_dir}/example4_optimization.png")
        
        return results_dict, opt_df
    
    async def example_5_risk_analysis(self):
        """Ejemplo 5: Análisis de riesgo detallado"""
        print("\n" + "="*60)
        print("🎯 EJEMPLO 5: ANÁLISIS DE RIESGO")
        print("="*60)
        
        # Crear configuraciones con diferentes niveles de riesgo
        risk_configs = {
            'low_risk': create_quick_config('BTCUSDT', '1m', 'conservative', 30),
            'medium_risk': create_quick_config('BTCUSDT', '1m', 'moderate', 30),
            'high_risk': create_quick_config('BTCUSDT', '1m', 'aggressive', 30)
        }
        
        # Modificar configuraciones para análisis de riesgo
        for name, config in risk_configs.items():
            if name == 'low_risk':
                config.position_size_pct = 0.05
                config.stop_loss_pct = 0.01
            elif name == 'high_risk':
                config.position_size_pct = 0.25
                config.stop_loss_pct = 0.04
        
        print("📊 Analizando diferentes niveles de riesgo...")
        
        # Ejecutar backtests
        results_dict = await run_multiple_backtests(list(risk_configs.values()))
        
        # Análisis de riesgo
        risk_analysis = self._analyze_risk(results_dict)
        
        print("\n📈 ANÁLISIS DE RIESGO:")
        for name, analysis in risk_analysis.items():
            print(f"\n{name.upper()}:")
            print(f"  Sharpe Ratio: {analysis['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {analysis['max_drawdown']:.2f}%")
            print(f"  Volatilidad: {analysis['volatility']:.2f}%")
            print(f"  VaR 95%: {analysis['var_95']:.2f}%")
        
        # Visualizar análisis de riesgo
        self._plot_risk_analysis(risk_analysis, f"{self.results_dir}/example5_risk_analysis.png")
        
        return results_dict, risk_analysis
    
    async def example_6_multi_symbol_analysis(self):
        """Ejemplo 6: Análisis multi-símbolo"""
        print("\n" + "="*60)
        print("🎯 EJEMPLO 6: ANÁLISIS MULTI-SÍMBOLO")
        print("="*60)
        
        # Símbolos para análisis
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT']
        
        # Crear configuraciones para cada símbolo
        configs = []
        for symbol in symbols:
            config = create_quick_config(
                symbol=symbol,
                timeframe='1m',
                strategy='moderate',
                days=14
            )
            configs.append(config)
        
        print(f"📊 Analizando símbolos: {symbols}")
        
        # Ejecutar backtests
        results_dict = await run_multiple_backtests(configs)
        
        # Análisis multi-símbolo
        multi_analysis = self._analyze_multi_symbol(results_dict)
        
        print("\n📈 ANÁLISIS MULTI-SÍMBOLO:")
        for symbol, analysis in multi_analysis.items():
            print(f"\n{symbol}:")
            print(f"  Retorno: {analysis['total_return']:.2f}%")
            print(f"  Sharpe: {analysis['sharpe_ratio']:.2f}")
            print(f"  Drawdown: {analysis['max_drawdown']:.2f}%")
            print(f"  Trades: {analysis['total_trades']}")
        
        # Visualizar análisis multi-símbolo
        self._plot_multi_symbol_analysis(multi_analysis, f"{self.results_dir}/example6_multi_symbol.png")
        
        return results_dict, multi_analysis
    
    def _plot_comparison_chart(self, df: pd.DataFrame, group_col: str, save_path: str):
        """Plotear gráfico de comparación"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Win Rate
        axes[0, 0].bar(df['Name'], df['Win Rate (%)'])
        axes[0, 0].set_title('Win Rate por Configuración')
        axes[0, 0].set_ylabel('Win Rate (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Total Return
        axes[0, 1].bar(df['Name'], df['Total Return (%)'])
        axes[0, 1].set_title('Retorno Total por Configuración')
        axes[0, 1].set_ylabel('Retorno Total (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Max Drawdown
        axes[1, 0].bar(df['Name'], df['Max Drawdown (%)'])
        axes[1, 0].set_title('Max Drawdown por Configuración')
        axes[1, 0].set_ylabel('Max Drawdown (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Sharpe Ratio
        axes[1, 1].bar(df['Name'], df['Sharpe Ratio'])
        axes[1, 1].set_title('Sharpe Ratio por Configuración')
        axes[1, 1].set_ylabel('Sharpe Ratio')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Gráfico guardado en: {save_path}")
    
    def _plot_optimization_results(self, df: pd.DataFrame, save_path: str):
        """Plotear resultados de optimización"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Top 10 configuraciones por Sharpe Ratio
        top_10 = df.head(10)
        
        axes[0, 0].barh(range(len(top_10)), top_10['sharpe_ratio'])
        axes[0, 0].set_yticks(range(len(top_10)))
        axes[0, 0].set_yticklabels([f"Config {i+1}" for i in range(len(top_10))])
        axes[0, 0].set_title('Top 10 Configuraciones - Sharpe Ratio')
        axes[0, 0].set_xlabel('Sharpe Ratio')
        
        # Retorno vs Drawdown
        axes[0, 1].scatter(df['max_drawdown'], df['total_return'], alpha=0.6)
        axes[0, 1].set_xlabel('Max Drawdown (%)')
        axes[0, 1].set_ylabel('Total Return (%)')
        axes[0, 1].set_title('Retorno vs Drawdown')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Win Rate vs Sharpe Ratio
        axes[1, 0].scatter(df['win_rate'], df['sharpe_ratio'], alpha=0.6)
        axes[1, 0].set_xlabel('Win Rate (%)')
        axes[1, 0].set_ylabel('Sharpe Ratio')
        axes[1, 0].set_title('Win Rate vs Sharpe Ratio')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Distribución de Sharpe Ratios
        axes[1, 1].hist(df['sharpe_ratio'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Sharpe Ratio')
        axes[1, 1].set_ylabel('Frecuencia')
        axes[1, 1].set_title('Distribución de Sharpe Ratios')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Gráfico de optimización guardado en: {save_path}")
    
    def _analyze_risk(self, results_dict: Dict[str, BacktestResults]) -> Dict[str, Dict]:
        """Analizar métricas de riesgo"""
        risk_analysis = {}
        
        for name, result in results_dict.items():
            risk_analysis[name] = {
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'volatility': result.volatility,
                'var_95': result.var_95,
                'cvar_95': result.cvar_95,
                'total_return': result.total_return,
                'win_rate': result.win_rate
            }
        
        return risk_analysis
    
    def _plot_risk_analysis(self, risk_analysis: Dict[str, Dict], save_path: str):
        """Plotear análisis de riesgo"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        names = list(risk_analysis.keys())
        sharpe_ratios = [risk_analysis[name]['sharpe_ratio'] for name in names]
        max_drawdowns = [risk_analysis[name]['max_drawdown'] for name in names]
        volatilities = [risk_analysis[name]['volatility'] for name in names]
        total_returns = [risk_analysis[name]['total_return'] for name in names]
        
        # Sharpe Ratio
        axes[0, 0].bar(names, sharpe_ratios, color=['green', 'orange', 'red'])
        axes[0, 0].set_title('Sharpe Ratio por Nivel de Riesgo')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        
        # Max Drawdown
        axes[0, 1].bar(names, max_drawdowns, color=['green', 'orange', 'red'])
        axes[0, 1].set_title('Max Drawdown por Nivel de Riesgo')
        axes[0, 1].set_ylabel('Max Drawdown (%)')
        
        # Volatilidad
        axes[1, 0].bar(names, volatilities, color=['green', 'orange', 'red'])
        axes[1, 0].set_title('Volatilidad por Nivel de Riesgo')
        axes[1, 0].set_ylabel('Volatilidad (%)')
        
        # Retorno vs Riesgo
        axes[1, 1].scatter(max_drawdowns, total_returns, s=100, c=['green', 'orange', 'red'])
        for i, name in enumerate(names):
            axes[1, 1].annotate(name, (max_drawdowns[i], total_returns[i]), 
                              xytext=(5, 5), textcoords='offset points')
        axes[1, 1].set_xlabel('Max Drawdown (%)')
        axes[1, 1].set_ylabel('Total Return (%)')
        axes[1, 1].set_title('Retorno vs Riesgo')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Análisis de riesgo guardado en: {save_path}")
    
    def _analyze_multi_symbol(self, results_dict: Dict[str, BacktestResults]) -> Dict[str, Dict]:
        """Analizar resultados multi-símbolo"""
        multi_analysis = {}
        
        for name, result in results_dict.items():
            multi_analysis[name] = {
                'total_return': result.total_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
                'volatility': result.volatility
            }
        
        return multi_analysis
    
    def _plot_multi_symbol_analysis(self, multi_analysis: Dict[str, Dict], save_path: str):
        """Plotear análisis multi-símbolo"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        symbols = list(multi_analysis.keys())
        returns = [multi_analysis[symbol]['total_return'] for symbol in symbols]
        sharpe_ratios = [multi_analysis[symbol]['sharpe_ratio'] for symbol in symbols]
        drawdowns = [multi_analysis[symbol]['max_drawdown'] for symbol in symbols]
        trades = [multi_analysis[symbol]['total_trades'] for symbol in symbols]
        
        # Retorno por símbolo
        axes[0, 0].bar(symbols, returns, color='skyblue')
        axes[0, 0].set_title('Retorno Total por Símbolo')
        axes[0, 0].set_ylabel('Retorno Total (%)')
        
        # Sharpe Ratio por símbolo
        axes[0, 1].bar(symbols, sharpe_ratios, color='lightgreen')
        axes[0, 1].set_title('Sharpe Ratio por Símbolo')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        
        # Drawdown por símbolo
        axes[1, 0].bar(symbols, drawdowns, color='lightcoral')
        axes[1, 0].set_title('Max Drawdown por Símbolo')
        axes[1, 0].set_ylabel('Max Drawdown (%)')
        
        # Número de trades por símbolo
        axes[1, 1].bar(symbols, trades, color='gold')
        axes[1, 1].set_title('Número de Trades por Símbolo')
        axes[1, 1].set_ylabel('Total Trades')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Análisis multi-símbolo guardado en: {save_path}")

async def run_all_examples():
    """Ejecutar todos los ejemplos"""
    examples = BacktestExamples()
    
    print("🚀 INICIANDO EJEMPLOS DE BACKTEST")
    print("="*60)
    
    try:
        # Ejemplo 1: Backtest básico
        await examples.example_1_basic_backtest()
        
        # Ejemplo 2: Comparación de timeframes
        await examples.example_2_timeframe_comparison()
        
        # Ejemplo 3: Comparación de estrategias
        await examples.example_3_strategy_comparison()
        
        # Ejemplo 4: Optimización de parámetros
        await examples.example_4_parameter_optimization()
        
        # Ejemplo 5: Análisis de riesgo
        await examples.example_5_risk_analysis()
        
        # Ejemplo 6: Análisis multi-símbolo
        await examples.example_6_multi_symbol_analysis()
        
        print("\n✅ TODOS LOS EJEMPLOS COMPLETADOS")
        print(f"📁 Resultados guardados en: {examples.results_dir}")
        
    except Exception as e:
        print(f"❌ Error ejecutando ejemplos: {e}")
        import traceback
        traceback.print_exc()

async def run_quick_test():
    """Ejecutar test rápido"""
    print("🚀 TEST RÁPIDO DEL SISTEMA DE BACKTEST")
    print("="*50)
    
    # Crear configuración rápida
    config = create_quick_config(
        symbol='BTCUSDT',
        timeframe='1m',
        strategy='moderate',
        days=3  # Solo 3 días para test rápido
    )
    
    print(f"📊 Configuración: {config.symbol} {config.timeframe}")
    print(f"📅 Período: {config.start_date} - {config.end_date}")
    
    # Ejecutar backtest
    results = await run_single_backtest(config)
    
    # Mostrar resultados
    print(f"\n📈 RESULTADOS DEL TEST:")
    print(f"  Total Trades: {results.total_trades}")
    print(f"  Win Rate: {results.win_rate:.2f}%")
    print(f"  Retorno Total: {results.total_return:.2f}%")
    print(f"  Max Drawdown: {results.max_drawdown:.2f}%")
    print(f"  Sharpe Ratio: {results.sharpe_ratio:.2f}")
    
    print("\n✅ Test completado exitosamente!")

if __name__ == "__main__":
    # Ejecutar test rápido por defecto
    asyncio.run(run_quick_test())
    
    # Descomentar para ejecutar todos los ejemplos
    # asyncio.run(run_all_examples())
