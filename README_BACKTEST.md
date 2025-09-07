# 🎯 Sistema de Backtest para Predictores Técnicos

Sistema completo para evaluar y optimizar predictores técnicos de 1m, 3m y 5m con configuraciones flexibles.

## 📋 Características Principales

### 🔧 Funcionalidades Core
- **Soporte Multi-Timeframe**: 1m, 3m, 5m
- **Predictores Integrados**: predictor1m_talib, predictor3m_core_optimized, predictor5m_talib
- **Modo Ensemble**: Combinación inteligente de múltiples predictores
- **Datos Históricos**: Integración con Binance API + datos sintéticos para testing
- **Métricas Avanzadas**: Sharpe Ratio, Sortino Ratio, Calmar Ratio, VaR, CVaR
- **Visualizaciones**: Gráficos de equity, drawdown, distribución de trades
- **Reportes**: Generación automática de reportes en Markdown

### ⚙️ Configuración Flexible
- **Estrategias Predefinidas**: Conservative, Moderate, Aggressive, Scalping, Swing, Momentum, Mean Reversion
- **Optimización de Parámetros**: Grid Search, Random Search, Bayesian Optimization
- **Validación de Configuraciones**: Validación automática de parámetros
- **Templates**: Configuraciones predefinidas para diferentes escenarios

### 📊 Análisis Avanzado
- **Comparación de Timeframes**: Evaluación de rendimiento por timeframe
- **Comparación de Estrategias**: Análisis de diferentes enfoques de trading
- **Análisis de Riesgo**: Evaluación detallada de métricas de riesgo
- **Análisis Multi-Símbolo**: Evaluación simultánea de múltiples pares

## 🚀 Instalación

### Requisitos
```bash
pip install pandas numpy matplotlib seaborn scipy
pip install binance-python python-dotenv
pip install talib pandas-ta
```

### Archivos del Sistema
```
backtest_system.py          # Motor principal de backtesting
backtest_config_manager.py  # Gestor de configuraciones
backtest_examples.py        # Ejemplos de uso
README_BACKTEST.md          # Esta documentación
```

## 📖 Uso Básico

### 1. Backtest Simple
```python
from backtest_system import BacktestConfig, run_single_backtest
from backtest_config_manager import create_quick_config

# Crear configuración rápida
config = create_quick_config(
    symbol='BTCUSDT',
    timeframe='1m',
    strategy='moderate',
    days=30
)

# Ejecutar backtest
results = await run_single_backtest(config)

# Mostrar resultados
print(f"Total Trades: {results.total_trades}")
print(f"Win Rate: {results.win_rate:.2f}%")
print(f"Retorno Total: {results.total_return:.2f}%")
```

### 2. Comparación de Timeframes
```python
from backtest_examples import BacktestExamples

examples = BacktestExamples()
results, comparison = await examples.example_2_timeframe_comparison()
```

### 3. Optimización de Parámetros
```python
from backtest_config_manager import BacktestConfigManager

manager = BacktestConfigManager()

# Crear configuración base
base_config = create_quick_config('BTCUSDT', '1m', 'moderate', 30)

# Crear variaciones de parámetros
param_variations = {
    'position_size_pct': [0.05, 0.1, 0.15, 0.2],
    'stop_loss_pct': [0.01, 0.02, 0.03],
    'take_profit_pct': [0.02, 0.04, 0.06]
}

configs = manager.generate_config_variations(base_config, param_variations)
```

## ⚙️ Configuración Avanzada

### Estrategias Predefinidas

#### Conservative (Conservadora)
- **Position Size**: 5%
- **Stop Loss**: 1.5%
- **Take Profit**: 3%
- **Max Positions**: 2
- **Min Confidence**: 70%
- **Risk Level**: Solo LOW

#### Moderate (Moderada)
- **Position Size**: 10%
- **Stop Loss**: 2%
- **Take Profit**: 4%
- **Max Positions**: 3
- **Min Confidence**: 60%
- **Risk Level**: LOW + MEDIUM

#### Aggressive (Agresiva)
- **Position Size**: 20%
- **Stop Loss**: 3%
- **Take Profit**: 6%
- **Max Positions**: 5
- **Min Confidence**: 50%
- **Risk Level**: LOW + MEDIUM + HIGH

#### Scalping
- **Position Size**: 15%
- **Stop Loss**: 0.5%
- **Take Profit**: 1%
- **Max Positions**: 10
- **Min Confidence**: 55%
- **Enfoque**: Predictor 1m (70% peso)

### Configuración Personalizada
```python
from backtest_system import BacktestConfig

config = BacktestConfig(
    symbol='BTCUSDT',
    timeframe='1m',
    start_date='2024-01-01',
    end_date='2024-12-31',
    initial_balance=10000.0,
    
    # Predictores
    predictors=['1m', '3m', '5m'],
    use_ensemble=True,
    ensemble_weights={'1m': 0.4, '3m': 0.4, '5m': 0.2},
    
    # Trading
    position_size_pct=0.1,
    stop_loss_pct=0.02,
    take_profit_pct=0.04,
    max_positions=3,
    
    # Filtros
    min_confidence=60.0,
    min_signal_strength=0.6,
    risk_level_filter=[RiskLevel.LOW, RiskLevel.MEDIUM],
    
    # Comisiones
    commission_rate=0.001,
    slippage_rate=0.0005
)
```

## 📊 Métricas de Evaluación

### Métricas Básicas
- **Total Trades**: Número total de operaciones
- **Win Rate**: Porcentaje de trades ganadores
- **Total Return**: Retorno total del período
- **Annualized Return**: Retorno anualizado

### Métricas de Riesgo
- **Max Drawdown**: Máxima pérdida desde un pico
- **Volatility**: Volatilidad anualizada
- **VaR 95%**: Value at Risk al 95%
- **CVaR 95%**: Conditional Value at Risk al 95%

### Métricas Avanzadas
- **Sharpe Ratio**: Retorno ajustado por riesgo
- **Sortino Ratio**: Similar a Sharpe pero solo considera volatilidad negativa
- **Calmar Ratio**: Retorno anualizado / Max Drawdown
- **Profit Factor**: Ganancia bruta / Pérdida bruta
- **Kelly Criterion**: Tamaño óptimo de posición

## 🎨 Visualizaciones

### Gráficos Disponibles
1. **Curva de Equity**: Evolución del capital a lo largo del tiempo
2. **Drawdown**: Pérdidas máximas en cada momento
3. **Distribución de Trades**: Histograma de PnL por trade
4. **PnL Acumulado**: Evolución del PnL acumulado
5. **Análisis Mensual**: Rendimiento por mes
6. **Comparaciones**: Gráficos comparativos entre configuraciones

### Ejemplo de Visualización
```python
from backtest_system import BacktestVisualizer

visualizer = BacktestVisualizer(results)

# Plotear curva de equity
visualizer.plot_equity_curve('equity_curve.png')

# Plotear distribución de trades
visualizer.plot_trade_distribution('trade_distribution.png')

# Generar reporte
report = visualizer.generate_report('backtest_report.md')
```

## 🔧 Ejemplos de Uso

### Ejemplo 1: Backtest Básico
```python
import asyncio
from backtest_examples import BacktestExamples

async def main():
    examples = BacktestExamples()
    await examples.example_1_basic_backtest()

asyncio.run(main())
```

### Ejemplo 2: Comparación de Estrategias
```python
async def main():
    examples = BacktestExamples()
    await examples.example_3_strategy_comparison()

asyncio.run(main())
```

### Ejemplo 3: Optimización de Parámetros
```python
async def main():
    examples = BacktestExamples()
    await examples.example_4_parameter_optimization()

asyncio.run(main())
```

## 📁 Estructura de Archivos de Resultados

```
backtest_results/
├── example1_equity_curve.png
├── example1_trade_distribution.png
├── example1_report.md
├── example2_timeframe_comparison.csv
├── example2_timeframe_comparison.png
├── example3_strategy_comparison.csv
├── example3_strategy_comparison.png
├── example4_optimization_results.csv
├── example4_optimization.png
├── example5_risk_analysis.png
└── example6_multi_symbol.png
```

## 🚨 Consideraciones Importantes

### Limitaciones
1. **Datos Sintéticos**: Si no hay API keys de Binance, se usan datos sintéticos
2. **Slippage**: El slippage se simula de forma simplificada
3. **Liquidez**: No se considera el impacto en el mercado
4. **Costos**: Solo se incluyen comisiones básicas

### Mejores Prácticas
1. **Validación**: Siempre validar configuraciones antes de ejecutar
2. **Datos**: Usar datos históricos reales cuando sea posible
3. **Períodos**: Probar con diferentes períodos de tiempo
4. **Robustez**: Evaluar rendimiento en diferentes condiciones de mercado

## 🔍 Troubleshooting

### Problemas Comunes

#### Error: "Predictores no disponibles"
```python
# Verificar que los predictores estén en el directorio
import os
print(os.listdir('.'))
```

#### Error: "No se pudieron obtener datos históricos"
```python
# Verificar API keys de Binance
import os
print(os.getenv('BINANCE_API_KEY'))
```

#### Error: "Configuración inválida"
```python
# Validar configuración
from backtest_config_manager import BacktestConfigManager

manager = BacktestConfigManager()
is_valid, errors = manager.validate_config(config)
print(errors)
```

## 📈 Próximas Mejoras

### Funcionalidades Planificadas
1. **Machine Learning**: Integración con modelos ML para optimización
2. **Walk-Forward Analysis**: Análisis de validación walk-forward
3. **Monte Carlo Simulation**: Simulaciones Monte Carlo para robustez
4. **Portfolio Optimization**: Optimización de carteras multi-símbolo
5. **Real-time Integration**: Integración con trading en tiempo real

### Contribuciones
Las contribuciones son bienvenidas. Por favor:
1. Fork el repositorio
2. Crea una rama para tu feature
3. Haz commit de tus cambios
4. Abre un Pull Request

## 📞 Soporte

Para soporte o preguntas:
1. Revisa la documentación
2. Ejecuta los ejemplos incluidos
3. Verifica los logs de error
4. Abre un issue en el repositorio

---

**¡Disfruta evaluando tus predictores técnicos! 🚀**
