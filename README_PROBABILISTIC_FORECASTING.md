# 🎯 Sistema de Predicción Probabilística con Distribución Chi-Cuadrado

## 📋 Descripción General

El **Sistema de Predicción Probabilística** utiliza los predictores de 1m, 3m y 5m para generar predicciones con horizonte temporal utilizando una distribución chi-cuadrado. Este sistema proporciona:

- **Predicciones probabilísticas** con intervalos de confianza
- **Horizonte temporal** de minutos en el futuro
- **Ajuste por contexto de mercado** (volatilidad, tendencia, liquidez)
- **Distribución chi-cuadrado** para modelar incertidumbre temporal
- **Integración completa** con los predictores existentes

## 🚀 Características Principales

### 1. **Predicciones Multi-Timeframe**
- Combina predicciones de 1m, 3m y 5m
- Pesos balanceados: 1m (40%), 3m (35%), 5m (25%)
- Integración con predictores técnicos existentes

### 2. **Distribución Chi-Cuadrado**
- Modela incertidumbre temporal de manera realista
- Grados de libertad ajustables según horizonte
- Percentil 75 para nivel de confianza razonable

### 3. **Horizonte Temporal**
- Predicciones para: 1, 3, 5, 10, 15, 30, 60 minutos
- Incertidumbre creciente con el tiempo
- Horizonte recomendado basado en confianza

### 4. **Contexto de Mercado**
- Detección automática de volatilidad
- Ajuste de predicciones según contexto
- Pesos adaptativos por régimen de mercado

## 📊 Estructura del Sistema

```
ProbabilisticForecastingSystem
├── get_timeframe_predictions()     # Obtener predicciones de timeframes
├── calculate_ensemble_probabilities() # Calcular probabilidades del ensemble
├── calculate_chi_square_uncertainty() # Calcular incertidumbre temporal
├── generate_temporal_predictions() # Generar predicciones temporales
├── calculate_confidence_interval() # Calcular intervalos de confianza
├── get_market_context()           # Obtener contexto de mercado
└── generate_comprehensive_forecast() # Generar predicción completa
```

## 🎯 Uso Básico

### Ejemplo 1: Predicción Simple

```python
from probabilistic_forecasting_system import generate_probabilistic_forecast

# Generar predicción para BTCUSDT
forecast = generate_probabilistic_forecast('BTCUSDT')

if forecast:
    print(f"Señal: {forecast['primary_signal']}")
    print(f"Confianza: {forecast['confidence']*100:.1f}%")
    print(f"Horizonte recomendado: {forecast['recommended_horizon']} minutos")
```

### Ejemplo 2: Uso Avanzado

```python
from probabilistic_forecasting_system import ProbabilisticForecastingSystem

# Inicializar sistema
system = ProbabilisticForecastingSystem()

# Generar predicción completa
forecast = system.generate_comprehensive_forecast('BTCUSDT')

# Acceder a predicciones temporales
for horizon, pred in forecast['temporal_predictions'].items():
    print(f"{horizon}m: {pred['primary_signal']} (conf: {pred['confidence']*100:.1f}%)")
```

## 📈 Interpretación de Resultados

### Estructura de la Predicción

```python
{
    'symbol': 'BTCUSDT',
    'timestamp': datetime.now(),
    'market_context': 'low_volatility',
    'ensemble_probabilities': {
        'BUY': 0.345,
        'HOLD': 0.160,
        'SELL': 0.495
    },
    'temporal_predictions': {
        1: {
            'primary_signal': 'SELL',
            'confidence': 0.495,
            'probabilities': {'BUY': 0.345, 'HOLD': 0.160, 'SELL': 0.495},
            'uncertainty': 0.509,
            'confidence_interval': {...},
            'horizon_minutes': 1
        },
        # ... más horizontes temporales
    },
    'primary_signal': 'SELL',
    'confidence': 0.495
}
```

### Interpretación de Métricas

- **primary_signal**: Señal principal (BUY/SELL/HOLD)
- **confidence**: Confianza en la señal principal (0-1)
- **uncertainty**: Incertidumbre temporal (0-1)
- **confidence_interval**: Intervalos de confianza para cada señal
- **market_context**: Contexto de mercado detectado

## 🎲 Distribución Chi-Cuadrado

### Parámetros

```python
chi_square_params = {
    'df': 3,        # Grados de libertad (3 timeframes)
    'scale': 1.0,   # Escala de la distribución
    'location': 0.0 # Ubicación de la distribución
}
```

### Cálculo de Incertidumbre

```python
def calculate_chi_square_uncertainty(horizon_minutes, market_context):
    # Ajustar grados de libertad por horizonte
    df = 3 + (horizon_minutes / 10.0)
    
    # Calcular percentil 75
    chi_value = chi2.ppf(0.75, df)
    
    # Normalizar y ajustar por contexto
    uncertainty = min(1.0, chi_value / 10.0)
    context_weight = market_context_weights[market_context]
    
    return uncertainty * context_weight
```

## 🌍 Contexto de Mercado

### Regímenes de Volatilidad

- **low_volatility**: < 2.0 (200%) - Mercados tranquilos
- **normal_volatility**: 2.0 - 5.0 (200% - 500%) - Mercados normales
- **high_volatility**: 5.0 - 15.0 (500% - 1500%) - Mercados volátiles
- **extreme_volatility**: > 15.0 (1500%+) - Mercados extremos

### Pesos por Contexto

```python
market_context_weights = {
    'low_volatility': 1.2,    # Aumentar confianza
    'normal_volatility': 1.0,  # Peso normal
    'high_volatility': 0.8,    # Reducir confianza
    'extreme_volatility': 0.6  # Reducir significativamente
}
```

## ⏰ Horizonte Temporal

### Horizontes Disponibles

- **1m**: Predicción inmediata
- **3m**: Predicción a corto plazo
- **5m**: Predicción a corto-medio plazo
- **10m**: Predicción a medio plazo
- **15m**: Predicción a medio-largo plazo
- **30m**: Predicción a largo plazo
- **60m**: Predicción a muy largo plazo

### Incertidumbre por Horizonte

```
1m:  42.4% (baja incertidumbre)
3m:  45.0%
5m:  47.5%
10m: 53.9%
15m: 60.1%
30m: 78.4%
60m: 100.0% (alta incertidumbre)
```

## 📊 Intervalos de Confianza

### Cálculo de Intervalos

```python
def calculate_confidence_interval(probabilities, uncertainty):
    confidence_interval = {}
    
    for signal, prob in probabilities.items():
        margin_error = prob * uncertainty * 0.5
        
        confidence_interval[signal] = {
            'lower': max(0.0, prob - margin_error),
            'upper': min(1.0, prob + margin_error),
            'center': prob
        }
    
    return confidence_interval
```

### Interpretación

- **lower**: Límite inferior del intervalo
- **upper**: Límite superior del intervalo
- **center**: Probabilidad central
- **margin_error**: Margen de error basado en incertidumbre

## 🔧 Configuración Avanzada

### Personalizar Pesos de Timeframe

```python
system = ProbabilisticForecastingSystem()
system.timeframe_weights = {
    '1m': 0.50,  # Mayor peso para 1m
    '3m': 0.30,  # Peso medio para 3m
    '5m': 0.20   # Menor peso para 5m
}
```

### Personalizar Parámetros Chi-Cuadrado

```python
system.chi_square_params = {
    'df': 4,        # Más grados de libertad
    'scale': 1.5,   # Mayor escala
    'location': 0.0
}
```

### Personalizar Horizontes

```python
system.prediction_horizons = [1, 5, 15, 30, 60, 120]  # Horizontes personalizados
```

## 📈 Ejemplos de Uso

### 1. Predicción para Múltiples Símbolos

```python
symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
system = ProbabilisticForecastingSystem()

for symbol in symbols:
    forecast = system.generate_comprehensive_forecast(symbol)
    if forecast:
        print(f"{symbol}: {forecast['primary_signal']} (conf: {forecast['confidence']*100:.1f}%)")
```

### 2. Análisis Temporal Detallado

```python
forecast = system.generate_comprehensive_forecast('BTCUSDT')

# Crear DataFrame para análisis
temporal_data = []
for horizon, pred in forecast['temporal_predictions'].items():
    temporal_data.append({
        'Horizonte (min)': horizon,
        'Señal': pred['primary_signal'],
        'Confianza (%)': pred['confidence'] * 100,
        'Incertidumbre (%)': pred['uncertainty'] * 100
    })

df = pd.DataFrame(temporal_data)
print(df)
```

### 3. Análisis de Intervalos de Confianza

```python
forecast = system.generate_comprehensive_forecast('BTCUSDT')

for horizon, pred in forecast['temporal_predictions'].items():
    print(f"\nHorizonte {horizon} minutos:")
    for signal, interval in pred['confidence_interval'].items():
        print(f"  {signal}: [{interval['lower']*100:.1f}%, {interval['upper']*100:.1f}%]")
```

## 🎯 Casos de Uso

### 1. **Trading de Alta Frecuencia**
- Usar horizonte de 1-3 minutos
- Alta confianza requerida (>70%)
- Baja incertidumbre (<50%)

### 2. **Trading de Medio Plazo**
- Usar horizonte de 5-15 minutos
- Confianza media (50-70%)
- Incertidumbre moderada (50-70%)

### 3. **Análisis de Tendencias**
- Usar horizonte de 30-60 minutos
- Confianza variable
- Alta incertidumbre aceptable (>70%)

## ⚠️ Consideraciones Importantes

### 1. **Limitaciones**
- Las predicciones son probabilísticas, no determinísticas
- La incertidumbre aumenta con el horizonte temporal
- El contexto de mercado puede cambiar rápidamente

### 2. **Recomendaciones**
- Usar múltiples horizontes para validación
- Considerar el contexto de mercado
- Monitorear la evolución de las predicciones

### 3. **Mejores Prácticas**
- Combinar con análisis fundamental
- Usar stop-loss y take-profit
- Diversificar horizontes temporales

## 🔗 Integración con Otros Sistemas

### Con TCN Ensemble Predictor

```python
from tcn_ensemble_predictor import TCNEnsemblePredictor
from probabilistic_forecasting_system import ProbabilisticForecastingSystem

# Combinar predicciones ML con probabilísticas
tcn_predictor = TCNEnsemblePredictor()
prob_system = ProbabilisticForecastingSystem()

# Obtener predicción ML
ml_prediction = await tcn_predictor.predict_ensemble_v3('BTCUSDT')

# Obtener predicción probabilística
prob_prediction = prob_system.generate_comprehensive_forecast('BTCUSDT')

# Combinar resultados
combined_signal = combine_predictions(ml_prediction, prob_prediction)
```

### Con Sistema de Trading

```python
def generate_trading_signal(symbol):
    # Obtener predicción probabilística
    forecast = generate_probabilistic_forecast(symbol)
    
    if not forecast:
        return None
    
    # Determinar señal de trading
    signal = forecast['primary_signal']
    confidence = forecast['confidence']
    horizon = forecast['recommended_horizon']
    
    # Aplicar filtros de confianza
    if confidence < 0.6:
        return 'HOLD'  # Confianza insuficiente
    
    # Aplicar filtros de horizonte
    if horizon > 15:
        return 'HOLD'  # Horizonte muy largo
    
    return signal
```

## 📚 Referencias

- **Distribución Chi-Cuadrado**: https://en.wikipedia.org/wiki/Chi-squared_distribution
- **Predicción Probabilística**: https://en.wikipedia.org/wiki/Probabilistic_forecasting
- **Análisis Técnico**: https://en.wikipedia.org/wiki/Technical_analysis
- **Trading de Alta Frecuencia**: https://en.wikipedia.org/wiki/High-frequency_trading

## 🆘 Soporte

Para preguntas o problemas con el sistema:

1. Revisar la documentación
2. Ejecutar los ejemplos incluidos
3. Verificar la configuración de la API de Binance
4. Comprobar la conectividad de red

---

**🎯 Sistema de Predicción Probabilística v1.0**  
*Desarrollado para trading algorítmico con distribución chi-cuadrado*
