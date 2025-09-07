# 🛡️ INTEGRACIÓN TCN ENSEMBLE + RISK FILTER

## 📋 Resumen

Esta documentación describe la integración completa del **TCN Ensemble Predictor** con el **Market Risk Filter**, proporcionando una capa de protección contra condiciones de mercado anómalas.

## 🎯 Objetivo

Combinar la potencia predictiva del TCN Ensemble con la detección de anomalías del Risk Filter para:
- **Proteger** contra condiciones de mercado adversas
- **Mejorar** la robustez de las señales de trading
- **Reducir** falsos positivos en mercados volátiles
- **Optimizar** el rendimiento con cache compartido

## 🏗️ Arquitectura de Integración

### Componentes Principales

1. **TCNEnsemblePredictorWithRiskFilter**: Clase principal que extiende el predictor base
2. **SynchronizedRiskFilter**: Filtro de riesgo con cache compartido
3. **Market Risk Filter**: Módulo base de detección de anomalías

### Flujo de Integración

```
📊 Datos de Mercado (1m)
         ↓
🛡️ Risk Filter (Pre-filtrado)
         ↓
🎯 TCN Ensemble Predictor
         ↓
📊 Post-procesamiento de Confianza
         ↓
🛡️ Filtro de Señales Final
         ↓
✅ Señal de Trading Filtrada
```

## 🚀 Instalación y Configuración

### 1. Dependencias

```bash
# Instalar dependencias del Risk Filter
pip install -r requirements_risk_filter.txt

# Verificar que el TCN Ensemble Predictor esté disponible
python -c "from tcn_ensemble_predictor import TCNEnsemblePredictor; print('✅ TCN Predictor disponible')"
```

### 2. Configuración Básica

```python
from tcn_ensemble_predictor_with_risk_filter import TCNEnsemblePredictorWithRiskFilter

# Configuración recomendada para trading activo
predictor = TCNEnsemblePredictorWithRiskFilter(
    risk_filter_config={
        'cache_duration': 300,           # 5 minutos
        'context_update_interval': 60,   # 1 minuto
        'risk_threshold': 40.0,          # Umbral de riesgo
        'min_data_points': 200,          # Datos mínimos para entrenamiento
        'symbols': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'DOTUSDT', 'ADAUSDT']
    }
)
```

### 3. Configuración Avanzada

```python
# Configuración personalizada
predictor = TCNEnsemblePredictorWithRiskFilter(
    risk_filter_config={
        'cache_duration': 180,           # 3 minutos (más agresivo)
        'context_update_interval': 30,   # 30 segundos
        'risk_threshold': 35.0,          # Más conservador
        'min_data_points': 150,          # Menos datos requeridos
        'contamination': 0.15,           # Más sensibilidad a anomalías
        'n_estimators': 150,             # Más árboles para mayor precisión
    }
)

# Configurar integración
predictor.risk_integration_config.update({
    'high_risk_threshold': 65.0,        # Forzar HOLD si riesgo > 65
    'confidence_reduction_max': 0.25,    # Máximo 25% reducción de confianza
    'risk_override_enabled': True,       # Permitir override de señales
})
```

## 📊 Funcionalidades de Integración

### 1. Pre-filtrado Antes de Predicción

```python
# Verificar condiciones de mercado antes de predecir
async def predict_ensemble_v3_with_risk_filter(self, symbol: str):
    # 1. Obtener datos de mercado recientes
    market_data = await self.get_market_data(symbol, '1m', hours=2)
    
    # 2. Evaluar riesgo
    risk_score = self.risk_filter.get_risk_score(market_data, symbol)
    safe_to_trade = self.risk_filter.is_safe_to_trade(market_data, symbol)
    
    # 3. Si no es seguro, retornar HOLD
    if not safe_to_trade:
        return {
            'symbol': symbol,
            'ensemble_signal': 'HOLD',
            'risk_filtered': True,
            'risk_score': risk_score,
            'reason': 'Market conditions deemed risky'
        }
    
    # 4. Continuar con predicción normal
    return await self.predict_ensemble_v3(symbol)
```

### 2. Post-procesamiento de Confianza

```python
def adjust_confidence_with_risk(self, ensemble_result: Dict, symbol: str) -> Dict:
    # Obtener score de riesgo
    risk_score = self.risk_filter.get_risk_score(market_data, symbol)
    original_confidence = ensemble_result['ensemble_confidence']
    
    # Factor de ajuste: risk_score alto = menor confianza
    risk_factor = 1.0 - (risk_score / 100.0) * 0.3  # Máximo 30% reducción
    adjusted_confidence = original_confidence * risk_factor
    
    # Actualizar resultado
    ensemble_result['ensemble_confidence'] = adjusted_confidence
    ensemble_result['risk_adjustment'] = {
        'original_confidence': original_confidence,
        'risk_score': risk_score,
        'risk_factor': risk_factor,
        'adjusted_confidence': adjusted_confidence
    }
    
    return ensemble_result
```

### 3. Filtro de Señales Activas

```python
def filter_trading_signals(self, ensemble_result: Dict, symbol: str) -> Dict:
    risk_score = self.risk_filter.get_risk_score(market_data, symbol)
    high_risk_threshold = 70.0  # Configurable
    
    # Si riesgo es muy alto, forzar HOLD
    if risk_score > high_risk_threshold:
        ensemble_result['ensemble_signal'] = 'HOLD'
        ensemble_result['risk_override'] = True
        ensemble_result['original_signal'] = ensemble_result.get('ensemble_signal')
        ensemble_result['risk_override_reason'] = f'High risk score: {risk_score:.1f}'
    
    return ensemble_result
```

### 4. Cache Compartido

```python
async def get_market_data_with_risk(self, symbol: str, timeframe: str) -> Tuple[pd.DataFrame, float]:
    # Usar cache compartido con TCN predictor
    if self.risk_integration_config['cache_sharing_enabled']:
        market_data = await self.risk_filter.get_synchronized_data(symbol, timeframe)
    else:
        market_data = await self.get_market_data(symbol, timeframe)
    
    # Calcular score de riesgo
    risk_score = self.risk_filter.get_risk_score(market_data, symbol)
    
    return market_data, risk_score
```

## 🎯 Uso Práctico

### 1. Entrenamiento Inicial

```python
# Entrenar filtro de riesgo
async def setup_integration():
    predictor = TCNEnsemblePredictorWithRiskFilter()
    
    # Entrenar filtro con datos históricos
    risk_trained = await predictor.train_risk_filter_with_ensemble(days=7)
    
    if risk_trained:
        print("✅ Filtro de riesgo entrenado exitosamente")
    else:
        print("⚠️ Filtro de riesgo no pudo entrenarse completamente")
    
    return predictor
```

### 2. Predicciones con Filtro

```python
# Predicción individual con filtro
async def predict_with_risk_filter(predictor, symbol: str):
    result = await predictor.predict_ensemble_v3_with_risk_filter(symbol)
    
    if result:
        signal = result['ensemble_signal']
        confidence = result['ensemble_confidence'] * 100
        risk_score = result['risk_score']
        safe_to_trade = result['risk_metrics']['safe_to_trade']
        
        print(f"{symbol}: {signal} ({confidence:.1f}%) | Risk: {risk_score:.1f} | Safe: {safe_to_trade}")
        
        # Lógica de trading
        if safe_to_trade and confidence > 60:
            if signal == 'BUY':
                # Ejecutar orden de compra
                pass
            elif signal == 'SELL':
                # Ejecutar orden de venta
                pass
        else:
            # Mantener posición actual
            pass
    
    return result

# Predicciones múltiples
async def predict_all_with_risk_filter(predictor):
    results = await predictor.predict_all_symbols_v3_with_risk()
    
    for symbol, result in results.items():
        await predict_with_risk_filter(predictor, symbol)
    
    return results
```

### 3. Monitoreo y Métricas

```python
# Obtener estado de integración
def monitor_integration(predictor):
    status = predictor.get_risk_integration_status()
    
    print("📊 Estado de Integración:")
    print(f"   Filtro entrenado: {status['risk_filter_trained']}")
    print(f"   Configuración: {status['integration_config']}")
    
    metrics = status['integration_metrics']
    print(f"   Señales filtradas: {metrics['signals_filtered']}")
    print(f"   Ajustes de confianza: {metrics['confidence_adjustments']}")
    print(f"   Overrides de riesgo: {metrics['risk_overrides']}")
    print(f"   Cache hits: {metrics['cache_hits']}")
    print(f"   Impacto performance: {metrics['performance_impact_ms']:.1f}ms")
    
    return status
```

## ⚙️ Configuración Avanzada

### 1. Thresholds Configurables

```python
# Configurar thresholds según estrategia
predictor.risk_integration_config.update({
    'high_risk_threshold': 70.0,        # Forzar HOLD si riesgo > 70
    'confidence_reduction_max': 0.3,     # Máximo 30% reducción de confianza
    'risk_override_enabled': True,       # Permitir override de señales
    'calibration_enabled': True,         # Habilitar calibración
    'calibration_window': 1000,          # Ventana para calibración
})
```

### 2. Cache Adaptativo

```python
# Configurar cache adaptativo por volatilidad
predictor.risk_filter.volatility_cache_duration = {
    'extreme_volatility': 60,    # 1 minuto
    'high_volatility': 300,      # 5 minutos
    'normal_volatility': 900,    # 15 minutos
    'low_volatility': 1800       # 30 minutos
}
```

### 3. Configuración por Símbolo

```python
# Configuración específica por símbolo
symbol_configs = {
    'BTCUSDT': {'risk_threshold': 35.0, 'high_risk_threshold': 65.0},
    'ETHUSDT': {'risk_threshold': 40.0, 'high_risk_threshold': 70.0},
    'XRPUSDT': {'risk_threshold': 45.0, 'high_risk_threshold': 75.0},
}

for symbol, config in symbol_configs.items():
    # Aplicar configuración específica
    pass
```

## 📈 Métricas y Monitoreo

### 1. Métricas de Integración

```python
# Obtener métricas completas
metrics = predictor.risk_integration_metrics

print("📊 Métricas de Integración:")
print(f"   Señales filtradas: {metrics['signals_filtered']}")
print(f"   Ajustes de confianza: {metrics['confidence_adjustments']}")
print(f"   Overrides de riesgo: {metrics['risk_overrides']}")
print(f"   Cache hits: {metrics['cache_hits']}")
print(f"   Impacto performance: {metrics['performance_impact_ms']:.1f}ms")
```

### 2. Información del Cache

```python
# Obtener información del cache adaptativo
cache_info = predictor.risk_filter.get_cache_info()

print("📊 Información del Cache:")
print(f"   Duración base: {cache_info['base_duration']}s")
print(f"   Intervalo contexto: {cache_info['context_interval']}s")
print(f"   Entradas en cache: {cache_info['cache_entries']}")

for symbol, regime_info in cache_info['volatility_regimes'].items():
    print(f"   {symbol}: {regime_info['regime']} (cache: {regime_info['cache_duration']}s)")
```

### 3. Estadísticas de Anomalías

```python
# Obtener estadísticas de anomalías por símbolo
for symbol in predictor.symbols:
    stats = predictor.risk_filter.get_anomaly_statistics(symbol)
    print(f"{symbol}: {stats['total_anomalies']} anomalías totales, {stats['recent_anomalies']} recientes")
```

## 🔧 Troubleshooting

### 1. Problemas Comunes

#### Error: "Filtro no entrenado"
```python
# Solución: Entrenar filtro
await predictor.train_risk_filter_with_ensemble(days=7)
```

#### Error: "Datos insuficientes"
```python
# Solución: Reducir min_data_points
predictor = TCNEnsemblePredictorWithRiskFilter(
    risk_filter_config={'min_data_points': 100}
)
```

#### Error: "Cache expirado"
```python
# Solución: Aumentar duración de cache
predictor = TCNEnsemblePredictorWithRiskFilter(
    risk_filter_config={'cache_duration': 600}  # 10 minutos
)
```

### 2. Optimización de Performance

```python
# Configuración optimizada para alta frecuencia
predictor = TCNEnsemblePredictorWithRiskFilter(
    risk_filter_config={
        'cache_duration': 180,           # 3 minutos
        'context_update_interval': 30,   # 30 segundos
        'min_data_points': 150,          # Menos datos
        'n_estimators': 100,             # Menos árboles
    }
)

# Deshabilitar funcionalidades no críticas
predictor.risk_integration_config.update({
    'calibration_enabled': False,        # Deshabilitar calibración
    'post_processing_enabled': False,    # Deshabilitar post-procesamiento
})
```

## 🧪 Testing

### 1. Test de Integración

```bash
# Ejecutar test completo
python test_tcn_risk_integration.py
```

### 2. Test de Performance

```python
# Medir tiempo de predicción
import time

start_time = time.time()
result = await predictor.predict_ensemble_v3_with_risk_filter('BTCUSDT')
end_time = time.time()

prediction_time = (end_time - start_time) * 1000
print(f"Predicción completada en {prediction_time:.1f}ms")
```

## 📚 Referencias

- [Market Risk Filter Documentation](MARKET_RISK_FILTER_INTEGRATION.md)
- [TCN Ensemble Predictor](tcn_ensemble_predictor.py)
- [SynchronizedRiskFilter](market_risk_filter.py)

## 🤝 Contribución

Para contribuir a la integración:

1. Revisar el código existente
2. Proponer mejoras en issues
3. Crear pull requests con cambios
4. Mantener compatibilidad con versiones anteriores

## 📄 Licencia

Este proyecto mantiene la misma licencia que el proyecto principal.
