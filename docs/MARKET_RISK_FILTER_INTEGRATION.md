# 🛡️ MARKET RISK FILTER - Integración con Bot de Trading

## 📋 Descripción General

El `MarketRiskFilter` es un módulo standalone que utiliza **Isolation Forest** para detectar anomalías de mercado en tiempo real. Se integra perfectamente con tu bot de trading TCN existente para filtrar condiciones de mercado adversas antes de ejecutar señales.

## 🎯 Características Principales

### ✅ Funcionalidades Core
- **Detección de Anomalías**: Isolation Forest para identificar patrones anómalos
- **Features Automáticas**: 20 features técnicas calculadas automáticamente
- **Score de Riesgo**: Escala continua de 0-100
- **Flag de Trading**: Booleano para decisión inmediata
- **Persistencia**: Guardar/cargar modelos entrenados
- **Cache Inteligente**: Optimización de requests a Binance API

### 🎯 Features Técnicas Automáticas
1. **Precio**: Volatilidad, momentum, aceleración, reversión
2. **Volumen**: Volatilidad, momentum, aceleración, reversión
3. **Velas**: Spread ratio, wick ratio, body ratio
4. **Correlación**: Volumen-precio
5. **Eficiencia**: Precios y volumen (Hurst-like)
6. **Impacto**: Market impact, liquidity ratio
7. **Régimen**: Volatilidad, trend strength
8. **Reversión**: Mean reversion, momentum divergence

## 🔧 Instalación y Dependencias

### Dependencias Requeridas
```bash
pip install scikit-learn pandas numpy aiohttp talib
```

### Estructura de Archivos
```
├── market_risk_filter.py          # Módulo principal
├── models/                        # Directorio para modelos
│   └── risk_model.pkl            # Modelo entrenado
├── docs/
│   └── MARKET_RISK_FILTER_INTEGRATION.md
└── market_risk_filter.log         # Logs automáticos
```

## 🚀 Uso Básico

### 1. Crear Instancia
```python
from market_risk_filter import MarketRiskFilter

# Configuración recomendada para crypto
risk_filter = MarketRiskFilter(
    contamination=0.1,        # 10% anomalías esperadas
    risk_threshold=40.0,      # Umbral de riesgo
    min_data_points=1000,     # Mínimo datos para entrenar
    n_estimators=100,         # Árboles en ensemble
    max_samples=256           # Muestras por árbol
)
```

### 2. Entrenar Modelo
```python
import asyncio

async def train_risk_filter():
    # Obtener datos históricos "normales"
    training_data = await risk_filter.get_binance_data('BTCUSDT', '1m', 2000)
    
    # Entrenar modelo
    if risk_filter.fit(training_data):
        print("✅ Modelo entrenado exitosamente")
        
        # Guardar modelo
        risk_filter.save_model('models/risk_model.pkl')
    else:
        print("❌ Error en entrenamiento")

# Ejecutar entrenamiento
asyncio.run(train_risk_filter())
```

### 3. Predicción en Tiempo Real
```python
# Cargar modelo entrenado
risk_filter.load_model('models/risk_model.pkl')

# Obtener datos recientes
recent_data = await risk_filter.get_binance_data('BTCUSDT', '1m', 100)

# Predecir riesgo
risk_score = risk_filter.get_risk_score(recent_data)
safe_to_trade = risk_filter.is_safe_to_trade(recent_data)

print(f"Risk Score: {risk_score:.1f}/100")
print(f"Safe to Trade: {safe_to_trade}")
```

## 🔗 Integración con TCN Ensemble Predictor

### Integración en Loop Principal
```python
# En tu bot principal (tcn_ensemble_predictor.py)

from market_risk_filter import MarketRiskFilter

class TCNEnsemblePredictor:
    def __init__(self):
        # ... código existente ...
        
        # 🛡️ AGREGAR FILTRO DE RIESGO
        self.risk_filter = MarketRiskFilter(
            contamination=0.1,
            risk_threshold=40.0
        )
        
        # Cargar modelo entrenado
        if os.path.exists('models/risk_model.pkl'):
            self.risk_filter.load_model('models/risk_model.pkl')
            print("✅ Filtro de riesgo cargado")
        else:
            print("⚠️ Modelo de riesgo no encontrado")
    
    async def get_market_data(self, symbol: str, timeframe: str, hours: int = 1):
        """Obtener datos de mercado con validación de riesgo"""
        try:
            # Obtener datos OHLCV
            market_data = await self._get_ohlcv_data(symbol, timeframe, hours)
            
            if market_data.empty:
                return pd.DataFrame()
            
            # 🛡️ VALIDAR RIESGO DE MERCADO
            if timeframe == '1m':  # Solo validar en 1m
                risk_score = self.risk_filter.get_risk_score(market_data)
                safe_to_trade = self.risk_filter.is_safe_to_trade(market_data)
                
                print(f"🛡️ Risk Score: {risk_score:.1f}/100, Safe: {safe_to_trade}")
                
                # Si no es seguro, retornar DataFrame vacío
                if not safe_to_trade:
                    print("⚠️ Condiciones de mercado anómalas detectadas")
                    return pd.DataFrame()
            
            return market_data
            
        except Exception as e:
            print(f"❌ Error obteniendo datos: {e}")
            return pd.DataFrame()
    
    async def predict_ensemble(self, symbol: str):
        """Predicción con filtro de riesgo"""
        try:
            # Obtener datos de múltiples timeframes
            data_1m = await self.get_market_data(symbol, '1m', hours=1)
            data_3m = await self.get_market_data(symbol, '3m', hours=3)
            data_5m = await self.get_market_data(symbol, '5m', hours=5)
            
            # 🛡️ VERIFICAR RIESGO ANTES DE PREDECIR
            if data_1m.empty:
                print("⚠️ Datos 1m bloqueados por filtro de riesgo")
                return None, 0.0
            
            # ... resto de lógica de predicción ...
            
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            return None, 0.0
```

### Integración en Trading Manager
```python
# En professional_trading_manager.py

class ProfessionalTradingManager:
    def __init__(self):
        # ... código existente ...
        
        # 🛡️ AGREGAR FILTRO DE RIESGO
        self.risk_filter = MarketRiskFilter(risk_threshold=35.0)
        self.risk_filter.load_model('models/risk_model.pkl')
    
    async def execute_signal(self, signal_data):
        """Ejecutar señal con validación de riesgo"""
        try:
            # Obtener datos 1m para validación
            market_data = await self.risk_filter.get_binance_data(
                signal_data['symbol'], '1m', 100
            )
            
            if market_data.empty:
                print("❌ No se pudieron obtener datos para validación de riesgo")
                return False
            
            # 🛡️ VALIDAR RIESGO ANTES DE EJECUTAR
            risk_score = self.risk_filter.get_risk_score(market_data)
            safe_to_trade = self.risk_filter.is_safe_to_trade(market_data)
            
            print(f"🛡️ Risk Score: {risk_score:.1f}/100")
            
            if not safe_to_trade:
                print("⚠️ Señal bloqueada por condiciones de riesgo")
                return False
            
            # Si es seguro, ejecutar señal
            if risk_score < 40:
                return await self._execute_trade(signal_data)
            else:
                print("⚠️ Score de riesgo demasiado alto")
                return False
                
        except Exception as e:
            print(f"❌ Error ejecutando señal: {e}")
            return False
```

## ⚙️ Configuración Avanzada

### Parámetros Recomendados por Mercado

#### Bitcoin (BTCUSDT)
```python
risk_filter = MarketRiskFilter(
    contamination=0.08,       # Menos anomalías (mercado más estable)
    risk_threshold=35.0,      # Umbral más conservador
    n_estimators=150,         # Más árboles para mayor precisión
    max_samples=512           # Más muestras por árbol
)
```

#### Altcoins (XRPUSDT, DOTUSDT)
```python
risk_filter = MarketRiskFilter(
    contamination=0.12,       # Más anomalías (mercado más volátil)
    risk_threshold=45.0,      # Umbral más permisivo
    n_estimators=100,         # Árboles estándar
    max_samples=256           # Muestras estándar
)
```

#### Mercado Bearish
```python
risk_filter = MarketRiskFilter(
    contamination=0.15,       # Más anomalías en bear market
    risk_threshold=30.0,      # Umbral más conservador
    n_estimators=200,         # Más árboles para robustez
    max_samples=128           # Menos muestras para adaptabilidad
)
```

### Ajuste Dinámico de Umbral
```python
# Ajustar umbral según condiciones de mercado
def adjust_risk_threshold(risk_filter, market_conditions):
    if market_conditions == 'bullish':
        risk_filter.update_risk_threshold(45.0)
    elif market_conditions == 'bearish':
        risk_filter.update_risk_threshold(25.0)
    else:  # neutral
        risk_filter.update_risk_threshold(35.0)
```

## 📊 Monitoreo y Estadísticas

### Obtener Estadísticas
```python
# Estadísticas de anomalías detectadas
stats = risk_filter.get_anomaly_statistics()

print(f"📊 Estadísticas de Riesgo:")
print(f"   - Anomalías totales: {stats['total_anomalies']}")
print(f"   - Anomalías recientes (24h): {stats['recent_anomalies']}")
print(f"   - Score promedio: {stats['avg_risk_score']:.1f}")

if stats['last_anomaly']:
    print(f"   - Última anomalía: {stats['last_anomaly']['timestamp']}")
```

### Logs Automáticos
El módulo genera logs automáticos en `market_risk_filter.log`:
```
2024-01-15 10:30:15 - INFO - 🛡️ MarketRiskFilter inicializado con threshold: 40.0
2024-01-15 10:30:20 - INFO - 📊 Obteniendo datos de Binance para BTCUSDT (1m)...
2024-01-15 10:30:22 - INFO - ✅ Datos obtenidos: 1000 registros
2024-01-15 10:30:25 - INFO - 🎯 Predicción: Risk=25.3, Safe=True, Anomaly=False
```

## 🎯 Casos de Uso Específicos

### 1. Filtrado de Pump & Dump
```python
# Detectar movimientos sospechosos de volumen
if risk_score > 70 and volume_spike_detected:
    print("🚨 Posible pump & dump detectado")
    return False
```

### 2. Protección en Noticias
```python
# Durante eventos de noticias importantes
if news_event_active and risk_score > 50:
    print("📰 Condiciones de noticias detectadas")
    return False
```

### 3. Filtrado de Manipulación
```python
# Detectar patrones de manipulación
if manipulation_pattern_detected and risk_score > 60:
    print("🎭 Posible manipulación detectada")
    return False
```

## 🔧 Mantenimiento y Optimización

### Reentrenamiento Periódico
```python
async def retrain_weekly():
    """Reentrenar modelo semanalmente"""
    risk_filter = MarketRiskFilter()
    
    # Obtener datos de la última semana
    training_data = await risk_filter.get_binance_data('BTCUSDT', '1m', 10080)  # 7 días
    
    if risk_filter.fit(training_data):
        risk_filter.save_model('models/risk_model_updated.pkl')
        print("✅ Modelo reentrenado exitosamente")

# Programar reentrenamiento semanal
import schedule
schedule.every().monday.at("02:00").do(retrain_weekly)
```

### Backup de Modelos
```python
import shutil
from datetime import datetime

def backup_model():
    """Crear backup del modelo actual"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"models/backup/risk_model_{timestamp}.pkl"
    
    shutil.copy('models/risk_model.pkl', backup_path)
    print(f"✅ Backup creado: {backup_path}")
```

## 🚨 Troubleshooting

### Problemas Comunes

#### 1. "Modelo no entrenado"
```python
# Solución: Entrenar modelo primero
if not risk_filter.is_trained:
    training_data = await risk_filter.get_binance_data('BTCUSDT', '1m', 2000)
    risk_filter.fit(training_data)
```

#### 2. "Features faltantes"
```python
# Solución: Verificar datos OHLCV
if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
    print("❌ Datos OHLCV incompletos")
```

#### 3. "Score siempre alto"
```python
# Solución: Ajustar contamination
risk_filter = MarketRiskFilter(contamination=0.05)  # Menos anomalías
```

### Validación de Datos
```python
# Verificar integridad de datos
def validate_data_integrity(df):
    issues = []
    
    if df.empty:
        issues.append("DataFrame vacío")
    
    if df.isnull().any().any():
        issues.append("Valores nulos detectados")
    
    if (df <= 0).any().any():
        issues.append("Valores negativos o cero")
    
    return issues
```

## 📈 Métricas de Rendimiento

### Métricas Recomendadas
- **Precisión de Anomalías**: > 85%
- **Recall de Anomalías**: > 80%
- **Falsos Positivos**: < 15%
- **Tiempo de Respuesta**: < 100ms

### Evaluación del Modelo
```python
def evaluate_model_performance(risk_filter, test_data):
    """Evaluar rendimiento del modelo"""
    predictions = []
    actual_anomalies = []  # Basado en eventos conocidos
    
    for i in range(len(test_data)):
        batch = test_data.iloc[i:i+100]
        risk_score, safe = risk_filter.predict(batch)
        predictions.append(risk_score)
    
    # Calcular métricas
    precision = precision_score(actual_anomalies, [p > 50 for p in predictions])
    recall = recall_score(actual_anomalies, [p > 50 for p in predictions])
    
    print(f"Precisión: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
```

## 🎯 Conclusión

El `MarketRiskFilter` proporciona una capa adicional de protección para tu bot de trading, detectando condiciones de mercado anómalas antes de ejecutar señales. Se integra perfectamente con tu arquitectura TCN existente y mejora significativamente la robustez del sistema.

### Próximos Pasos
1. Entrenar modelo con datos históricos
2. Integrar en loop principal del bot
3. Monitorear rendimiento y ajustar parámetros
4. Implementar reentrenamiento automático

---

**⚠️ IMPORTANTE**: Este módulo usa ÚNICAMENTE datos reales de Binance. No se permiten datos inventados o simulados.
