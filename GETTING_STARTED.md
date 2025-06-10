# 🎓 Getting Started - Educational Trading Bot

Guía paso a paso para experimentar con el trading bot educacional.

## ⚠️ Importante - Solo Educacional

Este es un proyecto **puramente educacional**. NO ejecuta trades reales y solo funciona con testnet de Binance.

## 🚀 Configuración Rápida

### 1. **Preparar Entorno**

```bash
# Activar entorno virtual
source .venv/bin/activate

# Instalar dependencias educacionales
pip install -r requirements.txt
```

### 2. **Configurar Variables (Opcional)**

Crea un archivo `.env` (o usa defaults educacionales):

```bash
# .env - Configuración educacional
DRY_RUN=true                    # 🚨 SIEMPRE true para educación
BINANCE_TESTNET=true           # 🚨 SIEMPRE true para aprendizaje
ENVIRONMENT=development        # 🚨 NUNCA production

# Configuración experimental
TRADING_SYMBOLS=["BTCUSDT", "ETHUSDT"]
TRADING_INTERVAL_SECONDS=30
MAX_POSITION_PERCENT=0.001     # 0.1% experimental
MAX_DAILY_LOSS_PERCENT=0.005   # 0.5% experimental

# Credenciales de testnet (opcionales para demo)
# BINANCE_API_KEY=tu_testnet_key
# BINANCE_SECRET=tu_testnet_secret
```

### 3. **Verificar Modelo TCN**

```bash
# Verificar que el modelo esté presente
python -c "
import tensorflow as tf
model = tf.keras.models.load_model('models/tcn_anti_bias_fixed.h5')
print(f'✅ Modelo TCN: {model.count_params():,} parámetros')
print('✅ Modelo listo para experimentación')
"
```

## 🎯 Modos de Ejecución

### **Modo 1: Demo Automática (5 minutos)**

Ejecuta una demostración automática que muestra todo el sistema funcionando:

```bash
python main_educational.py
# Seleccionar opción: 1
```

**Qué hace:**
- Inicializa todos los servicios SOLID
- Ejecuta ciclo completo de trading por 5 minutos
- Muestra logging estructurado en tiempo real
- Demuestra integración ML + Risk Management

### **Modo 2: Interactivo Manual**

Ejecuta el sistema indefinidamente hasta que lo detengas:

```bash
python main_educational.py
# Seleccionar opción: 2
# Presionar Ctrl+C para detener
```

**Qué hace:**
- Sistema funciona continuamente
- Estadísticas cada minuto
- Control manual de start/stop
- Ideal para experimentación extendida

### **Modo 3: Solo Estado**

Muestra el estado de todos los servicios sin ejecutar trading:

```bash
python main_educational.py
# Seleccionar opción: 3
```

**Qué hace:**
- Verifica conexiones y servicios
- Muestra configuración actual
- Performance del modelo ML
- Útil para debugging

## 📊 Qué Podrás Observar

### **1. Logging Estructurado Educacional**

```json
{
  "event": "educational_ml_prediction",
  "symbol": "BTCUSDT",
  "action": "BUY",
  "confidence": 0.75,
  "model_type": "TCN",
  "educational_note": "Predicción generada por modelo TCN experimental"
}
```

### **2. Ciclo Completo de Trading**

1. **Obtención de datos** → Binance testnet en tiempo real
2. **Feature Engineering** → 14 indicadores técnicos (RSI, MACD, Bollinger)
3. **Predicción ML** → Modelo TCN con 60 períodos de secuencia
4. **Evaluación de Riesgo** → 6 capas de validación
5. **Ejecución Simulada** → Orden DRY-RUN (no real)

### **3. Servicios SOLID en Acción**

- **EducationalBinanceClient** → Interface ITradingClient
- **EducationalMLPredictor** → Interface IMLPredictor  
- **EducationalRiskManager** → Interface IRiskManager
- **EducationalTradingOrchestrator** → Coordinador principal

## 🧠 Características del Modelo ML

### **TCN (Temporal Convolutional Networks)**
- **Parámetros**: ~1.1M parámetros entrenados
- **Secuencia**: 60 períodos históricos
- **Features**: 14 indicadores técnicos
- **Output**: Probabilidad de dirección del precio

### **Feature Engineering Educacional**
1. Precio y volumen normalizados
2. RSI (Relative Strength Index)
3. MACD (Moving Average Convergence Divergence)
4. Bollinger Bands
5. Volatilidad y momentum
6. Ratios de volumen

## 🛡️ Sistema de Risk Management

### **6 Capas de Validación**
1. **Validación de símbolo** → Solo pares permitidos
2. **Validación de balance** → Fondos suficientes
3. **Position sizing** → Kelly Criterion + volatilidad
4. **Circuit breakers** → Parada automática ante pérdidas
5. **Daily limits** → Límites diarios configurables
6. **Cooldown periods** → Prevención de overtrading

### **Configuración de Seguridad Educacional**
- Máximo 0.1% por posición
- Máximo 0.5% pérdida diaria
- Circuit breaker a 1%
- Todas las órdenes en DRY-RUN

## 📁 Estructura del Código

```
src/
├── interfaces/          # Contratos SOLID
│   └── trading_interfaces.py
├── schemas/            # Validación Pydantic
│   └── trading_schemas.py
├── core/               # Configuración y Factory
│   ├── config.py
│   ├── logging_config.py
│   └── service_factory.py
├── services/           # Implementaciones
│   ├── binance_client.py
│   ├── ml_predictor.py
│   ├── risk_manager.py
│   └── trading_orchestrator.py
└── database/           # Modelos SQLAlchemy
    └── models.py
```

## 🧪 Experimentación Avanzada

### **Modificar Configuración ML**

```python
# En ml_predictor.py - línea 53
self.confidence_threshold = 0.6  # Cambiar threshold
```

### **Ajustar Risk Management**

```python
# En service_factory.py - línea 305
max_position_percent=0.001,   # Cambiar tamaño de posición
max_daily_loss_percent=0.005, # Cambiar límite diario
```

### **Cambiar Símbolos de Trading**

```python
# En main_educational.py - línea 80
trading_symbols=["BTCUSDT", "ETHUSDT", "ADAUSDT"],
```

### **Modificar Intervalos**

```python
# En main_educational.py - línea 81
trading_interval_seconds=60,  # Cambiar frecuencia
```

## 🔍 Debugging y Troubleshooting

### **Problemas Comunes**

1. **Error: Modelo no encontrado**
   ```bash
   # Verificar que existe
   ls -la models/tcn_anti_bias_fixed.h5
   ```

2. **Error: TensorFlow**
   ```bash
   # Reinstalar TensorFlow
   pip install --upgrade tensorflow==2.15.0
   ```

3. **Error: Binance API**
   ```bash
   # Verificar conexión (usa datos dummy en modo educational)
   python -c "from binance.client import Client; print('Binance client OK')"
   ```

### **Logs Detallados**

Los logs se guardan en formato JSON estructurado para análisis:

```bash
# Ver logs en tiempo real
tail -f logs/trading_*.log | jq '.'
```

## 📚 Próximos Pasos de Aprendizaje

### **1. Entender el Código**
- Leer interfaces en `src/interfaces/`
- Analizar implementaciones en `src/services/`
- Revisar validaciones en `src/schemas/`

### **2. Experimentar con ML**
- Modificar features en `ml_predictor.py`
- Cambiar thresholds de confianza
- Analizar predicciones del modelo

### **3. Probar Risk Management**
- Ajustar límites de riesgo
- Modificar circuit breakers
- Experimentar con position sizing

### **4. Extender Funcionalidad**
- Añadir nuevos indicadores técnicos
- Implementar notificaciones
- Crear estrategias personalizadas

## 🎯 Objetivos Educacionales

Al completar esta experimentación habrás aprendido:

- ✅ **Arquitectura SOLID** en Python
- ✅ **Machine Learning** aplicado a trading
- ✅ **Risk Management** algorítmico
- ✅ **APIs** de exchanges de crypto
- ✅ **Logging estructurado** profesional
- ✅ **Testing** con mocks y fixtures
- ✅ **Patrones de diseño** (Factory, Strategy, Observer)

---

**🎓 ¡Disfruta experimentando y aprendiendo!**

**Recuerda**: Este es un proyecto puramente educacional. Nunca uses dinero real para trading algorítmico sin una comprensión profunda de los riesgos involucrados. 