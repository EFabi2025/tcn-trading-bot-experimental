# 🧪 TCN Trading Bot - **PROYECTO EXPERIMENTAL**

> **⚠️ IMPORTANTE**: Este es un proyecto **puramente educacional y experimental**. No está destinado para uso en producción real ni trading con dinero real. Su propósito es el aprendizaje y la experimentación con tecnologías de ML y trading.

## 🎓 Objetivo Educacional

Este proyecto explora la implementación de un trading bot profesional combinando:
- **Machine Learning**: Redes neuronales TCN (Temporal Convolutional Networks)
- **Arquitectura SOLID**: Principios de ingeniería de software profesional
- **Gestión de Riesgo**: Implementación de sistemas de control de riesgo
- **Buenas Prácticas**: Logging estructurado, validación de datos, testing

## ✅ Estado del Proyecto - **EXPERIMENTAL**

### 🏗️ ARQUITECTURA IMPLEMENTADA (Para Aprendizaje)

- **✅ Interfaces SOLID**: Contratos definidos siguiendo principios SOLID
- **✅ Schemas Pydantic**: Validación estricta de datos con tipos seguros
- **✅ Configuración Centralizada**: BaseSettings con validación automática
- **✅ Logging Estructurado**: Structlog con contexto profesional
- **✅ Modelos SQLAlchemy**: Base de datos con integridad referencial
- **✅ Risk Manager**: Servicio experimental de gestión de riesgos

### 📋 COMPONENTES EDUCACIONALES

#### 🔧 Core Infrastructure (Para Estudio)
- `src/interfaces/trading_interfaces.py` - Interfaces y contratos
- `src/schemas/trading_schemas.py` - Validación Pydantic
- `src/core/config.py` - Configuración centralizada
- `src/core/logging_config.py` - Logging estructurado
- `src/database/models.py` - Modelos SQLAlchemy

#### 🛡️ Risk Management (Experimental)
- `src/services/risk_manager.py` - Gestión de riesgos educacional
  - ✅ Validación multi-capa de órdenes
  - ✅ Cálculo de posiciones (Kelly Criterion)
  - ✅ Circuit breakers automáticos
  - ✅ Límites de pérdida diaria
  - ✅ Análisis de correlación

### 🚀 Configuración para Experimentación

```bash
# 1. Activar entorno virtual
source .venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variables (SOLO TESTNET)
cp .env.example .env
# Editar .env - MANTENER DRY_RUN=true

# 4. Verificar modelo TCN experimental
python -c "
import tensorflow as tf
model = tf.keras.models.load_model('models/tcn_anti_bias_fixed.h5')
print(f'✅ Modelo experimental: {model.count_params():,} parámetros')
"
```

### 📁 Estructura del Proyecto Experimental

```
.
├── src/                          # Código experimental
│   ├── interfaces/               # Interfaces SOLID
│   ├── schemas/                  # Validación Pydantic
│   ├── core/                     # Configuración
│   ├── database/                 # Modelos experimentales
│   ├── services/                 # Servicios de prueba
│   └── models/                   # Modelos ML experimentales
├── models/                       # Modelos entrenados (1.1MB)
├── tests/                        # Tests educacionales
├── .env.example                  # Configuración de ejemplo
└── requirements.txt              # Dependencias
```

## 🔜 Experimentos Planificados

### 1. **Servicios de Trading** (Experimentales)
- [ ] `BinanceClient` - Cliente de prueba con Binance API
- [ ] `MLPredictor` - Integración experimental del modelo TCN
- [ ] `MarketDataProvider` - Proveedor de datos para experimentos

### 2. **Base de Datos** (Para Aprendizaje)
- [ ] Migraciones con Alembic
- [ ] Repositories experimentales

### 3. **Testing** (Educacional)
- [ ] Tests unitarios como ejemplos
- [ ] Tests de integración educacionales

## 🛡️ Características de Seguridad (Experimentales)

### Protecciones Implementadas
- **DRY RUN**: Por defecto, NO ejecuta trades reales
- **TESTNET**: Solo opera en entornos de prueba
- **Circuit Breakers**: Parada automática ante riesgos
- **Validación Multi-capa**: 6 niveles de validación
- **Logging Completo**: Trazabilidad de todas las operaciones

## ⚠️ **CONFIGURACIÓN OBLIGATORIA DE SEGURIDAD**

```bash
# En .env - CONFIGURACIÓN EXPERIMENTAL
DRY_RUN=true                     # 🚨 NUNCA cambiar a false
BINANCE_TESTNET=true            # 🚨 SOLO testnet
ENVIRONMENT=development         # 🚨 NUNCA production
MAX_POSITION_PERCENT=0.01       # 🚨 Máximo 1% experimental
MAX_DAILY_LOSS_PERCENT=0.02     # 🚨 Máximo 2% de prueba
```

## 📚 Propósito Educacional

### Tecnologías Exploradas
- **TensorFlow**: Redes neuronales TCN
- **Pydantic**: Validación de datos robusta
- **SQLAlchemy**: ORM y gestión de base de datos
- **Structlog**: Logging estructurado profesional
- **AsyncIO**: Programación asíncrona
- **SOLID Principles**: Arquitectura limpia

### Patrones de Diseño Implementados
- **Repository Pattern**: Abstracción de datos
- **Strategy Pattern**: Algoritmos intercambiables
- **Observer Pattern**: Notificaciones asíncronas
- **Factory Pattern**: Creación controlada

## 🎯 Disclaimer

> **⚠️ AVISO IMPORTANTE**: 
> 
> Este proyecto es **únicamente para fines educacionales y experimentales**. 
> 
> - ❌ **NO** usar con dinero real
> - ❌ **NO** es asesoramiento financiero
> - ❌ **NO** garantiza rentabilidad
> - ✅ **SÍ** es para aprender programación
> - ✅ **SÍ** es para experimentar con ML
> - ✅ **SÍ** es para practicar buenas prácticas de código

## 📄 Licencia

MIT License - Proyecto educacional abierto para el aprendizaje.

---

**🎯 OBJETIVO**: Aprender sobre ML, trading algorítmico y arquitectura de software
**🧪 ESTADO**: Experimental - Solo para educación y experimentación
**📖 PROPÓSITO**: Compartir conocimiento y técnicas de programación
