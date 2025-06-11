# 🚀 Professional Trading Bot con TCN

Sistema de trading automatizado profesional usando **Temporal Convolutional Networks (TCN)** para predicciones de mercado con datos reales de Binance.

## 🎯 Características Principales

### 🤖 **IA Avanzada**
- **Modelos TCN** entrenados para BTC, ETH y BNB
- **Datos reales de Binance** via API
- **21 indicadores técnicos** profesionales
- **TensorFlow 2.15.0** optimizado para Apple Silicon

### 💼 **Portfolio Management Profesional**
- **Seguimiento de posiciones múltiples** por par
- **Algoritmo FIFO** para cálculo preciso de PnL
- **Reportes TCN cada 5 minutos** estilo profesional
- **Gestión de riesgo avanzada** con trailing stops

### 🛡️ **Gestión de Riesgo**
- **Stop Loss automático:** 3%
- **Take Profit:** 6%
- **Trailing Stops** por posición individual
- **Límites de exposición:** 15% por posición, 2 posiciones máx

### 🔔 **Notificaciones Inteligentes**
- **Discord** con filtros inteligentes
- **Reportes TCN** cada 5 minutos
- **Alertas de trading** con contexto completo

## 📋 Requisitos del Sistema

### 🖥️ **Compatibilidad**
- **Python 3.10+**
- **macOS** (optimizado para Apple Silicon M1/M2/M3)
- **Windows/Linux** (compatible)

### 🔑 **APIs Requeridas**
- **Binance API** (testnet o producción)
- **Discord Webhook** (opcional)

## 🚀 Instalación Rápida

### 1️⃣ **Clonar Repositorio**
```bash
git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
cd BinanceBotClean_20250610_095103
```

### 2️⃣ **Crear Entorno Virtual**
```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# o
.venv\Scripts\activate     # Windows
```

### 3️⃣ **Instalar Dependencias**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4️⃣ **Configurar Variables de Entorno**
```bash
cp .env.example .env
# Editar .env con tus credenciales
```

**Configuración mínima en `.env`:**
```env
# Binance API
BINANCE_API_KEY=tu_api_key_aqui
BINANCE_SECRET_KEY=tu_secret_key_aqui
BINANCE_BASE_URL=https://testnet.binance.vision  # Para testnet

# Discord (opcional)
DISCORD_WEBHOOK_URL=tu_webhook_discord

# Configuración de Trading
ENVIRONMENT=testnet
MAX_POSITION_SIZE_PERCENT=15
TRADE_MODE=live
```

### 5️⃣ **Ejecutar el Bot**
```bash
python run_trading_manager.py
```

## 🎯 Uso

### 🔴 **Inicio Rápido**
```bash
# Ejecutar sistema completo
python run_trading_manager.py

# Probar señales TCN
python test_tcn_signals.py

# Verificar modelos
python analyze_model_requirements.py
```

### 📊 **Monitoreo**
El sistema mostrará:
- ✅ **Balance en tiempo real** desde Binance
- 🤖 **Predicciones TCN** con datos reales
- 💼 **Posiciones individuales** con PnL
- 🛡️ **Estado de trailing stops**
- 📈 **Reportes TCN cada 5 minutos**

## 🏗️ Arquitectura

### 📁 **Estructura del Proyecto**
```
├── 🤖 simple_professional_manager.py  # Sistema principal
├── 💼 professional_portfolio_manager.py  # Gestión de portfolio
├── 🛡️ advanced_risk_manager.py       # Gestión de riesgo
├── 📊 real_market_data_provider.py   # Datos reales de mercado
├── 🔔 smart_discord_notifier.py      # Notificaciones inteligentes
├── 🗄️ trading_database.py           # Base de datos
├── 📈 models/                        # Modelos TCN activos
│   ├── tcn_final_btcusdt.h5         # ✅ Modelo BTC (50,21)
│   ├── tcn_final_ethusdt.h5         # ✅ Modelo ETH (50,21)
│   └── tcn_final_bnbusdt.h5         # ✅ Modelo BNB (50,21)
├── 📦 archived_models/               # Modelos no utilizados
└── 🧪 test_*.py                     # Scripts de testing
```

### 🤖 **Modelos TCN**
- **Input:** `(50, 21)` - 50 timesteps, 21 features
- **Output:** `[SELL, HOLD, BUY]` - 3 clases
- **Features:** OHLCV, Returns, SMA, EMA, RSI, MACD, Bollinger, Volumen
- **Datos:** Tiempo real de Binance via `get_klines()`

## ⚙️ Configuración Avanzada

### 🛡️ **Gestión de Riesgo**
```python
# En advanced_risk_manager.py
max_position_percent = 15.0    # 15% máximo por posición  
max_daily_loss_percent = 10.0  # 10% pérdida máxima diaria
stop_loss_percent = 3.0        # 3% stop loss
take_profit_percent = 6.0      # 6% take profit
max_simultaneous_positions = 2 # 2 posiciones máximo
```

### 🔔 **Filtros de Discord**
```python
# En smart_discord_notifier.py
min_trade_value_usd = 12.0         # Solo trades > $12
min_pnl_percent_notify = 2.0       # Solo PnL > 2%
max_notifications_per_hour = 8     # Máximo 8/hora
suppress_similar_minutes = 10      # 10 min entre similares
```

## 🧪 Testing

### 🔍 **Scripts de Prueba**
```bash
# Verificar modelos TCN
python test_tcn_signals.py

# Analizar requerimientos de modelos
python analyze_model_requirements.py

# Probar carga de modelos
python test_model_loading.py

# Verificar portfolio manager
python -c "from professional_portfolio_manager import *; test_portfolio_manager()"
```

## 🚨 Seguridad

### 🔐 **Mejores Prácticas**
- ✅ **Nunca** commits credenciales
- ✅ Usar **testnet** para pruebas
- ✅ **Variables de entorno** para secrets
- ✅ **Límites de riesgo** configurados
- ✅ **Dry-run mode** disponible

### 🛡️ **Protecciones Implementadas**
- **Circuit breakers** para pérdidas
- **Rate limiting** para APIs
- **Validación** de inputs
- **Fallbacks** para fallos de modelo
- **Emergency stop** manual

## 🤝 Contribuir

### 📝 **Cómo Contribuir**
1. Fork el repositorio
2. Crear branch: `git checkout -b feature/nueva-caracteristica`
3. Commit: `git commit -m "Agregar nueva característica"`
4. Push: `git push origin feature/nueva-caracteristica`
5. Crear Pull Request

### 🐛 **Reportar Bugs**
- Usar **GitHub Issues**
- Incluir **logs completos**
- **Pasos para reproducir**
- **Configuración del sistema**

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## ⚠️ Disclaimer

**Este software es para fines educativos y de investigación. El trading automatizado conlleva riesgos financieros significativos. Usa bajo tu propio riesgo y nunca inviertas más de lo que puedes permitirte perder.**

---

## 📞 Soporte

- 📧 **Issues:** GitHub Issues
- 💬 **Discord:** [Tu servidor]
- 📖 **Docs:** [Wiki del proyecto]

---

**Hecho con ❤️ y TensorFlow para la comunidad de trading algorítmico**
