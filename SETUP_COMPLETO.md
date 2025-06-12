# 🚀 SETUP COMPLETO - TRADING BOT PROFESIONAL

## 📋 **REQUISITOS PREVIOS**
- Python 3.8+
- Git
- Cuenta Binance (Testnet o Real)
- Discord Webhook (opcional)

## ⚙️ **INSTALACIÓN PASO A PASO**

### 1️⃣ **Clonar Repositorio**
```bash
git clone https://github.com/EFabi2025/tcn-trading-bot-experimental.git
cd tcn-trading-bot-experimental
```

### 2️⃣ **Crear Entorno Virtual**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

### 3️⃣ **Instalar Dependencias**
```bash
pip install -r requirements_professional_fixed.txt
```

### 4️⃣ **Configurar Variables de Entorno**
```bash
cp .env.example .env
nano .env  # Editar con tus credenciales
```

**Configuración requerida en `.env`:**
```env
# BINANCE API (REQUERIDO)
BINANCE_API_KEY=tu_api_key_aqui
BINANCE_SECRET_KEY=tu_secret_key_aqui
BINANCE_BASE_URL=https://testnet.binance.vision  # Para testnet
# BINANCE_BASE_URL=https://api.binance.com      # Para producción

# ENVIRONMENT
ENVIRONMENT=testnet  # o 'production'

# DISCORD (OPCIONAL)
DISCORD_WEBHOOK_URL=tu_webhook_url_aqui

# DATABASE
DATABASE_URL=sqlite:///trading_bot.db
```

### 5️⃣ **Verificar Instalación**
```bash
python test_installation.py
```

### 6️⃣ **Ejecutar Sistema de Trading**
```bash
python run_trading_manager.py
```

## 🎯 **ARCHIVOS PRINCIPALES**

### **Sistema Core:**
- `run_trading_manager.py` - Script principal de ejecución
- `simple_professional_manager.py` - Manager principal de trading
- `professional_portfolio_manager.py` - Gestión avanzada de portafolio
- `advanced_risk_manager.py` - Sistema de gestión de riesgo
- `trading_database.py` - Sistema de base de datos
- `smart_discord_notifier.py` - Notificaciones inteligentes

### **Modelos ML:**
- `production_model_*.h5` - Modelos TCN entrenados para cada par
- `tcn_production_ready.py` - Predictor principal TCN
- `binance_tcn_integration.py` - Integración completa Binance + TCN

### **Testing:**
- `test_professional_portfolio.py` - Tests del sistema de portafolio
- `test_trailing_stop_professional.py` - Tests del trailing stop

## 🔧 **CARACTERÍSTICAS PRINCIPALES**

### ✅ **Trading Profesional:**
- Sistema de múltiples posiciones por par
- Trailing stops profesionales adaptivos
- Stop loss y take profit automáticos
- Circuit breakers por pérdida diaria máxima

### ✅ **Gestión de Riesgo:**
- Límites de exposición configurables
- Máximo 2 posiciones simultáneas
- Validación de balance antes de órdenes
- Gestión automática de riesgo

### ✅ **Monitoreo en Tiempo Real:**
- Dashboard profesional en consola
- Reportes TCN automáticos cada 5 minutos
- Notificaciones Discord inteligentes
- Métricas de rendimiento en tiempo real

### ✅ **Portfolio Tracking:**
- PnL individual por posición
- Agrupación FIFO de órdenes
- Tracking de duración de posiciones
- Balance total actualizado de Binance

## ⚠️ **CONFIGURACIÓN DE SEGURIDAD**

### **Para TESTNET (Recomendado para pruebas):**
```env
BINANCE_BASE_URL=https://testnet.binance.vision
ENVIRONMENT=testnet
```

### **Para PRODUCCIÓN (Solo con experiencia):**
```env
BINANCE_BASE_URL=https://api.binance.com
ENVIRONMENT=production
```

## 🚨 **AVISOS IMPORTANTES**

1. **NUNCA** compartir API keys
2. **SIEMPRE** probar en testnet primero
3. **VERIFICAR** balance mínimo de 11 USDT
4. **REVISAR** configuración de riesgo antes de ejecutar
5. **MONITOREAR** el sistema durante ejecución

## 🛠️ **TROUBLESHOOTING**

### **Error: ModuleNotFoundError**
```bash
pip install -r requirements_professional_fixed.txt
```

### **Error: API key inválida**
```bash
# Verificar .env
cat .env
# Verificar conexión
python test_binance_connection.py
```

### **Error: Balance insuficiente**
```bash
# Verificar balance mínimo 11 USDT
python test_binance_limits.py
```

## 📊 **MONITOREO DEL SISTEMA**

El sistema mostrará información como:
```
🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥
🕐 21:23:32 | ⏱️ Uptime: 1.1min | 🎯 Trading Manager Professional
💼 PORTAFOLIO: $102.21 USDT
💰 USDT Libre: $12.72
📈 PnL No Realizado: $+0.20
🎯 Posiciones Activas: 6/5

📈 POSICIONES:
   🔴 BNBUSDT: 1 posición (-0.24%)
   🔴 BTCUSDT: 3 posiciones (-0.1% promedio)
   🟢 ETHUSDT: 2 posiciones (+0.9% promedio)
🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥
```

## 🎯 **RESULTADO ESPERADO**

Después de completar estos pasos, tendrás:
- ✅ Sistema de trading completamente funcional
- ✅ Gestión profesional de portafolio y riesgo
- ✅ Trailing stops y métricas en tiempo real
- ✅ Notificaciones Discord automáticas
- ✅ Sistema 100% portable entre PCs

## 📞 **SOPORTE**

Si encuentras problemas:
1. Verificar `.env` está configurado correctamente
2. Probar `test_installation.py`
3. Revisar logs en `trading_bot.db`
4. Asegurar balance mínimo en Binance

¡**Tu sistema de trading profesional estará listo para funcionar**! 🚀
