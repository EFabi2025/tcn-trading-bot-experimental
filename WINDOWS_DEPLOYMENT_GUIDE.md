# 🪟 GUÍA DE DESPLIEGUE EN WINDOWS
## Sistema de Trading Bot con Diversificación de Portafolio

### 🚨 REQUISITOS PREVIOS

#### 1. Software Base
```powershell
# Verificar Python 3.8+ instalado
python --version

# Verificar Git instalado
git --version

# Instalar Visual Studio Build Tools (para compilar dependencias)
# Descargar desde: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

#### 2. Variables de Entorno Windows
```powershell
# Crear archivo .env en la raíz del proyecto
# NUNCA subir este archivo a Git
BINANCE_API_KEY=tu_api_key_aqui
BINANCE_SECRET_KEY=tu_secret_key_aqui
ENVIRONMENT=production
DISCORD_WEBHOOK_URL=tu_webhook_url_aqui
```

### 📥 ACTUALIZACIÓN DEL SISTEMA

#### 1. Sincronizar Cambios desde Repositorio
```powershell
# Navegar al directorio del proyecto
cd C:\ruta\a\tu\proyecto

# Hacer backup de configuración local
copy .env .env.backup
copy config\trading_config.py config\trading_config.py.backup

# Obtener últimos cambios
git fetch origin
git pull origin main

# Restaurar configuración local si es necesario
copy .env.backup .env
```

#### 2. Verificar Archivos Nuevos
```powershell
# Verificar que estos archivos existan:
dir portfolio_diversification_manager.py
dir config\trading_config.py
dir simple_professional_manager.py
```

### 🔧 INSTALACIÓN DE DEPENDENCIAS

#### 1. Entorno Virtual Windows
```powershell
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual (PowerShell)
.\venv\Scripts\Activate.ps1

# O activar en CMD
.\venv\Scripts\activate.bat

# Verificar activación
where python
# Debe mostrar: C:\ruta\proyecto\venv\Scripts\python.exe
```

#### 2. Instalar Dependencias
```powershell
# Actualizar pip
python -m pip install --upgrade pip

# Instalar dependencias base
pip install -r requirements.txt

# Dependencias específicas para Windows
pip install --upgrade tensorflow
pip install --upgrade numpy
pip install --upgrade pandas
pip install --upgrade scikit-learn
```

### 🎯 CONFIGURACIÓN ESPECÍFICA WINDOWS

#### 1. TensorFlow Optimización
```python
# En config/trading_config.py - Verificar configuración Windows
TENSORFLOW_CONFIG = {
    'use_gpu': False,  # Cambiar a True si tienes GPU NVIDIA
    'memory_growth': True,
    'log_device_placement': False,
    'inter_op_parallelism_threads': 0,  # Auto-detect
    'intra_op_parallelism_threads': 0   # Auto-detect
}
```

#### 2. Rutas de Archivos Windows
```python
# El sistema ya usa os.path.join() - Compatible con Windows
# Verificar en portfolio_diversification_manager.py
import os
log_path = os.path.join('logs', 'diversification.log')  # ✅ Compatible
```

#### 3. Configuración de Logging
```python
# Windows requiere configuración específica para logs
import logging
import os

# Crear directorio logs si no existe
os.makedirs('logs', exist_ok=True)

# Configuración compatible Windows
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
```

### 🧪 TESTING EN WINDOWS

#### 1. Verificar Conexión Binance
```powershell
# Crear script de prueba
python -c "
from binance.client import Client
import os
from dotenv import load_dotenv

load_dotenv()
client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_SECRET_KEY'))
print('Conexión exitosa:', client.get_account_status())
"
```

#### 2. Test del Sistema de Diversificación
```powershell
# Ejecutar test básico
python -c "
from portfolio_diversification_manager import PortfolioDiversificationManager
from config.trading_config import DIVERSIFICATION_CONFIG

manager = PortfolioDiversificationManager(DIVERSIFICATION_CONFIG)
print('Sistema de diversificación inicializado correctamente')
print('Configuración:', manager.config)
"
```

### 🚀 EJECUCIÓN EN WINDOWS

#### 1. Script de Inicio Windows
```powershell
# Crear start_trading.bat
@echo off
cd /d "C:\ruta\a\tu\proyecto"
call venv\Scripts\activate.bat
python run_trading_manager.py
pause
```

#### 2. Ejecución Manual
```powershell
# Activar entorno
.\venv\Scripts\Activate.ps1

# Ejecutar sistema
python run_trading_manager.py
```

#### 3. Ejecución como Servicio (Opcional)
```powershell
# Instalar NSSM (Non-Sucking Service Manager)
# Descargar desde: https://nssm.cc/download

# Crear servicio
nssm install TradingBot "C:\ruta\proyecto\venv\Scripts\python.exe"
nssm set TradingBot Arguments "C:\ruta\proyecto\run_trading_manager.py"
nssm set TradingBot AppDirectory "C:\ruta\proyecto"
nssm start TradingBot
```

### 🔍 TROUBLESHOOTING WINDOWS

#### 1. Errores Comunes
```powershell
# Error: Microsoft Visual C++ 14.0 is required
# Solución: Instalar Visual Studio Build Tools

# Error: Permission denied
# Solución: Ejecutar PowerShell como Administrador

# Error: SSL Certificate
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt
```

#### 2. Verificación de Funcionamiento
```powershell
# Verificar procesos
tasklist | findstr python

# Verificar logs
type logs\trading.log | more

# Verificar conexión de red
netstat -an | findstr :443
```

### 📊 MONITOREO EN WINDOWS

#### 1. Task Manager
- Monitorear uso de CPU/RAM del proceso Python
- Verificar que no haya memory leaks

#### 2. Logs Específicos
```powershell
# Ver logs en tiempo real
Get-Content logs\trading.log -Wait -Tail 50

# Buscar errores
Select-String -Path logs\trading.log -Pattern "ERROR"
```

### 🔐 SEGURIDAD WINDOWS

#### 1. Firewall
```powershell
# Permitir Python a través del firewall
netsh advfirewall firewall add rule name="Python Trading Bot" dir=out action=allow program="C:\ruta\proyecto\venv\Scripts\python.exe"
```

#### 2. Antivirus
- Agregar carpeta del proyecto a exclusiones
- Agregar python.exe a exclusiones

### 📋 CHECKLIST DE DESPLIEGUE

- [ ] Python 3.8+ instalado
- [ ] Git instalado y configurado
- [ ] Visual Studio Build Tools instalado
- [ ] Repositorio clonado/actualizado
- [ ] Archivo .env configurado
- [ ] Entorno virtual creado y activado
- [ ] Dependencias instaladas
- [ ] Test de conexión Binance exitoso
- [ ] Test de diversificación exitoso
- [ ] Sistema ejecutándose correctamente
- [ ] Logs generándose correctamente
- [ ] Monitoreo configurado

### 🆘 SOPORTE

Si encuentras problemas específicos de Windows:

1. **Verificar versión Python**: `python --version`
2. **Verificar dependencias**: `pip list`
3. **Revisar logs**: `type logs\trading.log`
4. **Verificar configuración**: Revisar .env y trading_config.py

### 🎯 DIFERENCIAS CLAVE WINDOWS vs macOS

| Aspecto | Windows | macOS |
|---------|---------|-------|
| Activación venv | `.\venv\Scripts\Activate.ps1` | `source venv/bin/activate` |
| Separador rutas | `\` (automático con os.path.join) | `/` |
| TensorFlow | CPU por defecto | Metal por defecto |
| Logs encoding | UTF-8 explícito | UTF-8 automático |
| Servicios | NSSM o Task Scheduler | launchd |

¡El sistema está diseñado para ser completamente compatible con Windows! 🎉
