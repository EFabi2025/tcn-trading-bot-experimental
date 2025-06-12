@echo off
REM 🪟 INSTALADOR AUTOMÁTICO PARA WINDOWS
REM Sistema de Trading Bot con Diversificación de Portafolio

echo ============================================================
echo 🪟 INSTALADOR TRADING BOT - WINDOWS
echo Sistema de Trading Bot con Diversificación de Portafolio
echo ============================================================
echo.

REM Verificar si Python está instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python no está instalado o no está en PATH
    echo.
    echo 💡 SOLUCIÓN:
    echo 1. Descargar Python 3.8+ desde: https://www.python.org/downloads/
    echo 2. Durante instalación, marcar "Add Python to PATH"
    echo 3. Reiniciar este script
    pause
    exit /b 1
)

echo ✅ Python detectado:
python --version
echo.

REM Verificar si Git está instalado
git --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Git no está instalado o no está en PATH
    echo.
    echo 💡 SOLUCIÓN:
    echo 1. Descargar Git desde: https://git-scm.com/download/win
    echo 2. Instalar con configuración por defecto
    echo 3. Reiniciar este script
    pause
    exit /b 1
)

echo ✅ Git detectado:
git --version
echo.

REM Crear entorno virtual
echo 📦 Creando entorno virtual...
if exist venv (
    echo ⚠️  Entorno virtual ya existe, eliminando...
    rmdir /s /q venv
)

python -m venv venv
if errorlevel 1 (
    echo ❌ Error creando entorno virtual
    pause
    exit /b 1
)

echo ✅ Entorno virtual creado
echo.

REM Activar entorno virtual
echo 🔄 Activando entorno virtual...
call venv\Scripts\activate.bat

REM Actualizar pip
echo 📦 Actualizando pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo ⚠️  Error actualizando pip, continuando...
)

REM Instalar dependencias
echo 📦 Instalando dependencias...
echo.

REM Instalar dependencias básicas primero
echo Instalando dependencias básicas...
pip install wheel setuptools
pip install numpy
pip install pandas
pip install requests
pip install python-dotenv

REM Instalar TensorFlow
echo Instalando TensorFlow...
pip install tensorflow

REM Instalar dependencias de trading
echo Instalando dependencias de trading...
pip install python-binance
pip install scikit-learn
pip install structlog
pip install pydantic
pip install sqlalchemy

REM Instalar desde requirements.txt si existe
if exist requirements.txt (
    echo Instalando desde requirements.txt...
    pip install -r requirements.txt
)

echo.
echo ✅ Dependencias instaladas
echo.

REM Verificar archivos necesarios
echo 📁 Verificando archivos del sistema...

if not exist "run_trading_manager.py" (
    echo ❌ Archivo faltante: run_trading_manager.py
    echo ⚠️  Asegúrate de estar en el directorio correcto del proyecto
    pause
    exit /b 1
)

if not exist "simple_professional_manager.py" (
    echo ❌ Archivo faltante: simple_professional_manager.py
    pause
    exit /b 1
)

if not exist "portfolio_diversification_manager.py" (
    echo ❌ Archivo faltante: portfolio_diversification_manager.py
    pause
    exit /b 1
)

if not exist "config\trading_config.py" (
    echo ❌ Archivo faltante: config\trading_config.py
    pause
    exit /b 1
)

echo ✅ Archivos del sistema verificados
echo.

REM Configurar archivo .env
echo 🔐 Configurando variables de entorno...

if not exist ".env" (
    if exist ".env.example" (
        echo Copiando .env.example a .env...
        copy .env.example .env
        echo.
        echo ⚠️  IMPORTANTE: Edita el archivo .env con tus credenciales:
        echo    - BINANCE_API_KEY=tu_api_key_aqui
        echo    - BINANCE_SECRET_KEY=tu_secret_key_aqui
        echo    - DISCORD_WEBHOOK_URL=tu_webhook_url_aqui
        echo.
    ) else (
        echo Creando archivo .env básico...
        echo # Configuración Trading Bot > .env
        echo BINANCE_API_KEY=tu_api_key_aqui >> .env
        echo BINANCE_SECRET_KEY=tu_secret_key_aqui >> .env
        echo ENVIRONMENT=production >> .env
        echo DISCORD_WEBHOOK_URL=tu_webhook_url_aqui >> .env
        echo.
        echo ⚠️  IMPORTANTE: Edita el archivo .env con tus credenciales reales
        echo.
    )
) else (
    echo ✅ Archivo .env ya existe
)

REM Crear directorios necesarios
echo 📁 Creando directorios...
if not exist "logs" mkdir logs
if not exist "data" mkdir data
if not exist "models" mkdir models

echo ✅ Directorios creados
echo.

REM Ejecutar verificación de compatibilidad
echo 🧪 Ejecutando verificación de compatibilidad...
python windows_compatibility_check.py
if errorlevel 1 (
    echo.
    echo ⚠️  Se detectaron problemas durante la verificación
    echo Revisa el reporte arriba y corrige los errores antes de continuar
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo 🎉 ¡INSTALACIÓN COMPLETADA EXITOSAMENTE!
echo ============================================================
echo.
echo 📋 PRÓXIMOS PASOS:
echo.
echo 1. 🔐 CONFIGURAR CREDENCIALES:
echo    - Editar archivo .env con tus API keys de Binance
echo    - Configurar Discord webhook (opcional)
echo.
echo 2. 🚀 EJECUTAR EL SISTEMA:
echo    - Doble clic en: start_trading.bat
echo    - O manualmente: python run_trading_manager.py
echo.
echo 3. 📊 MONITOREAR:
echo    - Revisar logs en carpeta: logs\
echo    - Verificar reportes de diversificación
echo.
echo ⚠️  IMPORTANTE:
echo - Nunca compartir tus API keys
echo - Empezar con cantidades pequeñas para probar
echo - Monitorear el sistema regularmente
echo.

REM Crear script de inicio
echo 📝 Creando script de inicio...
echo @echo off > start_trading.bat
echo cd /d "%~dp0" >> start_trading.bat
echo call venv\Scripts\activate.bat >> start_trading.bat
echo echo 🚀 Iniciando Trading Bot... >> start_trading.bat
echo python run_trading_manager.py >> start_trading.bat
echo pause >> start_trading.bat

echo ✅ Script de inicio creado: start_trading.bat
echo.

echo 🎯 CARACTERÍSTICAS INSTALADAS:
echo ✅ Sistema de diversificación de portafolio
echo ✅ Límites de concentración automáticos
echo ✅ Protección sin liquidación forzada
echo ✅ Integración completa con TCN
echo ✅ Reportes y alertas Discord
echo ✅ Optimización para Windows
echo.

pause
