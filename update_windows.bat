@echo off
REM 🔄 ACTUALIZADOR PARA WINDOWS
REM Sistema de Trading Bot con Diversificación de Portafolio

echo ============================================================
echo 🔄 ACTUALIZADOR TRADING BOT - WINDOWS
echo Sistema de Trading Bot con Diversificación de Portafolio
echo ============================================================
echo.

REM Verificar si estamos en un repositorio Git
if not exist ".git" (
    echo ❌ No se detectó repositorio Git
    echo ⚠️  Asegúrate de estar en el directorio correcto del proyecto
    pause
    exit /b 1
)

echo ✅ Repositorio Git detectado
echo.

REM Hacer backup de configuración local
echo 💾 Creando backup de configuración local...

if exist ".env" (
    copy .env .env.backup.%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
    echo ✅ Backup de .env creado
) else (
    echo ⚠️  Archivo .env no encontrado
)

if exist "config\trading_config.py" (
    copy "config\trading_config.py" "config\trading_config.py.backup.%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
    echo ✅ Backup de trading_config.py creado
)

echo.

REM Verificar estado del repositorio
echo 🔍 Verificando estado del repositorio...
git status --porcelain > temp_status.txt

for /f %%i in ("temp_status.txt") do set size=%%~zi
if %size% gtr 0 (
    echo ⚠️  Hay cambios locales no confirmados:
    git status --short
    echo.
    echo 💡 OPCIONES:
    echo 1. Confirmar cambios locales: git add . && git commit -m "Cambios locales"
    echo 2. Descartar cambios locales: git reset --hard HEAD
    echo 3. Guardar cambios temporalmente: git stash
    echo.
    set /p choice="¿Deseas continuar? (s/n): "
    if /i not "%choice%"=="s" (
        del temp_status.txt
        echo Actualización cancelada
        pause
        exit /b 1
    )
)

del temp_status.txt
echo.

REM Obtener últimos cambios
echo 📥 Obteniendo últimos cambios del repositorio...
git fetch origin
if errorlevel 1 (
    echo ❌ Error obteniendo cambios del repositorio
    echo Verifica tu conexión a internet y permisos
    pause
    exit /b 1
)

echo ✅ Cambios obtenidos exitosamente
echo.

REM Mostrar cambios disponibles
echo 📋 Cambios disponibles:
git log --oneline HEAD..origin/main
echo.

REM Aplicar cambios
echo 🔄 Aplicando cambios...
git pull origin main
if errorlevel 1 (
    echo ❌ Error aplicando cambios
    echo.
    echo 💡 POSIBLES SOLUCIONES:
    echo 1. Resolver conflictos manualmente
    echo 2. Usar: git reset --hard origin/main (CUIDADO: elimina cambios locales)
    echo 3. Contactar soporte técnico
    pause
    exit /b 1
)

echo ✅ Cambios aplicados exitosamente
echo.

REM Verificar archivos críticos
echo 📁 Verificando archivos críticos...

set "critical_files=run_trading_manager.py simple_professional_manager.py portfolio_diversification_manager.py config\trading_config.py"

for %%f in (%critical_files%) do (
    if exist "%%f" (
        echo ✅ %%f
    ) else (
        echo ❌ ARCHIVO FALTANTE: %%f
        set "missing_files=true"
    )
)

if defined missing_files (
    echo.
    echo ❌ ARCHIVOS CRÍTICOS FALTANTES
    echo La actualización puede estar incompleta
    pause
    exit /b 1
)

echo.

REM Activar entorno virtual si existe
if exist "venv\Scripts\activate.bat" (
    echo 🔄 Activando entorno virtual...
    call venv\Scripts\activate.bat

    REM Actualizar dependencias
    echo 📦 Actualizando dependencias...
    pip install --upgrade pip

    if exist "requirements.txt" (
        pip install -r requirements.txt --upgrade
        echo ✅ Dependencias actualizadas
    ) else (
        echo ⚠️  requirements.txt no encontrado, instalando dependencias básicas...
        pip install --upgrade tensorflow numpy pandas scikit-learn python-binance python-dotenv requests structlog pydantic sqlalchemy
    )

    echo.
) else (
    echo ⚠️  Entorno virtual no encontrado
    echo 💡 Ejecuta install_windows.bat para configurar el entorno
    echo.
)

REM Ejecutar verificación de compatibilidad
if exist "windows_compatibility_check.py" (
    echo 🧪 Ejecutando verificación de compatibilidad...
    python windows_compatibility_check.py
    if errorlevel 1 (
        echo.
        echo ⚠️  Se detectaron problemas durante la verificación
        echo Revisa el reporte y corrige los errores antes de ejecutar el sistema
        echo.
    ) else (
        echo ✅ Verificación de compatibilidad exitosa
    )
) else (
    echo ⚠️  Verificador de compatibilidad no encontrado
)

echo.

REM Restaurar configuración local si es necesario
if exist ".env.backup.*" (
    echo 🔧 ¿Deseas restaurar tu configuración local (.env)?
    set /p restore_env="(s/n): "
    if /i "%restore_env%"=="s" (
        for /f %%f in ('dir /b .env.backup.* 2^>nul ^| sort /r') do (
            copy "%%f" .env
            echo ✅ Configuración .env restaurada desde %%f
            goto :env_restored
        )
        :env_restored
    )
)

echo.
echo ============================================================
echo 🎉 ¡ACTUALIZACIÓN COMPLETADA EXITOSAMENTE!
echo ============================================================
echo.

REM Mostrar resumen de cambios
echo 📊 RESUMEN DE LA ACTUALIZACIÓN:
echo.

REM Mostrar últimos commits
echo 📋 ÚLTIMOS CAMBIOS:
git log --oneline -5
echo.

echo 🎯 CARACTERÍSTICAS ACTUALIZADAS:
echo ✅ Sistema de diversificación de portafolio
echo ✅ Límites de concentración automáticos
echo ✅ Protección sin liquidación forzada
echo ✅ Integración completa con TCN
echo ✅ Reportes y alertas Discord
echo ✅ Optimización para Windows
echo.

echo 📋 PRÓXIMOS PASOS:
echo.
echo 1. 🔐 VERIFICAR CONFIGURACIÓN:
echo    - Revisar archivo .env con tus credenciales
echo    - Verificar config\trading_config.py
echo.
echo 2. 🧪 PROBAR SISTEMA:
echo    - Ejecutar: python windows_compatibility_check.py
echo    - Verificar que no hay errores
echo.
echo 3. 🚀 EJECUTAR SISTEMA:
echo    - Doble clic en: start_trading.bat
echo    - O manualmente: python run_trading_manager.py
echo.
echo 4. 📊 MONITOREAR:
echo    - Revisar logs en carpeta: logs\
echo    - Verificar reportes de diversificación
echo.

echo ⚠️  IMPORTANTE:
echo - Revisar que tus API keys sigan siendo válidas
echo - Verificar que el sistema funcione correctamente antes de trading real
echo - Monitorear el nuevo sistema de diversificación
echo.

pause
