#!/bin/bash

# 🚀 Professional Trading Bot - Script de Instalación Automática
# ==============================================================

set -e  # Salir si hay errores

echo "🚀 PROFESSIONAL TRADING BOT - INSTALACIÓN AUTOMÁTICA"
echo "====================================================="

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para imprimir en colores
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Detectar OS
OS=""
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
elif [[ "$OSTYPE" == "msys" ]]; then
    OS="Windows"
else
    OS="Unknown"
fi

print_info "Sistema operativo detectado: $OS"

# Verificar Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 no está instalado. Por favor instala Python 3.10+ primero."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
print_success "Python $PYTHON_VERSION encontrado"

# Verificar pip
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 no está instalado. Por favor instala pip primero."
    exit 1
fi

print_success "pip3 encontrado"

# Crear entorno virtual
print_info "Creando entorno virtual..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    print_success "Entorno virtual creado en .venv"
else
    print_warning "Entorno virtual .venv ya existe"
fi

# Activar entorno virtual
print_info "Activando entorno virtual..."
source .venv/bin/activate || {
    print_error "Error activando entorno virtual"
    exit 1
}
print_success "Entorno virtual activado"

# Actualizar pip
print_info "Actualizando pip..."
pip install --upgrade pip

# Instalar dependencias según OS
print_info "Instalando dependencias..."

if [[ "$OS" == "macOS" ]]; then
    print_info "Instalación optimizada para macOS (Apple Silicon compatible)"
    # Instalar TensorFlow para macOS
    pip install tensorflow-macos==2.15.0
    pip install tensorflow-metal
    print_success "TensorFlow para Apple Silicon instalado"
else
    print_info "Instalación para $OS"
    # Instalar TensorFlow estándar
    pip install tensorflow==2.15.0
    print_success "TensorFlow estándar instalado"
fi

# Instalar el resto de dependencias
print_info "Instalando dependencias principales..."
pip install -r requirements.txt
print_success "Todas las dependencias instaladas"

# Crear archivo de configuración
print_info "Configurando archivo de entorno..."
if [ ! -f ".env" ]; then
    cp config_example.env .env
    print_success "Archivo .env creado desde config_example.env"
    print_warning "IMPORTANTE: Edita .env con tus credenciales de Binance antes de usar"
else
    print_warning "Archivo .env ya existe - no se sobrescribió"
fi

# Verificar modelos TCN
print_info "Verificando modelos TCN..."
MODELS_FOUND=0
for model in models/tcn_final_btcusdt.h5 models/tcn_final_ethusdt.h5 models/tcn_final_bnbusdt.h5; do
    if [ -f "$model" ]; then
        MODELS_FOUND=$((MODELS_FOUND + 1))
        print_success "Modelo encontrado: $model"
    else
        print_warning "Modelo no encontrado: $model"
    fi
done

if [ $MODELS_FOUND -eq 3 ]; then
    print_success "Todos los modelos TCN están disponibles"
else
    print_warning "Faltan modelos TCN - algunas funciones pueden no estar disponibles"
fi

# Crear directorio de logs
print_info "Creando directorio de logs..."
mkdir -p logs
print_success "Directorio logs/ creado"

# Test de instalación
print_info "Ejecutando test de instalación..."
if python3 -c "import tensorflow as tf; import numpy as np; import pandas as pd; import binance; print('✅ Importaciones principales OK')"; then
    print_success "Test de importaciones OK"
else
    print_error "Falló el test de importaciones"
    exit 1
fi

# Información final
echo ""
print_success "🎉 INSTALACIÓN COMPLETADA EXITOSAMENTE"
echo "======================================"
print_info "Próximos pasos:"
echo "1. 📝 Edita .env con tus credenciales de Binance:"
echo "   - BINANCE_API_KEY=tu_api_key"
echo "   - BINANCE_SECRET_KEY=tu_secret_key"
echo "   - DISCORD_WEBHOOK_URL=tu_webhook (opcional)"
echo ""
echo "2. 🧪 Para empezar con testnet (recomendado):"
echo "   - Usa BINANCE_BASE_URL=https://testnet.binance.vision"
echo "   - Obtén API keys del testnet en: https://testnet.binance.vision/"
echo ""
echo "3. 🚀 Ejecutar el bot:"
echo "   source .venv/bin/activate"
echo "   python run_trading_manager.py"
echo ""
echo "4. 🧪 Ejecutar tests:"
echo "   python test_tcn_signals.py"
echo ""
print_warning "⚠️  IMPORTANTE:"
echo "   - SIEMPRE prueba en testnet primero"
echo "   - NUNCA compartas tus API keys"
echo "   - Configura límites de riesgo apropiados"
echo "   - Monitorea constantemente el bot"
echo ""
print_info "📖 Documentación completa en README.md"
print_info "🐛 Reportar problemas en GitHub Issues"
echo ""
print_success "¡Listo para trading profesional con TCN! 🤖📈" 