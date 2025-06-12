#!/bin/bash
# Script de inicio para Binance TCN Trading System

echo "🚀 INICIANDO BINANCE TCN TRADING SYSTEM"
echo "========================================"

# Verificar archivo .env
if [ ! -f .env ]; then
    echo "❌ Archivo .env no encontrado"
    echo "📝 Copiar .env.example a .env y configurar API keys"
    exit 1
fi

# Cargar variables de entorno
source .env

# Verificar API keys
if [ -z "$BINANCE_API_KEY" ] || [ "$BINANCE_API_KEY" = "your_testnet_api_key_here" ]; then
    echo "❌ BINANCE_API_KEY no configurado"
    echo "📝 Editar .env con tu API key real"
    exit 1
fi

if [ -z "$BINANCE_API_SECRET" ] || [ "$BINANCE_API_SECRET" = "your_testnet_secret_here" ]; then
    echo "❌ BINANCE_API_SECRET no configurado"
    echo "📝 Editar .env con tu secret real"
    exit 1
fi

echo "✅ Configuración verificada"
echo "🔄 Iniciando sistema de trading..."

# Exportar variables y ejecutar
export BINANCE_API_KEY
export BINANCE_API_SECRET
python binance_tcn_integration.py
