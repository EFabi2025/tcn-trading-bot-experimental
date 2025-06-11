#!/usr/bin/env python3
"""
🔍 Test de variables de entorno - Verificar lectura de API keys
"""

import os
from dotenv import load_dotenv

def test_env_loading():
    print("🔍 VERIFICANDO CARGA DE VARIABLES DE ENTORNO")
    print("=" * 50)
    
    # Cargar variables
    load_dotenv()
    
    # Obtener variables
    api_key = os.getenv('BINANCE_API_KEY', '').strip('"')
    secret_key = os.getenv('BINANCE_SECRET_KEY', '').strip('"') or os.getenv('BINANCE_API_SECRET', '').strip('"')
    environment = os.getenv('ENVIRONMENT', 'testnet').strip('"')
    
    print(f"🔑 API Key encontrada: {'✅ Sí' if api_key else '❌ No'}")
    if api_key:
        print(f"   📝 Primeros 8 chars: {api_key[:8]}...")
        print(f"   📏 Longitud total: {len(api_key)} caracteres")
    
    print(f"🔐 Secret Key encontrada: {'✅ Sí' if secret_key else '❌ No'}")
    if secret_key:
        print(f"   📝 Primeros 8 chars: {secret_key[:8]}...")
        print(f"   📏 Longitud total: {len(secret_key)} caracteres")
    
    print(f"🌍 Environment: {environment}")
    
    # Verificar valores válidos
    valid_keys = api_key and secret_key and \
                api_key != 'tu_api_key_de_binance_aqui' and \
                secret_key != 'tu_secret_key_de_binance_aqui' and \
                not api_key.startswith('tu_') and \
                not secret_key.startswith('tu_')
    
    print(f"\n✅ Configuración válida: {'Sí' if valid_keys else 'No'}")
    
    return valid_keys, api_key, secret_key

if __name__ == "__main__":
    test_env_loading() 