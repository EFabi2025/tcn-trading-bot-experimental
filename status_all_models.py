#!/usr/bin/env python3
"""
📊 STATUS DE TODOS LOS MODELOS DEFINITIVOS
Verificación completa del estado de entrenamiento
"""

import os
import requests
from datetime import datetime

def check_model_status(symbol):
    """🔍 Verificar estado de un modelo"""

    symbol_lower = symbol.lower()
    model_path = f"models/definitivo_{symbol_lower}/best_model.h5"
    scaler_path = f"models/definitivo_{symbol_lower}/scaler.pkl"
    features_path = f"models/definitivo_{symbol_lower}/feature_columns.pkl"

    status = {
        'symbol': symbol,
        'model_exists': False,
        'model_size_mb': 0,
        'model_date': None,
        'scaler_exists': False,
        'features_exists': False,
        'ready_for_prediction': False
    }

    # Verificar modelo
    if os.path.exists(model_path):
        stat_info = os.stat(model_path)
        status['model_exists'] = True
        status['model_size_mb'] = stat_info.st_size / (1024 * 1024)
        status['model_date'] = datetime.fromtimestamp(stat_info.st_mtime)

        # Verificar scaler y features
        status['scaler_exists'] = os.path.exists(scaler_path)
        status['features_exists'] = os.path.exists(features_path)

        # Listo para predicción si tiene todo
        status['ready_for_prediction'] = (
            status['model_exists'] and
            status['scaler_exists'] and
            status['features_exists']
        )

    return status

def get_current_prices():
    """💰 Obtener precios actuales"""
    try:
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        prices = {}

        for symbol in symbols:
            url = "https://api.binance.com/api/v3/ticker/price"
            params = {"symbol": symbol}

            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                prices[symbol] = float(data['price'])
            else:
                prices[symbol] = None

        return prices

    except Exception as e:
        print(f"❌ Error obteniendo precios: {e}")
        return {}

def print_status_table(statuses, prices):
    """📊 Imprimir tabla de estado"""

    print("📊 ESTADO DE MODELOS DEFINITIVOS")
    print("=" * 80)
    print(f"{'SÍMBOLO':<10} {'MODELO':<8} {'SCALER':<8} {'FEATURES':<10} {'LISTO':<8} {'PRECIO':<12}")
    print("-" * 80)

    for status in statuses:
        symbol = status['symbol']
        model_icon = "✅" if status['model_exists'] else "❌"
        scaler_icon = "✅" if status['scaler_exists'] else "❌"
        features_icon = "✅" if status['features_exists'] else "❌"
        ready_icon = "🟢" if status['ready_for_prediction'] else "🟡" if status['model_exists'] else "🔴"

        price = prices.get(symbol, 0)
        price_str = f"${price:,.2f}" if price else "N/A"

        print(f"{symbol:<10} {model_icon:<8} {scaler_icon:<8} {features_icon:<10} {ready_icon:<8} {price_str:<12}")

def print_detailed_status(statuses):
    """📋 Imprimir estado detallado"""

    print(f"\n📋 ESTADO DETALLADO:")
    print("-" * 50)

    for status in statuses:
        symbol = status['symbol']
        print(f"\n🎯 {symbol}:")

        if status['model_exists']:
            print(f"   ✅ Modelo: {status['model_size_mb']:.1f} MB")
            print(f"   📅 Fecha: {status['model_date'].strftime('%Y-%m-%d %H:%M:%S')}")

            if status['scaler_exists']:
                print(f"   ✅ Scaler: Disponible")
            else:
                print(f"   ⚠️ Scaler: FALTANTE")

            if status['features_exists']:
                print(f"   ✅ Features: Disponible")
            else:
                print(f"   ⚠️ Features: FALTANTE")

            if status['ready_for_prediction']:
                print(f"   🟢 Estado: LISTO PARA PREDICCIÓN")
            else:
                print(f"   🟡 Estado: NECESITA SCALER/FEATURES")
        else:
            print(f"   🔴 Estado: MODELO NO ENTRENADO")

def print_next_actions(statuses):
    """🎯 Imprimir próximas acciones"""

    print(f"\n🎯 PRÓXIMAS ACCIONES RECOMENDADAS:")
    print("-" * 50)

    actions = []

    for status in statuses:
        symbol = status['symbol']

        if status['model_exists'] and not status['ready_for_prediction']:
            actions.append(f"🔧 {symbol}: Recrear scaler y feature columns")
        elif not status['model_exists']:
            actions.append(f"🚀 {symbol}: Entrenar modelo definitivo")

    if not actions:
        actions.append("🎉 ¡Todos los modelos están listos!")

    for i, action in enumerate(actions, 1):
        print(f"   {i}. {action}")

def main():
    """🚀 Función principal"""

    print("📊 VERIFICACIÓN COMPLETA DE MODELOS DEFINITIVOS")
    print("=" * 80)

    # Verificar todos los modelos
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    statuses = [check_model_status(symbol) for symbol in symbols]

    # Obtener precios actuales
    print("💰 Obteniendo precios actuales...")
    prices = get_current_prices()

    # Mostrar tabla de estado
    print_status_table(statuses, prices)

    # Mostrar estado detallado
    print_detailed_status(statuses)

    # Mostrar próximas acciones
    print_next_actions(statuses)

    # Resumen final
    ready_count = sum(1 for s in statuses if s['ready_for_prediction'])
    trained_count = sum(1 for s in statuses if s['model_exists'])

    print(f"\n📈 RESUMEN FINAL:")
    print(f"   🎯 Modelos entrenados: {trained_count}/3")
    print(f"   🟢 Modelos listos: {ready_count}/3")
    print(f"   📊 Progreso total: {(ready_count/3)*100:.0f}%")

if __name__ == "__main__":
    main()
