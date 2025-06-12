#!/usr/bin/env python3
"""
🧪 Test de carga de modelos TCN con compatibilidad robusta
"""

import asyncio
from simple_professional_manager import SimpleProfessionalTradingManager

async def test_model_loading():
    """Probar la carga de modelos TCN"""
    print("🧪 Iniciando test de carga de modelos...")

    try:
        manager = SimpleProfessionalTradingManager()
        await manager._initialize_tcn_models()

        print(f"\n📊 RESULTADOS:")
        print(f"   Modelos activos: {manager.tcn_models_active}")
        print(f"   Modelos disponibles: {list(manager.tcn_models.keys())}")

        loaded_models = {k: v for k, v in manager.tcn_models.items() if v is not None}
        print(f"   Modelos cargados exitosamente: {len(loaded_models)}")

        for pair, model_info in loaded_models.items():
            print(f"   ✅ {pair}: {model_info['params']:,} parámetros")

    except Exception as e:
        print(f"❌ Error en test: {e}")

if __name__ == "__main__":
    asyncio.run(test_model_loading())
