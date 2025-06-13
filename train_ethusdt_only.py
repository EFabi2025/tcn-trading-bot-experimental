#!/usr/bin/env python3
"""
🎯 ENTRENADOR ETHUSDT DEFINITIVO
Entrena solo el modelo de ETHUSDT con técnicas anti-sesgo
"""

import asyncio
from tcn_definitivo_trainer import DefinitiveTCNTrainer

async def main():
    """🚀 Entrenar solo ETHUSDT"""

    print("🎯 ENTRENAMIENTO DEFINITIVO - ETHUSDT ÚNICAMENTE")
    print("=" * 70)

    try:
        # Crear trainer
        trainer = DefinitiveTCNTrainer()

        # Entrenar solo ETHUSDT
        print("🚀 Iniciando entrenamiento de ETHUSDT...")
        success = await trainer.train_definitive_model("ETHUSDT")

        if success:
            print(f"\n✅ ETHUSDT entrenado exitosamente")
        else:
            print(f"\n❌ Error entrenando ETHUSDT")

    except Exception as e:
        print(f"❌ Error general: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
