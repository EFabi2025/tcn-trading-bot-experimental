# 🍎 IMPLEMENTACIÓN MAC: FEATURE SETS OPTIMIZADOS

## 📋 RESUMEN DE CAMBIOS NECESARIOS

Este documento contiene todos los cambios necesarios para implementar la funcionalidad de feature sets optimizados en tu Mac, incluyendo soporte para argumentos de línea de comandos.

## ✅ CAMBIOS YA IMPLEMENTADOS

### 1. **Motor de Features (`centralized_features_engine2.py`)**
✅ **YA IMPLEMENTADO** - No requiere cambios adicionales
- Métodos `_get_optimized_crypto_features()` (25 features)
- Métodos `_get_ultra_optimized_features()` (15 features)
- Integrados en diccionario `self.feature_sets`

### 2. **Entrenador (`tcn_adaptative_trainer_v2.py`)**
✅ **YA IMPLEMENTADO** - No requiere cambios adicionales
- Parámetro `feature_set` en `TrainingConfig`
- Lista `available_feature_sets` con opciones
- Cálculo dinámico de features
- Validación de features por conjunto
- Nombre del modelo incluye feature set
- Configuración guardada en `config.json`

### 3. **Predictor (`tcn_ensemble_predictor.py`)**
✅ **YA IMPLEMENTADO** - No requiere cambios adicionales
- Diccionario `model_feature_sets` para almacenar feature sets
- Detección automática de feature set por modelo
- Cálculo de features específico por modelo

## 🔧 CAMBIOS PENDIENTES PARA MAC

### **1. AGREGAR SOPORTE DE ARGUMENTOS DE LÍNEA DE COMANDOS**

**Archivo:** `tcn_adaptative_trainer_v2.py`

**Ubicación:** Al final del archivo, reemplazar la sección `if __name__ == "__main__":`

**Código a agregar:**

```python
if __name__ == "__main__":
    import argparse

    # 🎯 CONFIGURAR ARGUMENTOS DE LÍNEA DE COMANDOS
    parser = argparse.ArgumentParser(description='🎯 Entrenador TCN Adaptativo con Feature Sets Optimizados')

    # Argumentos de feature sets
    parser.add_argument('--feature_set', type=str,
                       choices=['tcn_definitivo', 'optimized_crypto', 'ultra_optimized'],
                       help='Conjunto de features a usar (default: tcn_definitivo)')

    # Argumentos de configuración
    parser.add_argument('--timeframe', type=str, choices=['1m', '3m', '5m'],
                       help='Timeframe para entrenamiento')
    parser.add_argument('--pairs', nargs='+',
                       help='Pares de trading (ej: BTCUSDT ETHUSDT)')
    parser.add_argument('--prediction_horizon', type=int,
                       help='Horizonte de predicción en minutos')
    parser.add_argument('--lookback_window', type=int,
                       help='Ventana de análisis histórica')
    parser.add_argument('--training_days', type=int,
                       help='Días de datos para entrenamiento')
    parser.add_argument('--epochs', type=int,
                       help='Número de épocas de entrenamiento')
    parser.add_argument('--batch_size', type=int,
                       help='Tamaño de batch')

    # Argumento para modo no interactivo
    parser.add_argument('--non_interactive', action='store_true',
                       help='Ejecutar sin configuración interactiva')

    args = parser.parse_args()

    # 🎯 CREAR CONFIGURACIÓN DESDE ARGUMENTOS
    config = TrainingConfig()

    # Aplicar argumentos si están presentes
    if args.feature_set:
        config.feature_set = args.feature_set
    if args.timeframe:
        config.timeframe = args.timeframe
    if args.pairs:
        config.pairs = args.pairs
    if args.prediction_horizon:
        config.prediction_horizon = args.prediction_horizon
    if args.lookback_window:
        config.lookback_window = args.lookback_window
    if args.training_days:
        config.training_days = args.training_days
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size

    # 🎯 EJECUTAR ENTRENAMIENTO
    if args.non_interactive:
        # Modo no interactivo con argumentos
        print("🎯 ENTRENADOR TCN ADAPTATIVO - MODO NO INTERACTIVO")
        print("=" * 70)
        config.print_config()

        trainer = AdaptiveTCNTrainer(config)
        trainer.validate_configuration_consistency()

        print(f"\n🚀 INICIANDO ENTRENAMIENTO...")
        print(f"📊 Pares: {', '.join(trainer.pairs)}")
        print(f"⏰ Timeframe: {config.timeframe}")
        print(f"🔮 Horizonte: {config.prediction_horizon} minutos")
        print(f"📊 Ventana: {config.lookback_window} períodos")
        print(f"📅 Datos: {config.training_days} días")
        print(f"🎯 Épocas: {config.epochs}")
        print(f"🎯 Feature Set: {config.feature_set}")
        print("=" * 70)

        async def run_training():
            results = {}
            for symbol in trainer.pairs:
                print(f"\n🔥 Entrenando {symbol}...")

                if not trainer.validate_training_requirements(symbol):
                    print(f"❌ VALIDACIÓN FALLIDA para {symbol}. Saltando...")
                    results[symbol] = False
                    continue

                success = await trainer.train_adaptive_model(symbol)
                results[symbol] = success

            print(f"\n🎯 RESUMEN FINAL:")
            print("=" * 40)
            for symbol, success in results.items():
                status = "✅ ÉXITO" if success else "❌ FALLO"
                print(f"   {symbol}: {status}")

            successful = sum(results.values())
            print(f"\n🏆 Modelos entrenados exitosamente: {successful}/{len(results)}")

            if successful > 0:
                print(f"📁 Modelos guardados en: models/adaptive_<symbol>_<timeframe>_<config>/")
                print(f"🎯 ¡Listo para usar en trading!")
            else:
                print(f"❌ No se pudo entrenar ningún modelo. Revisa los errores arriba.")

        asyncio.run(run_training())
    else:
        # Modo interactivo (comportamiento original)
        asyncio.run(main())
```

### **2. AGREGAR OPCIÓN DE FEATURE SET EN CONFIGURACIÓN INTERACTIVA**

**Archivo:** `tcn_adaptative_trainer_v2.py`

**Ubicación:** En la función `configurar_interactivamente()`, después del paso de timeframe

**Código a agregar:**

```python
    # 2️⃣ FEATURE SET
    print(f"\n🎯 PASO 2: CONJUNTO DE FEATURES")
    print(f"¿Qué conjunto de features usar?")
    feature_sets = [
        ('tcn_definitivo', '88 features (completo)'),
        ('optimized_crypto', '25 features (optimizado)'),
        ('ultra_optimized', '15 features (ultra optimizado)')
    ]
    for i, (fs, desc) in enumerate(feature_sets, 1):
        print(f"  {i}. {fs} - {desc}")

    while True:
        respuesta = input(f"👉 Elige feature set [1-3] (default: 1): ").strip()
        if respuesta == '' or respuesta == '1':
            config.feature_set = 'tcn_definitivo'
            break
        elif respuesta == '2':
            config.feature_set = 'optimized_crypto'
            break
        elif respuesta == '3':
            config.feature_set = 'ultra_optimized'
            break
        else:
            print("❌ Opción inválida. Elige 1, 2 o 3")
```

**Nota:** Actualizar los números de los pasos siguientes (3→4, 4→5, etc.)

## 🚀 USO EN MAC

### **Entrenamiento con Argumentos de Línea de Comandos:**

```bash
# Entrenar con features optimizadas (25 features)
python3 tcn_adaptative_trainer_v2.py --feature_set optimized_crypto --non_interactive

# Entrenar con features ultra optimizadas (15 features)
python3 tcn_adaptative_trainer_v2.py --feature_set ultra_optimized --non_interactive

# Entrenar con features originales (88 features)
python3 tcn_adaptative_trainer_v2.py --feature_set tcn_definitivo --non_interactive

# Entrenar con configuración completa
python3 tcn_adaptative_trainer_v2.py \
  --feature_set optimized_crypto \
  --timeframe 1m \
  --pairs BTCUSDT ETHUSDT \
  --prediction_horizon 6 \
  --lookback_window 24 \
  --training_days 30 \
  --epochs 50 \
  --batch_size 64 \
  --non_interactive
```

### **Entrenamiento Interactivo:**

```bash
# Modo interactivo (incluye selección de feature set)
python3 tcn_adaptative_trainer_v2.py
```

## 📊 FEATURE SETS DISPONIBLES

### **`tcn_definitivo`** (88 features)
- Conjunto original completo
- Máxima información disponible
- Mayor tiempo de entrenamiento

### **`optimized_crypto`** (25 features) ⭐ **RECOMENDADO**
- Selección optimizada para trading de criptomonedas
- Balance entre predictibilidad y velocidad
- Reducción del 72% en features

### **`ultra_optimized`** (15 features)
- Las mejores de las mejores
- Máxima velocidad y eficiencia
- Reducción del 83% en features

## 🧪 VERIFICACIÓN

### **1. Probar argumentos:**
```bash
python3 tcn_adaptative_trainer_v2.py --help
```

### **2. Probar feature sets:**
```bash
python3 test_feature_sets_simple.py
```

## 📁 ARCHIVOS A MODIFICAR

1. **`tcn_adaptative_trainer_v2.py`**
   - Agregar soporte de argumentos de línea de comandos
   - Agregar opción de feature set en configuración interactiva

## ✅ ESTADO FINAL

- ✅ Motor de features: Implementado
- ✅ Entrenador: Implementado
- ✅ Predictor: Implementado
- 🔧 Argumentos de línea de comandos: Pendiente
- 🔧 Configuración interactiva: Pendiente

**Una vez implementados estos cambios, tendrás acceso completo a los feature sets optimizados tanto en modo interactivo como en línea de comandos.**
