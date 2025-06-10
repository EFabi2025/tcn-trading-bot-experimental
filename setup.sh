#!/bin/bash
# 🚀 CONFIGURACIÓN AUTOMÁTICA PROYECTO LIMPIO
echo "🚀 Configurando entorno limpio para TCN Anti-Bias Fixed..."

# Crear entorno virtual
echo "📦 Creando entorno virtual..."
python3 -m venv .venv
source .venv/bin/activate

# Actualizar pip
echo "⬆️ Actualizando pip..."
pip install --upgrade pip

# Instalar dependencias
echo "📚 Instalando dependencias..."
pip install -r requirements.txt

# Verificar instalación
echo "🧪 Verificando TensorFlow..."
python -c "import tensorflow as tf; print(f'✅ TensorFlow {tf.__version__} OK')"

# Verificar modelo
echo "🧪 Verificando modelo..."
python -c "
import tensorflow as tf
import os
if os.path.exists('models/tcn_anti_bias_fixed.h5'):
    model = tf.keras.models.load_model('models/tcn_anti_bias_fixed.h5')
    print(f'✅ Modelo cargado: {model.count_params():,} parámetros')
    print('🎉 ¡PROYECTO LISTO!')
else:
    print('❌ Modelo no encontrado')
"

echo ""
echo "🎉 CONFIGURACIÓN COMPLETADA"
echo "Para usar:"
echo "  cd BinanceBotClean_20250610_095103"
echo "  source .venv/bin/activate"
echo "  python scripts/test_tensorflow.py"
