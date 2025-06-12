# 🔒 GUÍA DE SEGURIDAD - TRADING BOT

## 🚨 PROTOCOLO DE RESPUESTA A INCIDENTES

### ¿Qué hacer si expones credenciales accidentalmente?

1. **ACCIÓN INMEDIATA (< 5 minutos)**
   ```bash
   # ❗ REVOCAR credenciales comprometidas INMEDIATAMENTE
   # - Binance: https://www.binance.com/en/my/settings/api-management
   # - Discord: Regenerar webhook URL
   # - Cualquier otro servicio afectado
   ```

2. **VERIFICACIÓN DE ACTIVIDAD SOSPECHOSA**
   - Revisar logs de trading recientes
   - Verificar transacciones no autorizadas
   - Monitorear balances de cuenta

3. **LIMPIEZA DEL HISTORIAL GIT**
   ```bash
   # Hacer backup
   cp -r proyecto proyecto-backup-$(date +%Y%m%d_%H%M%S)

   # Instalar herramientas
   pip install git-filter-repo

   # Crear archivo con credenciales a eliminar
   echo "tu_credencial_comprometida==>***REDACTED***" > secrets_to_remove.txt

   # Ejecutar limpieza
   git filter-repo --replace-text secrets_to_remove.txt --force

   # Reconectar remoto y forzar push
   git remote add origin tu_repo_url
   git push --force origin main

   # Limpiar archivo temporal
   rm secrets_to_remove.txt
   ```

## 🛡️ PREVENCIÓN DE INCIDENTES

### Pre-commit Hooks (YA CONFIGURADOS)
```bash
# Los hooks están configurados para detectar:
# ✅ Archivos .env en staging area
# ✅ Posibles API keys y secrets
# ✅ Credenciales hardcodeadas
# ✅ Archivos grandes accidentales
```

### Estructura de Archivos Segura
```
proyecto/
├── .env                    # ❌ NUNCA versionar
├── .env.example           # ✅ SÍ versionar (solo placeholders)
├── .gitignore             # ✅ Configurado para seguridad
├── .pre-commit-config.yaml # ✅ Hooks de seguridad
└── .secrets.baseline      # ✅ Baseline de secrets conocidos
```

### Variables de Entorno Seguras
```bash
# ✅ CORRECTO - .env (NO versionado)
BINANCE_API_KEY=tu_clave_real
BINANCE_SECRET_KEY=tu_secreto_real

# ✅ CORRECTO - .env.example (SÍ versionado)
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here
```

## 🔐 MEJORES PRÁCTICAS

### 1. Manejo de Credenciales
- ❌ **NUNCA** hardcodear credenciales en código
- ❌ **NUNCA** commitear archivos .env
- ❌ **NUNCA** copiar/pegar credenciales en chat/email
- ✅ **SIEMPRE** usar variables de entorno
- ✅ **SIEMPRE** rotar credenciales regularmente
- ✅ **SIEMPRE** usar principio de menor privilegio

### 2. Configuración de APIs
```python
# ✅ CORRECTO
import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    binance_api_key: str
    binance_secret: str

    class Config:
        env_file = ".env"

settings = Settings()
client = Client(settings.binance_api_key, settings.binance_secret)
```

### 3. Logging Seguro
```python
# ❌ PELIGROSO
logger.info(f"API Key: {api_key}")

# ✅ SEGURO
logger.info("API connection established", extra={
    'api_key_hash': hashlib.md5(api_key.encode()).hexdigest()[:8]
})
```

## 🧪 TESTING DE SEGURIDAD

### Verificar configuración actual
```bash
# Verificar que no hay credenciales en el historial
git rev-list --all | xargs git grep -l "tu_credencial" || echo "✅ Limpio"

# Probar pre-commit hooks
pre-commit run --all-files

# Verificar .gitignore
git check-ignore .env && echo "✅ .env ignorado correctamente"
```

### Simulacro de Incidente
```bash
# 1. Crear archivo .env de prueba
echo "TEST_API_KEY=fake_key_123" > test.env

# 2. Intentar agregarlo a Git
git add test.env

# 3. Verificar que los hooks lo bloquean
git commit -m "test" # Debería fallar

# 4. Limpiar
rm test.env
```

## 📊 MONITOREO CONTINUO

### Herramientas Recomendadas
- **detect-secrets**: Análisis estático de credenciales
- **pre-commit**: Hooks antes de commits
- **git-filter-repo**: Limpieza de historial
- **GitHub Security Advisories**: Alertas automáticas

### Checklist de Seguridad Semanal
- [ ] Revisar logs de trading por actividad sospechosa
- [ ] Verificar que credenciales no están en nuevo código
- [ ] Rotar credenciales si es necesario
- [ ] Actualizar dependencias con vulnerabilidades
- [ ] Verificar configuración de pre-commit hooks

## 🚨 CONTACTOS DE EMERGENCIA

### En caso de incidente crítico:
1. **Revocar credenciales** (Prioridad #1)
2. **Notificar al equipo** inmediatamente
3. **Documentar el incidente** para evitar repetición
4. **Implementar mejoras** en proceso de seguridad

---

**🎯 RECUERDA: La seguridad es responsabilidad de todos. Un error puede costar miles de dólares en un sistema de trading.**
