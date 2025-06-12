#!/usr/bin/env python3
"""
🔍 AUDITORÍA COMPLETA DE VALORES HARDCODEADOS
Sistema de Trading Profesional - Detección de Variables Críticas
"""

import os
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class HardcodedIssue:
    file: str
    line: int
    code: str
    issue_type: str
    severity: str
    description: str
    recommendation: str

class HardcodedValuesAuditor:
    def __init__(self):
        self.issues: List[HardcodedIssue] = []
        self.critical_files = [
            'simple_professional_manager.py',
            'advanced_risk_manager.py',
            'enhanced_real_predictor.py',
            'professional_portfolio_manager.py'
        ]

    def audit_system(self):
        """🔍 Auditar todo el sistema"""
        print("🔍 INICIANDO AUDITORÍA DE VALORES HARDCODEADOS")
        print("=" * 60)

        for file_path in self.critical_files:
            if os.path.exists(file_path):
                print(f"\n📁 Auditando: {file_path}")
                self._audit_file(file_path)
            else:
                print(f"⚠️ Archivo no encontrado: {file_path}")

        self._generate_report()

    def _audit_file(self, file_path: str):
        """🔍 Auditar archivo específico"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                self._check_line(file_path, line_num, line.strip())

        except Exception as e:
            print(f"❌ Error leyendo {file_path}: {e}")

    def _check_line(self, file_path: str, line_num: int, line: str):
        """🔍 Verificar línea específica"""

        # 1. CONFIANZA HARDCODEADA
        if re.search(r'confidence\s*[<>=]+\s*0\.[7-9]', line):
            self.issues.append(HardcodedIssue(
                file=file_path,
                line=line_num,
                code=line,
                issue_type="CONFIDENCE_THRESHOLD",
                severity="HIGH",
                description="Umbral de confianza hardcodeado",
                recommendation="Mover a configuración externa (.env o config.py)"
            ))

        # 2. PORCENTAJES DE RIESGO HARDCODEADOS
        risk_patterns = [
            r'stop_loss_percent.*=.*[0-9]+\.[0-9]+',
            r'take_profit_percent.*=.*[0-9]+\.[0-9]+',
            r'max_position_size_percent.*=.*[0-9]+\.[0-9]+',
            r'max_daily_loss_percent.*=.*[0-9]+\.[0-9]+',
            r'trailing_stop_percent.*=.*[0-9]+\.[0-9]+'
        ]

        for pattern in risk_patterns:
            if re.search(pattern, line):
                self.issues.append(HardcodedIssue(
                    file=file_path,
                    line=line_num,
                    code=line,
                    issue_type="RISK_PARAMETER",
                    severity="CRITICAL",
                    description="Parámetro de riesgo hardcodeado",
                    recommendation="CRÍTICO: Mover a configuración de riesgo externa"
                ))

        # 3. VALORES MONETARIOS HARDCODEADOS
        money_patterns = [
            r'min_position_value.*=.*[0-9]+\.[0-9]+',
            r'current_balance.*=.*[0-9]+\.[0-9]+',
            r'start_balance.*=.*[0-9]+\.[0-9]+'
        ]

        for pattern in money_patterns:
            if re.search(pattern, line):
                self.issues.append(HardcodedIssue(
                    file=file_path,
                    line=line_num,
                    code=line,
                    issue_type="MONETARY_VALUE",
                    severity="CRITICAL",
                    description="Valor monetario hardcodeado",
                    recommendation="CRÍTICO: Obtener de API de Binance o configuración"
                ))

        # 4. LÍMITES DE POSICIONES HARDCODEADOS
        if re.search(r'max_concurrent_positions.*=.*[0-9]+', line):
            self.issues.append(HardcodedIssue(
                file=file_path,
                line=line_num,
                code=line,
                issue_type="POSITION_LIMIT",
                severity="MEDIUM",
                description="Límite de posiciones hardcodeado",
                recommendation="Mover a configuración de riesgo"
            ))

        # 5. PRECIOS DE EJEMPLO HARDCODEADOS
        if re.search(r'[4-8][0-9]{4}', line) and ('price' in line.lower() or 'btc' in line.lower()):
            self.issues.append(HardcodedIssue(
                file=file_path,
                line=line_num,
                code=line,
                issue_type="EXAMPLE_PRICE",
                severity="LOW",
                description="Precio de ejemplo hardcodeado",
                recommendation="Usar datos reales de API o marcar claramente como ejemplo"
            ))

        # 6. TIMEOUTS Y DELAYS HARDCODEADOS
        if re.search(r'sleep\([0-9]+\)|timeout.*=.*[0-9]+', line):
            self.issues.append(HardcodedIssue(
                file=file_path,
                line=line_num,
                code=line,
                issue_type="TIMING",
                severity="MEDIUM",
                description="Timeout o delay hardcodeado",
                recommendation="Mover a configuración de sistema"
            ))

        # 7. FILTROS DE VOLUMEN HARDCODEADOS
        if re.search(r'volume.*[><=].*[0-9]{5,}', line):
            self.issues.append(HardcodedIssue(
                file=file_path,
                line=line_num,
                code=line,
                issue_type="VOLUME_FILTER",
                severity="MEDIUM",
                description="Filtro de volumen hardcodeado",
                recommendation="Calcular dinámicamente basado en datos históricos"
            ))

    def _generate_report(self):
        """📊 Generar reporte de auditoría"""
        print("\n" + "=" * 60)
        print("📊 REPORTE DE AUDITORÍA DE VALORES HARDCODEADOS")
        print("=" * 60)

        if not self.issues:
            print("✅ ¡EXCELENTE! No se encontraron valores hardcodeados problemáticos")
            return

        # Agrupar por severidad
        critical = [i for i in self.issues if i.severity == "CRITICAL"]
        high = [i for i in self.issues if i.severity == "HIGH"]
        medium = [i for i in self.issues if i.severity == "MEDIUM"]
        low = [i for i in self.issues if i.severity == "LOW"]

        print(f"🚨 CRÍTICOS: {len(critical)}")
        print(f"⚠️ ALTOS: {len(high)}")
        print(f"🔶 MEDIOS: {len(medium)}")
        print(f"🔵 BAJOS: {len(low)}")
        print(f"📊 TOTAL: {len(self.issues)}")

        # Mostrar issues críticos
        if critical:
            print("\n🚨 ISSUES CRÍTICOS (REQUIEREN ACCIÓN INMEDIATA):")
            print("-" * 50)
            for issue in critical:
                print(f"📁 {issue.file}:{issue.line}")
                print(f"🔍 Código: {issue.code}")
                print(f"⚠️ Problema: {issue.description}")
                print(f"💡 Recomendación: {issue.recommendation}")
                print()

        # Mostrar issues altos
        if high:
            print("\n⚠️ ISSUES ALTOS:")
            print("-" * 30)
            for issue in high:
                print(f"📁 {issue.file}:{issue.line} - {issue.description}")
                print(f"💡 {issue.recommendation}")
                print()

        # Resumen por archivo
        print("\n📊 RESUMEN POR ARCHIVO:")
        print("-" * 30)
        file_counts = {}
        for issue in self.issues:
            if issue.file not in file_counts:
                file_counts[issue.file] = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            file_counts[issue.file][issue.severity] += 1

        for file_path, counts in file_counts.items():
            total = sum(counts.values())
            print(f"📁 {file_path}: {total} issues")
            if counts['CRITICAL'] > 0:
                print(f"   🚨 {counts['CRITICAL']} críticos")
            if counts['HIGH'] > 0:
                print(f"   ⚠️ {counts['HIGH']} altos")
            if counts['MEDIUM'] > 0:
                print(f"   🔶 {counts['MEDIUM']} medios")
            if counts['LOW'] > 0:
                print(f"   🔵 {counts['LOW']} bajos")

        # Recomendaciones generales
        print("\n💡 RECOMENDACIONES GENERALES:")
        print("-" * 40)
        print("1. 🔧 Crear archivo config/risk_parameters.py")
        print("2. 📝 Mover todos los valores críticos a variables de entorno")
        print("3. 🔄 Implementar recarga dinámica de configuración")
        print("4. 📊 Usar APIs para obtener datos en tiempo real")
        print("5. ✅ Validar configuración al inicio del sistema")

        # Estado del sistema
        if critical or high:
            print("\n🚨 ESTADO DEL SISTEMA: REQUIERE ATENCIÓN")
            print("❌ El sistema tiene valores hardcodeados que pueden afectar el trading")
        else:
            print("\n✅ ESTADO DEL SISTEMA: ACEPTABLE")
            print("✅ Solo issues menores encontrados")

def main():
    """🚀 Función principal"""
    auditor = HardcodedValuesAuditor()
    auditor.audit_system()

if __name__ == "__main__":
    main()
