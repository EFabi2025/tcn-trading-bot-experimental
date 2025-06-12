#!/usr/bin/env python3
"""
🔍 VERIFICADOR DE RESULTADOS DE SINCRONIZACIÓN
Script para analizar y mostrar los datos sincronizados en la base de datos
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal

def connect_database():
    """🔌 Conectar a la base de datos"""
    try:
        conn = sqlite3.connect('trading_bot.db')
        conn.row_factory = sqlite3.Row  # Para acceso por nombre de columna
        return conn
    except Exception as e:
        print(f"❌ Error conectando a la base de datos: {e}")
        return None

def show_database_summary():
    """📊 Mostrar resumen general de la base de datos"""
    print("📊 RESUMEN GENERAL DE LA BASE DE DATOS")
    print("=" * 50)
    
    conn = connect_database()
    if not conn:
        return
    
    try:
        cursor = conn.cursor()
        
        # Contar registros en cada tabla
        tables = ['trades', 'performance_metrics', 'system_logs', 'risk_events', 'market_data_cache']
        
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
            count = cursor.fetchone()['count']
            print(f"📋 {table}: {count} registros")
        
        # Mostrar rango de fechas
        cursor.execute("SELECT MIN(entry_time) as min_date, MAX(entry_time) as max_date FROM trades")
        dates = cursor.fetchone()
        if dates['min_date']:
            print(f"📅 Rango de trades: {dates['min_date']} a {dates['max_date']}")
        
    except Exception as e:
        print(f"❌ Error obteniendo resumen: {e}")
    finally:
        conn.close()

def show_balance_history():
    """💰 Mostrar historial de balance"""
    print("\n💰 HISTORIAL DE BALANCE")
    print("=" * 50)
    
    conn = connect_database()
    if not conn:
        return
    
    try:
        cursor = conn.cursor()
        
        # Obtener últimas métricas de performance
        cursor.execute("""
            SELECT timestamp, total_balance, daily_pnl, total_pnl, win_rate, 
                   active_positions_count, trades_today
            FROM performance_metrics 
            ORDER BY timestamp DESC 
            LIMIT 10
        """)
        
        metrics = cursor.fetchall()
        
        if metrics:
            print("🕒 Últimas 10 métricas registradas:")
            print(f"{'Fecha':<20} {'Balance':<12} {'PnL Diario':<12} {'PnL Total':<12} {'Win Rate':<10}")
            print("-" * 70)
            
            for metric in metrics:
                timestamp = metric['timestamp'][:19]  # Solo fecha y hora
                balance = f"${float(metric['total_balance']):.2f}"
                daily_pnl = f"${float(metric['daily_pnl']):.2f}"
                total_pnl = f"${float(metric['total_pnl']):.2f}"
                win_rate = f"{float(metric['win_rate']):.1%}"
                
                print(f"{timestamp:<20} {balance:<12} {daily_pnl:<12} {total_pnl:<12} {win_rate:<10}")
        else:
            print("❌ No se encontraron métricas de balance")
    
    except Exception as e:
        print(f"❌ Error obteniendo balance: {e}")
    finally:
        conn.close()

def show_trades_analysis():
    """📈 Análisis de trades"""
    print("\n📈 ANÁLISIS DE TRADES")
    print("=" * 50)
    
    conn = connect_database()
    if not conn:
        return
    
    try:
        cursor = conn.cursor()
        
        # Resumen general de trades
        cursor.execute("""
            SELECT 
                COUNT(*) as total_trades,
                COUNT(DISTINCT symbol) as unique_symbols,
                SUM(CASE WHEN side = 'BUY' THEN 1 ELSE 0 END) as buy_trades,
                SUM(CASE WHEN side = 'SELL' THEN 1 ELSE 0 END) as sell_trades,
                SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END) as active_trades,
                AVG(confidence) as avg_confidence
            FROM trades
        """)
        
        summary = cursor.fetchone()
        
        print("📊 Resumen general:")
        print(f"   💼 Total trades: {summary['total_trades']}")
        print(f"   💱 Símbolos únicos: {summary['unique_symbols']}")
        print(f"   🟢 Compras (BUY): {summary['buy_trades']}")
        print(f"   🔴 Ventas (SELL): {summary['sell_trades']}")
        print(f"   ⏳ Trades activos: {summary['active_trades']}")
        print(f"   🎯 Confianza promedio: {float(summary['avg_confidence']):.2%}")
        
        # Análisis por símbolo
        cursor.execute("""
            SELECT 
                symbol,
                COUNT(*) as trade_count,
                AVG(entry_price) as avg_entry_price,
                AVG(confidence) as avg_confidence,
                SUM(CASE WHEN side = 'BUY' THEN 1 ELSE 0 END) as buys,
                SUM(CASE WHEN side = 'SELL' THEN 1 ELSE 0 END) as sells
            FROM trades 
            GROUP BY symbol 
            ORDER BY trade_count DESC
        """)
        
        symbols = cursor.fetchall()
        
        if symbols:
            print(f"\n📊 Análisis por símbolo:")
            print(f"{'Símbolo':<10} {'Trades':<8} {'Precio Avg':<12} {'Confianza':<12} {'Compras':<8} {'Ventas':<8}")
            print("-" * 70)
            
            for symbol in symbols:
                sym = symbol['symbol']
                count = symbol['trade_count']
                avg_price = f"${float(symbol['avg_entry_price']):.2f}"
                avg_conf = f"{float(symbol['avg_confidence']):.1%}"
                buys = symbol['buys']
                sells = symbol['sells']
                
                print(f"{sym:<10} {count:<8} {avg_price:<12} {avg_conf:<12} {buys:<8} {sells:<8}")
    
    except Exception as e:
        print(f"❌ Error analizando trades: {e}")
    finally:
        conn.close()

def show_recent_trades():
    """🕒 Mostrar trades recientes"""
    print("\n🕒 TRADES RECIENTES")
    print("=" * 50)
    
    conn = connect_database()
    if not conn:
        return
    
    try:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                symbol, side, quantity, entry_price, entry_time, 
                confidence, strategy, is_active
            FROM trades 
            ORDER BY entry_time DESC 
            LIMIT 15
        """)
        
        trades = cursor.fetchall()
        
        if trades:
            print("📋 Últimos 15 trades:")
            print(f"{'Fecha':<20} {'Símbolo':<10} {'Lado':<5} {'Cantidad':<12} {'Precio':<12} {'Conf.':<8} {'Activo':<7}")
            print("-" * 85)
            
            for trade in trades:
                date = trade['entry_time'][:19] if trade['entry_time'] else 'N/A'
                symbol = trade['symbol']
                side = trade['side']
                quantity = f"{float(trade['quantity']):.6f}"
                price = f"${float(trade['entry_price']):.2f}"
                confidence = f"{float(trade['confidence']):.2%}"
                active = "✅" if trade['is_active'] else "❌"
                
                print(f"{date:<20} {symbol:<10} {side:<5} {quantity:<12} {price:<12} {confidence:<8} {active:<7}")
        else:
            print("❌ No se encontraron trades")
    
    except Exception as e:
        print(f"❌ Error obteniendo trades recientes: {e}")
    finally:
        conn.close()

def show_sync_logs():
    """📝 Mostrar logs de sincronización"""
    print("\n📝 LOGS DE SINCRONIZACIÓN")
    print("=" * 50)
    
    conn = connect_database()
    if not conn:
        return
    
    try:
        cursor = conn.cursor()
        
        # Buscar logs relacionados con sincronización
        cursor.execute("""
            SELECT timestamp, level, category, message 
            FROM system_logs 
            WHERE category = 'SYNC' OR message LIKE '%sync%' OR message LIKE '%SYNC%'
            ORDER BY timestamp DESC 
            LIMIT 10
        """)
        
        logs = cursor.fetchall()
        
        if logs:
            print("📋 Últimos logs de sincronización:")
            for log in logs:
                timestamp = log['timestamp'][:19]
                level = log['level']
                message = log['message'][:60] + "..." if len(log['message']) > 60 else log['message']
                
                level_emoji = {
                    'INFO': '✅',
                    'WARNING': '⚠️',
                    'ERROR': '❌',
                    'CRITICAL': '🚨'
                }.get(level, '📝')
                
                print(f"   {level_emoji} {timestamp} [{level}] {message}")
        else:
            print("❌ No se encontraron logs de sincronización")
    
    except Exception as e:
        print(f"❌ Error obteniendo logs: {e}")
    finally:
        conn.close()

def show_data_quality_check():
    """🔍 Verificación de calidad de datos"""
    print("\n🔍 VERIFICACIÓN DE CALIDAD DE DATOS")
    print("=" * 50)
    
    conn = connect_database()
    if not conn:
        return
    
    try:
        cursor = conn.cursor()
        
        # Verificar datos inconsistentes
        issues = []
        
        # 1. Trades sin precios
        cursor.execute("SELECT COUNT(*) as count FROM trades WHERE entry_price <= 0")
        invalid_prices = cursor.fetchone()['count']
        if invalid_prices > 0:
            issues.append(f"❌ {invalid_prices} trades con precios inválidos")
        
        # 2. Trades sin fechas
        cursor.execute("SELECT COUNT(*) as count FROM trades WHERE entry_time IS NULL")
        no_dates = cursor.fetchone()['count']
        if no_dates > 0:
            issues.append(f"❌ {no_dates} trades sin fecha de entrada")
        
        # 3. Balances negativos
        cursor.execute("SELECT COUNT(*) as count FROM performance_metrics WHERE total_balance < 0")
        negative_balance = cursor.fetchone()['count']
        if negative_balance > 0:
            issues.append(f"⚠️ {negative_balance} registros con balance negativo")
        
        # 4. Verificar duplicados
        cursor.execute("""
            SELECT COUNT(*) as count FROM (
                SELECT symbol, entry_time, entry_price, quantity 
                FROM trades 
                GROUP BY symbol, entry_time, entry_price, quantity 
                HAVING COUNT(*) > 1
            )
        """)
        duplicates = cursor.fetchone()['count']
        if duplicates > 0:
            issues.append(f"⚠️ {duplicates} posibles trades duplicados")
        
        if issues:
            print("🚨 Problemas encontrados:")
            for issue in issues:
                print(f"   {issue}")
        else:
            print("✅ No se encontraron problemas en los datos")
        
        # Estadísticas de estrategias
        cursor.execute("""
            SELECT strategy, COUNT(*) as count 
            FROM trades 
            GROUP BY strategy 
            ORDER BY count DESC
        """)
        
        strategies = cursor.fetchall()
        if strategies:
            print(f"\n📊 Distribución por estrategia:")
            for strategy in strategies:
                print(f"   🎯 {strategy['strategy']}: {strategy['count']} trades")
    
    except Exception as e:
        print(f"❌ Error en verificación de calidad: {e}")
    finally:
        conn.close()

def main():
    """🚀 Función principal"""
    print("🔍 VERIFICADOR DE RESULTADOS DE SINCRONIZACIÓN")
    print("=" * 60)
    
    # Verificar que existe la base de datos
    try:
        conn = sqlite3.connect('trading_bot.db')
        conn.close()
    except Exception as e:
        print(f"❌ No se puede acceder a trading_bot.db: {e}")
        return
    
    # Ejecutar todas las verificaciones
    show_database_summary()
    show_balance_history()
    show_trades_analysis()
    show_recent_trades()
    show_sync_logs()
    show_data_quality_check()
    
    print(f"\n✅ Verificación completa!")
    print(f"📊 Revisa los resultados arriba para evaluar la sincronización")

if __name__ == "__main__":
    main() 