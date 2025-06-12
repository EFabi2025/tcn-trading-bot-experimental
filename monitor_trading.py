#!/usr/bin/env python3
"""
MONITOR TRADING SYSTEM - Monitoreo en tiempo real
"""

import os
import time
import json
from datetime import datetime
import asyncio
import aiohttp

class TradingMonitor:
    def __init__(self):
        self.api_key = os.getenv("BINANCE_API_KEY", "")
        self.base_url = "https://testnet.binance.vision"  # Cambiar para producción
    
    async def get_account_info(self):
        """Obtener información de cuenta"""
        if not self.api_key:
            return None
            
        headers = {"X-MBX-APIKEY": self.api_key}
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.base_url}/api/v3/account", headers=headers) as response:
                    if response.status == 200:
                        return await response.json()
            except:
                pass
        return None
    
    async def monitor_loop(self):
        """Bucle de monitoreo"""
        print("📊 TRADING SYSTEM MONITOR")
        print("="*40)
        
        while True:
            try:
                now = datetime.now().strftime("%H:%M:%S")
                
                # Estado del sistema
                print(f"\n[{now}] 🔄 Sistema activo")
                
                # Info de cuenta
                account_info = await self.get_account_info()
                if account_info:
                    balances = account_info.get('balances', [])
                    usdt_balance = next((b for b in balances if b['asset'] == 'USDT'), None)
                    if usdt_balance:
                        print(f"💰 Balance USDT: ${float(usdt_balance['free']):.2f}")
                
                # Verificar log de trading
                if os.path.exists('binance_tcn_trading.log'):
                    with open('binance_tcn_trading.log', 'r') as f:
                        lines = f.readlines()
                        if lines:
                            last_line = lines[-1].strip()
                            if 'TRADE EJECUTADO' in last_line:
                                print("🎯 Último trade detectado en logs")
                
                await asyncio.sleep(30)  # Check cada 30 segundos
                
            except KeyboardInterrupt:
                print("\n👋 Monitor detenido")
                break
            except Exception as e:
                print(f"❌ Error en monitor: {e}")
                await asyncio.sleep(10)

if __name__ == "__main__":
    monitor = TradingMonitor()
    asyncio.run(monitor.monitor_loop())
