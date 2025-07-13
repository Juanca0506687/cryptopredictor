#!/usr/bin/env python3
"""
Script simplificado para ejecutar CryptoPredictor
Instala las dependencias necesarias y ejecuta la aplicación
"""

import subprocess
import sys
import os

def install_requirements():
    """Instala las dependencias necesarias"""
    print("🔧 Instalando dependencias...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_simple.txt"])
        print("✅ Dependencias instaladas correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        return False

def run_app():
    """Ejecuta la aplicación"""
    print("🚀 Iniciando CryptoPredictor...")
    try:
        subprocess.run([sys.executable, "app_simple.py"])
    except KeyboardInterrupt:
        print("\n👋 Aplicación detenida por el usuario")
    except Exception as e:
        print(f"❌ Error ejecutando la aplicación: {e}")

def main():
    print("=" * 50)
    print("🤖 CryptoPredictor - Predicción de Criptomonedas")
    print("=" * 50)
    
    # Verificar si estamos en el directorio correcto
    if not os.path.exists("app_simple.py"):
        print("❌ Error: No se encontró app_simple.py")
        print("Asegúrate de estar en el directorio correcto")
        return
    
    # Instalar dependencias
    if not install_requirements():
        print("❌ No se pudieron instalar las dependencias")
        return
    
    # Ejecutar aplicación
    run_app()

if __name__ == "__main__":
    main() 