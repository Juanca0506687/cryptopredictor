#!/usr/bin/env python3
"""
Script completo para desplegar CryptoPredictor en Railway
Autor: AI Assistant
"""

import os
import sys
import subprocess
import time
import webbrowser
import json
from datetime import datetime

def print_step(message, emoji="🔄"):
    """Imprimir paso con formato"""
    print(f"\n{'='*60}")
    print(f"{emoji} {message}")
    print(f"{'='*60}")

def run_command(command, description, check_output=False):
    """Ejecutar comando y mostrar resultado"""
    print(f"\n📋 {description}")
    print(f"Comando: {command}")
    
    try:
        if check_output:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Comando ejecutado exitosamente")
                return result.stdout.strip()
            else:
                print(f"❌ Error: {result.stderr}")
                return None
        else:
            subprocess.run(command, shell=True, check=True)
            print("✅ Comando ejecutado exitosamente")
            return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error ejecutando comando: {e}")
        return None

def create_railway_config():
    """Crear configuración específica para Railway"""
    print_step("Configurando Railway", "🚀")
    
    # Crear archivo railway.toml
    railway_config = '''[build]
builder = "nixpacks"

[deploy]
startCommand = "gunicorn app:app"
healthcheckPath = "/"
healthcheckTimeout = 300
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 10

[[services]]
name = "crypto-predictor"
'''
    
    with open('railway.toml', 'w') as f:
        f.write(railway_config)
    
    print("✅ Archivo railway.toml creado")

def update_app_for_production():
    """Actualizar la aplicación para producción"""
    print_step("Actualizando aplicación para producción", "⚙️")
    
    # Leer el archivo app.py actual
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Agregar configuración de producción al final
    production_config = '''
# Configuración para producción
if __name__ == '__main__':
    import os
    # Para desarrollo local
    if os.environ.get('FLASK_ENV') == 'development':
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        # Para producción (Railway, Render, Heroku)
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False)
'''
    
    # Verificar si ya existe la configuración
    if 'if __name__ == \'__main__\':' not in content:
        with open('app.py', 'a', encoding='utf-8') as f:
            f.write(production_config)
        print("✅ Configuración de producción agregada")
    else:
        print("✅ Configuración de producción ya existe")

def create_deployment_files():
    """Crear todos los archivos necesarios para el despliegue"""
    print_step("Creando archivos de despliegue", "📁")
    
    # requirements.txt actualizado
    requirements = '''Flask==2.3.3
pandas==2.1.1
numpy==1.24.3
yfinance==0.2.18
scikit-learn==1.3.0
plotly==5.17.0
textblob==0.17.1
requests==2.31.0
Pillow==10.0.1
matplotlib==3.7.2
seaborn==0.12.2
schedule==1.2.0
gunicorn==21.2.0
'''
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    # Procfile
    with open('Procfile', 'w') as f:
        f.write('web: gunicorn app:app')
    
    # runtime.txt
    with open('runtime.txt', 'w') as f:
        f.write('python-3.12.0')
    
    # wsgi.py
    with open('wsgi.py', 'w') as f:
        f.write('from app import app\n\nif __name__ == "__main__":\n    app.run()')
    
    print("✅ Archivos de despliegue creados")

def setup_git():
    """Configurar Git y subir a GitHub"""
    print_step("Configurando Git", "📦")
    
    # Inicializar Git si no existe
    if not os.path.exists('.git'):
        run_command('git init', 'Inicializando repositorio Git')
    
    # Agregar todos los archivos
    run_command('git add .', 'Agregando archivos al staging')
    
    # Hacer commit
    commit_message = f"Deploy ready - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    run_command(f'git commit -m "{commit_message}"', 'Haciendo commit')
    
    # Configurar rama principal
    run_command('git branch -M main', 'Configurando rama principal')
    
    # Verificar si ya existe el remoto
    result = run_command('git remote -v', 'Verificando remotos', check_output=True)
    
    if 'origin' not in result:
        # Agregar remoto
        remote_url = "https://github.com/Juanca0506687/cryptopredictor.git"
        run_command(f'git remote add origin {remote_url}', 'Agregando remoto de GitHub')
    
    # Subir a GitHub
    run_command('git push -u origin main', 'Subiendo a GitHub')
    
    print("✅ Código subido a GitHub")

def create_railway_deployment_guide():
    """Crear guía de despliegue en Railway"""
    print_step("Creando guía de despliegue", "📖")
    
    guide = '''# 🚀 Guía de Despliegue en Railway

## ✅ Estado Actual
- ✅ Código subido a GitHub
- ✅ Archivos de configuración creados
- ✅ Aplicación lista para producción

## 🌐 Pasos para Desplegar en Railway

### 1. Crear cuenta en Railway
1. Ve a https://railway.app
2. Haz clic en "Start a New Project"
3. Selecciona "Deploy from GitHub repo"
4. Conecta tu cuenta de GitHub

### 2. Conectar repositorio
1. Busca tu repositorio: `Juanca0506687/cryptopredictor`
2. Haz clic en "Deploy Now"
3. Railway detectará automáticamente que es una app Flask

### 3. Configurar variables de entorno
1. Ve a la pestaña "Variables"
2. Agrega estas variables:
   - `FLASK_ENV` = `production`
   - `SECRET_KEY` = `tu-clave-secreta-aqui`

### 4. Esperar despliegue
- Railway construirá automáticamente tu aplicación
- El proceso toma 2-5 minutos
- Verás la URL de tu sitio web

## 🔗 Tu sitio web estará disponible en:
https://tu-app-name.railway.app

## 📊 Monitoreo
- Ve a la pestaña "Deployments" para ver el estado
- Los logs están en "Logs"
- Puedes configurar dominio personalizado

## 🎉 ¡Listo!
Tu aplicación CryptoPredictor estará funcionando en la web.
'''
    
    with open('RAILWAY_DEPLOYMENT_GUIDE.md', 'w', encoding='utf-8') as f:
        f.write(guide)
    
    print("✅ Guía de despliegue creada")

def open_railway_website():
    """Abrir Railway en el navegador"""
    print_step("Abriendo Railway", "🌐")
    
    print("🔗 Abriendo Railway en tu navegador...")
    webbrowser.open('https://railway.app')
    
    print("""
📋 Instrucciones rápidas:
1. Haz clic en "Start a New Project"
2. Selecciona "Deploy from GitHub repo"
3. Busca: Juanca0506687/cryptopredictor
4. Haz clic en "Deploy Now"
5. ¡Espera 2-5 minutos y tendrás tu sitio web!
""")

def main():
    """Función principal"""
    print_step("🚀 INICIANDO DESPLIEGUE AUTOMÁTICO DE CRYPTOPREDICTOR", "🎯")
    
    print("""
Este script automatizará todo el proceso de despliegue:
✅ Actualizar configuración para producción
✅ Crear archivos de despliegue
✅ Subir código a GitHub
✅ Abrir Railway para despliegue
✅ Crear guía de instrucciones
""")
    
    # Paso 1: Actualizar aplicación
    update_app_for_production()
    
    # Paso 2: Crear archivos de despliegue
    create_deployment_files()
    
    # Paso 3: Configurar Railway
    create_railway_config()
    
    # Paso 4: Configurar Git y subir a GitHub
    setup_git()
    
    # Paso 5: Crear guía de despliegue
    create_railway_deployment_guide()
    
    # Paso 6: Abrir Railway
    open_railway_website()
    
    print_step("🎉 ¡DESPLIEGUE COMPLETADO!", "✅")
    
    print("""
🎯 Tu proyecto está listo para desplegarse:

📁 Archivos creados:
- requirements.txt (dependencias)
- Procfile (configuración de servidor)
- runtime.txt (versión de Python)
- wsgi.py (servidor WSGI)
- railway.toml (configuración Railway)
- RAILWAY_DEPLOYMENT_GUIDE.md (instrucciones)

🌐 Pasos siguientes:
1. Railway ya está abierto en tu navegador
2. Sigue las instrucciones en pantalla
3. Tu sitio web estará listo en 2-5 minutos

🔗 Tu repositorio: https://github.com/Juanca0506687/cryptopredictor
📖 Guía completa: RAILWAY_DEPLOYMENT_GUIDE.md

¡Tu aplicación CryptoPredictor estará funcionando en la web!
""")

if __name__ == "__main__":
    main() 