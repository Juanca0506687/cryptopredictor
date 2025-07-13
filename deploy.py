#!/usr/bin/env python3
"""
Script para preparar el proyecto CryptoPredictor para despliegue
"""

import os
import subprocess
import sys
import json
from datetime import datetime

def print_step(message):
    """Imprimir paso con formato"""
    print(f"\n{'='*50}")
    print(f"🔄 {message}")
    print(f"{'='*50}")

def run_command(command, description):
    """Ejecutar comando y mostrar resultado"""
    print(f"\n📋 {description}")
    print(f"Comando: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Comando ejecutado exitosamente")
            if result.stdout:
                print(f"Salida: {result.stdout}")
        else:
            print(f"❌ Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error ejecutando comando: {e}")
        return False
    
    return True

def check_files():
    """Verificar que todos los archivos necesarios existan"""
    print_step("Verificando archivos del proyecto")
    
    required_files = [
        'app.py',
        'requirements.txt',
        'Procfile',
        'runtime.txt',
        'wsgi.py',
        'config.py',
        '.gitignore',
        'README.md',
        'templates/index.html',
        'templates/advanced_dashboard.html'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Archivos faltantes: {missing_files}")
        return False
    else:
        print("✅ Todos los archivos necesarios están presentes")
        return True

def initialize_git():
    """Inicializar repositorio Git"""
    print_step("Inicializando repositorio Git")
    
    commands = [
        ("git init", "Inicializando Git"),
        ("git add .", "Agregando archivos"),
        ("git commit -m 'Initial commit - CryptoPredictor ready for deployment'", "Haciendo commit inicial")
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    
    return True

def create_deployment_guide():
    """Crear guía de despliegue"""
    print_step("Creando guía de despliegue")
    
    guide = """
# 🚀 Guía de Despliegue - CryptoPredictor

## Opciones de Despliegue

### 1. Railway (Recomendado - Más Fácil)
1. Ve a [railway.app](https://railway.app)
2. Regístrate con tu cuenta de GitHub
3. Haz clic en "New Project"
4. Selecciona "Deploy from GitHub repo"
5. Conecta tu repositorio
6. Railway detectará automáticamente que es una app Flask
7. ¡Listo! Tu app estará disponible en unos minutos

### 2. Render
1. Ve a [render.com](https://render.com)
2. Regístrate con tu cuenta de GitHub
3. Haz clic en "New Web Service"
4. Conecta tu repositorio de GitHub
5. Configura:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
   - Environment: `FLASK_ENV=production`
6. Haz clic en "Create Web Service"

### 3. Heroku
1. Instala Heroku CLI desde [heroku.com](https://heroku.com)
2. Ejecuta en terminal:
   ```bash
   heroku login
   heroku create tu-app-name
   git push heroku main
   ```

## Variables de Entorno Recomendadas
- `FLASK_ENV=production`
- `SECRET_KEY=tu-clave-secreta-segura`

## Verificación Post-Despliegue
1. Verifica que la app responda en la URL proporcionada
2. Prueba las funcionalidades principales
3. Revisa los logs por errores

¡Tu CryptoPredictor estará en línea! 🎉
"""
    
    with open('DEPLOYMENT_GUIDE.md', 'w', encoding='utf-8') as f:
        f.write(guide)
    
    print("✅ Guía de despliegue creada: DEPLOYMENT_GUIDE.md")
    return True

def create_app_json():
    """Crear app.json para configuración de despliegue"""
    print_step("Creando app.json para configuración")
    
    app_config = {
        "name": "CryptoPredictor",
        "description": "Predicción de criptomonedas con IA",
        "repository": "https://github.com/tu-usuario/crypto-predictor",
        "keywords": ["python", "flask", "cryptocurrency", "machine-learning", "prediction"],
        "env": {
            "FLASK_ENV": {
                "description": "Environment de Flask",
                "value": "production"
            },
            "SECRET_KEY": {
                "description": "Clave secreta de Flask",
                "generator": "secret"
            }
        },
        "buildpacks": [
            {
                "url": "heroku/python"
            }
        ]
    }
    
    with open('app.json', 'w', encoding='utf-8') as f:
        json.dump(app_config, f, indent=2)
    
    print("✅ app.json creado")
    return True

def main():
    """Función principal"""
    print("🚀 Preparando CryptoPredictor para despliegue...")
    print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Verificar archivos
    if not check_files():
        print("❌ Error: Faltan archivos necesarios")
        sys.exit(1)
    
    # Inicializar Git
    if not initialize_git():
        print("❌ Error: No se pudo inicializar Git")
        sys.exit(1)
    
    # Crear archivos adicionales
    create_deployment_guide()
    create_app_json()
    
    print_step("🎉 ¡Proyecto listo para despliegue!")
    print("""
📋 Próximos pasos:
1. Sube tu código a GitHub
2. Elige una plataforma de despliegue
3. Sigue la guía en DEPLOYMENT_GUIDE.md
4. ¡Tu CryptoPredictor estará en línea!

🔗 Archivos creados:
- DEPLOYMENT_GUIDE.md (guía de despliegue)
- app.json (configuración para Heroku)

💡 Recomendación: Usa Railway para el despliegue más fácil
""")

if __name__ == "__main__":
    main() 