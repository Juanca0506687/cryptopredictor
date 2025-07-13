#!/usr/bin/env python3
"""
Script de despliegue automatizado para CryptoPredictor
"""

import os
import subprocess
import sys
import json
from datetime import datetime

def print_banner():
    print("=" * 60)
    print("🚀 CRYPTOPREDICTOR - DESPLIEGUE AUTOMATIZADO")
    print("=" * 60)
    print()

def check_requirements():
    """Verifica que todos los archivos necesarios estén presentes"""
    required_files = [
        'crypto_simple.py',
        'requirements.txt',
        'Procfile',
        'runtime.txt',
        'wsgi.py',
        'config.py',
        'templates/simple.html',
        '.gitignore'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Archivos faltantes:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ Todos los archivos necesarios están presentes")
    return True

def create_gitignore():
    """Crea archivo .gitignore si no existe"""
    if not os.path.exists('.gitignore'):
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Application specific
crypto_data.json
*.log
.env

# Heroku
.env
.env.local
.env.development.local
.env.test.local
.env.production.local
"""
        with open('.gitignore', 'w') as f:
            f.write(gitignore_content)
        print("✅ Archivo .gitignore creado")

def create_readme():
    """Crea archivo README.md si no existe"""
    if not os.path.exists('README.md'):
        readme_content = """# CryptoPredictor - Predicción de Criptomonedas con IA

Una aplicación web moderna para predecir precios de criptomonedas utilizando inteligencia artificial y análisis técnico.

## 🚀 Características

- **Predicción de Precios**: Modelo de machine learning para predecir precios futuros
- **Análisis Técnico**: Indicadores RSI, SMA y más
- **Comparación de Criptomonedas**: Compara múltiples criptomonedas con score de oportunidad
- **Interfaz Moderna**: Diseño responsive con Tailwind CSS
- **Gráficos Interactivos**: Visualización de predicciones con Plotly

## 🛠️ Tecnologías

- **Backend**: Flask (Python)
- **Machine Learning**: Scikit-learn
- **Datos**: Yahoo Finance API
- **Frontend**: HTML, JavaScript, Tailwind CSS
- **Gráficos**: Plotly.js

## 📋 Requisitos

- Python 3.8 o superior
- Conexión a internet

## 🚀 Despliegue

### Heroku
```bash
# Crear aplicación en Heroku
heroku create tu-app-name

# Configurar variables de entorno
heroku config:set SECRET_KEY=tu_clave_secreta

# Desplegar
git push heroku main
```

### Railway
```bash
# Conectar con Railway
railway login
railway init
railway up
```

### Render
```bash
# Conectar con Render
# Crear nuevo Web Service
# Conectar con GitHub
# Configurar build command: pip install -r requirements.txt
# Configurar start command: gunicorn crypto_simple:app
```

## 📊 Uso

1. Accede a la aplicación web
2. Selecciona una criptomoneda
3. Elige el número de días a predecir
4. Haz clic en "Predecir Precios"
5. Revisa los resultados y consejos de inversión

## ⚠️ Advertencia

Esta aplicación es solo para fines educativos. Las predicciones no constituyen consejos financieros.

## 📝 Licencia

MIT License

---
**¡Disfruta prediciendo el futuro de las criptomonedas! 🚀**
"""
        with open('README.md', 'w') as f:
            f.write(readme_content)
        print("✅ Archivo README.md creado")

def setup_git():
    """Configura Git si no está inicializado"""
    if not os.path.exists('.git'):
        try:
            subprocess.run(['git', 'init'], check=True)
            subprocess.run(['git', 'add', '.'], check=True)
            subprocess.run(['git', 'commit', '-m', 'Initial commit - CryptoPredictor'], check=True)
            print("✅ Repositorio Git inicializado")
        except subprocess.CalledProcessError:
            print("⚠️  No se pudo inicializar Git (puede que no esté instalado)")
    else:
        print("✅ Repositorio Git ya existe")

def deploy_to_heroku():
    """Guía para desplegar en Heroku"""
    print("\n🌐 DESPLIEGUE EN HEROKU")
    print("=" * 40)
    print("1. Instala Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli")
    print("2. Ejecuta estos comandos:")
    print()
    print("   # Login en Heroku")
    print("   heroku login")
    print()
    print("   # Crear aplicación")
    print("   heroku create tu-cryptopredictor-app")
    print()
    print("   # Configurar variables")
    print("   heroku config:set SECRET_KEY=tu_clave_secreta_aqui")
    print()
    print("   # Desplegar")
    print("   git push heroku main")
    print()
    print("   # Abrir aplicación")
    print("   heroku open")
    print()

def deploy_to_railway():
    """Guía para desplegar en Railway"""
    print("\n🚂 DESPLIEGUE EN RAILWAY")
    print("=" * 40)
    print("1. Ve a https://railway.app")
    print("2. Conecta tu cuenta de GitHub")
    print("3. Crea un nuevo proyecto")
    print("4. Selecciona 'Deploy from GitHub repo'")
    print("5. Selecciona este repositorio")
    print("6. Railway detectará automáticamente que es una app Flask")
    print("7. ¡Listo! Tu app estará disponible en la URL que te proporcione Railway")
    print()

def deploy_to_render():
    """Guía para desplegar en Render"""
    print("\n🎨 DESPLIEGUE EN RENDER")
    print("=" * 40)
    print("1. Ve a https://render.com")
    print("2. Conecta tu cuenta de GitHub")
    print("3. Crea un nuevo 'Web Service'")
    print("4. Selecciona este repositorio")
    print("5. Configura:")
    print("   - Build Command: pip install -r requirements.txt")
    print("   - Start Command: gunicorn crypto_simple:app")
    print("6. ¡Listo! Tu app estará disponible en la URL que te proporcione Render")
    print()

def main():
    print_banner()
    
    # Verificar archivos
    if not check_requirements():
        print("\n❌ Por favor, asegúrate de que todos los archivos estén presentes")
        return
    
    # Crear archivos adicionales
    create_gitignore()
    create_readme()
    setup_git()
    
    print("\n🎉 ¡PREPARACIÓN COMPLETADA!")
    print("=" * 40)
    print("Tu aplicación está lista para ser desplegada.")
    print("Elige una plataforma de despliegue:")
    print()
    print("1. Heroku (Recomendado para principiantes)")
    print("2. Railway (Gratis, fácil de usar)")
    print("3. Render (Gratis, bueno para proyectos)")
    print()
    
    choice = input("¿Qué plataforma prefieres? (1/2/3): ").strip()
    
    if choice == "1":
        deploy_to_heroku()
    elif choice == "2":
        deploy_to_railway()
    elif choice == "3":
        deploy_to_render()
    else:
        print("Mostrando todas las opciones:")
        deploy_to_heroku()
        deploy_to_railway()
        deploy_to_render()
    
    print("\n📚 RECURSOS ADICIONALES:")
    print("- Documentación Flask: https://flask.palletsprojects.com/")
    print("- Heroku Dev Center: https://devcenter.heroku.com/")
    print("- Railway Docs: https://docs.railway.app/")
    print("- Render Docs: https://render.com/docs")
    print()
    print("¡Buena suerte con tu despliegue! 🚀")

if __name__ == "__main__":
    main() 