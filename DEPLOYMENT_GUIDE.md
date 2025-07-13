# 🚀 Guía de Despliegue - CryptoPredictor

## 📋 Estado Actual
✅ **Proyecto listo para despliegue**
✅ **Git inicializado y commit realizado**
✅ **Todos los archivos necesarios creados**

## 🌐 Opciones de Despliegue

### 1. Railway (Recomendado - Más Fácil)

**Pasos:**
1. Ve a [railway.app](https://railway.app)
2. Regístrate con tu cuenta de GitHub
3. Haz clic en "New Project"
4. Selecciona "Deploy from GitHub repo"
5. Conecta tu repositorio de GitHub
6. Railway detectará automáticamente que es una app Flask
7. ¡Listo! Tu app estará disponible en unos minutos

**Ventajas:**
- ✅ Gratis
- ✅ Fácil de usar
- ✅ Despliegue automático
- ✅ SSL incluido

### 2. Render

**Pasos:**
1. Ve a [render.com](https://render.com)
2. Regístrate con tu cuenta de GitHub
3. Haz clic en "New Web Service"
4. Conecta tu repositorio de GitHub
5. Configura:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Environment**: `FLASK_ENV=production`
6. Haz clic en "Create Web Service"

**Ventajas:**
- ✅ Gratis
- ✅ Buena documentación
- ✅ SSL incluido

### 3. Heroku

**Pasos:**
1. Instala Heroku CLI desde [heroku.com](https://heroku.com)
2. Ejecuta en terminal:
   ```bash
   heroku login
   heroku create tu-app-name
   git push heroku main
   ```

**Ventajas:**
- ✅ Muy estable
- ✅ Buena documentación
- ✅ Muchos add-ons disponibles

## 🔧 Variables de Entorno Recomendadas

```bash
FLASK_ENV=production
SECRET_KEY=tu-clave-secreta-segura
```

## 📁 Archivos del Proyecto

```
crypto-predictor/
├── app.py                 # ✅ Aplicación principal Flask
├── requirements.txt       # ✅ Dependencias Python
├── Procfile             # ✅ Configuración para Heroku
├── runtime.txt          # ✅ Versión de Python
├── wsgi.py             # ✅ Servidor WSGI
├── config.py           # ✅ Configuración de la app
├── templates/          # ✅ Plantillas HTML
│   ├── index.html      # ✅ Página principal
│   └── advanced_dashboard.html
├── static/             # ✅ Archivos estáticos
├── .gitignore          # ✅ Archivos a ignorar
├── README.md           # ✅ Documentación
└── app.json           # ✅ Configuración Heroku
```

## 🚀 Pasos para Subir a GitHub

### 1. Crear cuenta en GitHub
1. Ve a [github.com](https://github.com)
2. Regístrate con tu email
3. Confirma tu cuenta

### 2. Crear repositorio
1. Haz clic en "New repository"
2. Nombre: `crypto-predictor`
3. Descripción: "Predicción de criptomonedas con IA"
4. Marca como "Public"
5. **NO** inicialices con README (ya tenemos uno)
6. Haz clic en "Create repository"

### 3. Subir código
```bash
# Agregar remoto (reemplaza TU_USUARIO con tu nombre de usuario)
git remote add origin https://github.com/TU_USUARIO/crypto-predictor.git

# Subir código
git branch -M main
git push -u origin main
```

## 🔍 Verificación Post-Despliegue

### 1. Verificar que la app responda
- Accede a la URL proporcionada por la plataforma
- Deberías ver la página principal de CryptoPredictor

### 2. Probar funcionalidades principales
- ✅ Predicción de criptomonedas
- ✅ Comparación múltiple
- ✅ Sistema de alertas
- ✅ Dashboard avanzado

### 3. Revisar logs por errores
- Cada plataforma tiene su sección de logs
- Busca errores relacionados con dependencias

## 🐛 Solución de Problemas Comunes

### Error: "Build failed"
- Verifica que `requirements.txt` esté correcto
- Asegúrate de que todas las dependencias estén listadas

### Error: "Module not found"
- Verifica que `runtime.txt` especifique Python 3.12
- Asegúrate de que `Procfile` apunte a `app:app`

### Error: "Port already in use"
- Las plataformas de despliegue manejan esto automáticamente
- No necesitas configurar puertos manualmente

## 📞 Soporte

### Railway
- [Documentación](https://docs.railway.app/)
- [Discord](https://discord.gg/railway)

### Render
- [Documentación](https://render.com/docs)
- [Soporte](https://render.com/support)

### Heroku
- [Dev Center](https://devcenter.heroku.com/)
- [Soporte](https://help.heroku.com/)

## 🎉 ¡Tu CryptoPredictor en Línea!

Una vez desplegado, tu aplicación estará disponible en:
- **Railway**: `https://tu-app.railway.app`
- **Render**: `https://tu-app.onrender.com`
- **Heroku**: `https://tu-app.herokuapp.com`

### Funcionalidades Disponibles:
- 🔮 **Predicción de precios** con IA
- 📊 **Análisis técnico** avanzado
- 🔔 **Sistema de alertas** personalizado
- 💼 **Simulación de portafolio**
- 📈 **Comparación múltiple** de criptomonedas
- 🎯 **Dashboard profesional** con métricas en tiempo real

---

**⚠️ Descargo de Responsabilidad**: Esta aplicación es solo para fines educativos. No constituye consejo financiero.

**⭐ Si te gusta el proyecto, ¡dale una estrella en GitHub!** 