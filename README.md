# 🚀 CryptoPredictor - Predicción de Criptomonedas con IA

Una aplicación web completa y profesional para predecir precios de criptomonedas usando inteligencia artificial y análisis técnico avanzado.

## 🌟 Características Principales

### 📊 **Predicción Inteligente**
- Modelo de Machine Learning (Random Forest)
- Análisis técnico avanzado
- Predicciones a 7 días
- Historial de predicciones

### 📈 **Análisis Avanzado**
- Comparación múltiple de criptomonedas
- Score de oportunidad de inversión
- Análisis de sentimiento de noticias
- Backtesting de estrategias

### 🔔 **Sistema de Alertas**
- Alertas de precio personalizadas
- Notificaciones en tiempo real
- Alertas técnicas avanzadas
- Verificación automática

### 💼 **Gestión de Portafolio**
- Simulación de portafolio virtual
- Análisis de riesgo
- Recomendaciones de inversión
- Seguimiento de rendimiento

### 📊 **Dashboard Profesional**
- Interfaz moderna con Tailwind CSS
- Gráficos interactivos con Plotly
- Vista general del mercado
- Métricas en tiempo real

## 🛠️ Instalación Local

### Prerrequisitos
- Python 3.12+
- pip (gestor de paquetes)

### Pasos de Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/tu-usuario/crypto-predictor.git
cd crypto-predictor
```

2. **Crear entorno virtual**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Ejecutar la aplicación**
```bash
python app.py
```

5. **Abrir en el navegador**
```
http://localhost:5000
```

## 🚀 Despliegue en la Web

### Opción 1: Railway (Recomendado)

1. **Crear cuenta en Railway**
   - Ve a [railway.app](https://railway.app)
   - Regístrate con tu cuenta de GitHub

2. **Conectar repositorio**
   - Haz clic en "New Project"
   - Selecciona "Deploy from GitHub repo"
   - Conecta tu repositorio de GitHub

3. **Configurar variables de entorno**
   - Ve a la pestaña "Variables"
   - Agrega: `FLASK_ENV=production`

4. **Desplegar**
   - Railway detectará automáticamente que es una app Flask
   - Se desplegará automáticamente

### Opción 2: Render

1. **Crear cuenta en Render**
   - Ve a [render.com](https://render.com)
   - Regístrate con tu cuenta de GitHub

2. **Crear nuevo servicio**
   - Selecciona "Web Service"
   - Conecta tu repositorio de GitHub

3. **Configurar**
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Environment**: `FLASK_ENV=production`

### Opción 3: Heroku

1. **Instalar Heroku CLI**
```bash
# Descargar desde heroku.com
```

2. **Login y crear app**
```bash
heroku login
heroku create tu-app-name
```

3. **Desplegar**
```bash
git add .
git commit -m "Initial deployment"
git push heroku main
```

## 📁 Estructura del Proyecto

```
crypto-predictor/
├── app.py                 # Aplicación principal Flask
├── requirements.txt       # Dependencias Python
├── Procfile             # Configuración para Heroku
├── runtime.txt          # Versión de Python
├── wsgi.py             # Servidor WSGI
├── config.py           # Configuración de la app
├── templates/          # Plantillas HTML
│   ├── index.html      # Página principal
│   └── advanced_dashboard.html
├── static/             # Archivos estáticos
│   ├── css/
│   ├── js/
│   └── images/
└── README.md           # Este archivo
```

## 🔧 Configuración

### Variables de Entorno

```bash
# Desarrollo
FLASK_ENV=development
SECRET_KEY=tu-clave-secreta

# Producción
FLASK_ENV=production
SECRET_KEY=clave-secreta-produccion
```

### Personalización

- **Criptomonedas**: Edita `POPULAR_CRYPTOS` en `app.py`
- **Modelo ML**: Modifica `CryptoPredictor` para cambiar el algoritmo
- **Interfaz**: Edita las plantillas HTML en `templates/`

## 📊 API Endpoints

### Predicción
- `POST /predict` - Predicción de precio
- `GET /api/popular_cryptos` - Lista de criptomonedas
- `POST /api/compare_cryptos` - Comparación múltiple

### Alertas
- `POST /api/set_alert` - Configurar alerta
- `GET /api/check_alerts` - Verificar alertas
- `DELETE /api/delete_alert` - Eliminar alerta

### Análisis Avanzado
- `GET /api/sentiment_analysis/<symbol>` - Análisis de sentimiento
- `GET /api/technical_analysis/<symbol>` - Análisis técnico
- `POST /api/backtest/<symbol>/<strategy>` - Backtesting

## 🎯 Uso de la Aplicación

### 1. **Predicción Básica**
1. Selecciona una criptomoneda
2. Haz clic en "Predecir"
3. Revisa las predicciones y consejos

### 2. **Comparación Múltiple**
1. Selecciona hasta 5 criptomonedas
2. Haz clic en "Comparar"
3. Analiza el ranking por oportunidad

### 3. **Configurar Alertas**
1. Ve a la sección "Alertas"
2. Establece precio objetivo
3. Recibe notificaciones automáticas

### 4. **Dashboard Avanzado**
1. Accede a `/advanced_dashboard`
2. Explora análisis técnico
3. Prueba estrategias de backtesting

## 🔒 Seguridad

- ✅ Validación de entrada
- ✅ Sanitización de datos
- ✅ Manejo de errores
- ✅ Rate limiting básico
- ✅ Variables de entorno seguras

## 🐛 Solución de Problemas

### Error: "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### Error: "Port already in use"
```bash
# Cambiar puerto en app.py
app.run(port=5001)
```

### Error: "JSON serialization"
- Los datos se convierten automáticamente a tipos nativos de Python

## 📈 Roadmap

- [ ] Autenticación de usuarios
- [ ] Base de datos PostgreSQL
- [ ] Más modelos de ML
- [ ] API pública
- [ ] App móvil
- [ ] Integración con exchanges

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama: `git checkout -b feature/nueva-funcionalidad`
3. Commit: `git commit -m 'Agregar nueva funcionalidad'`
4. Push: `git push origin feature/nueva-funcionalidad`
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 📞 Soporte

- **Issues**: [GitHub Issues](https://github.com/tu-usuario/crypto-predictor/issues)
- **Email**: soporte@cryptopredictor.com
- **Documentación**: [Wiki del proyecto](https://github.com/tu-usuario/crypto-predictor/wiki)

---

**⚠️ Descargo de Responsabilidad**: Esta aplicación es solo para fines educativos. No constituye consejo financiero. Siempre haz tu propia investigación antes de invertir.

**⭐ Si te gusta el proyecto, ¡dale una estrella en GitHub!** 