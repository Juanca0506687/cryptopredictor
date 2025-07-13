# 🚀 CryptoPredictor AI - Plataforma de Predicción de Criptomonedas

Una plataforma avanzada de inteligencia artificial para análisis y predicción de criptomonedas con múltiples funcionalidades.

## ✨ Características

### 🔮 **Predicción con IA**
- Modelos de Machine Learning avanzados (Random Forest, Regresión Lineal, SVR, Red Neuronal)
- Análisis técnico completo (SMA, EMA, Bollinger Bands, MACD, RSI)
- Predicciones a 7 días con gráficos interactivos

### 📊 **Análisis Avanzado**
- **Análisis de Sentimiento**: Evaluación del sentimiento de noticias relacionadas
- **Backtesting de Estrategias**: Prueba de estrategias (Buy & Hold, Medias Móviles, RSI, Momentum)
- **Comparación Múltiple**: Análisis simultáneo de hasta 5 criptomonedas
- **Dashboard en Tiempo Real**: Vista general del mercado

### 🔔 **Sistema de Alertas**
- Alertas de precio personalizables
- Alertas de volumen inusual
- Alertas técnicas (RSI, MACD)
- Alertas de sentimiento
- Notificaciones automáticas

### 💼 **Gestión de Portafolio**
- Simulación de portafolio virtual
- Cálculo de ganancias/pérdidas
- Registro de usuarios
- Guardado de portafolios personalizados

### 📈 **Exportación y Reportes**
- Exportación en formato JSON y CSV
- Reportes completos del mercado
- Historial de predicciones

## 🛠️ Instalación

### Requisitos
- Python 3.8+
- pip

### Pasos de Instalación

1. **Clonar el repositorio**
```bash
git clone <tu-repositorio>
cd crypto-predictor-ai
```

2. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

3. **Ejecutar la aplicación**
```bash
python app.py
```

4. **Acceder a la aplicación**
- Página principal: http://localhost:5000
- Dashboard avanzado: http://localhost:5000/advanced_dashboard

## 🌐 Despliegue en la Web

### Opción 1: Render (Recomendado)

1. **Crear cuenta en Render**
   - Ve a [render.com](https://render.com)
   - Regístrate con tu cuenta de GitHub

2. **Conectar repositorio**
   - Haz clic en "New Web Service"
   - Conecta tu repositorio de GitHub
   - Selecciona el repositorio

3. **Configurar el servicio**
   - **Name**: crypto-predictor-ai
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`

4. **Desplegar**
   - Haz clic en "Create Web Service"
   - Espera a que se complete el despliegue
   - Tu sitio estará disponible en: `https://tu-app.onrender.com`

### Opción 2: Heroku

1. **Instalar Heroku CLI**
```bash
# Windows
# Descarga desde: https://devcenter.heroku.com/articles/heroku-cli

# Mac
brew install heroku/brew/heroku
```

2. **Login y crear app**
```bash
heroku login
heroku create tu-app-name
```

3. **Desplegar**
```bash
git add .
git commit -m "Initial commit"
git push heroku main
```

4. **Abrir la aplicación**
```bash
heroku open
```

## 📱 Uso de la Aplicación

### Página Principal (`/`)
- **Predicción Básica**: Ingresa un símbolo (ej: BTC) y obtén predicciones
- **Comparación**: Selecciona hasta 5 criptomonedas para comparar
- **Vista del Mercado**: Resumen general del mercado crypto
- **Alertas**: Configura alertas de precio
- **Portafolio**: Simula inversiones virtuales

### Dashboard Avanzado (`/advanced_dashboard`)
- **Análisis de Sentimiento**: Evalúa el sentimiento de noticias
- **Backtesting**: Prueba estrategias de trading
- **Análisis Técnico**: Indicadores avanzados y señales
- **Predicción ML**: Múltiples modelos de machine learning
- **Exportación**: Descarga reportes en diferentes formatos
- **Alertas Avanzadas**: Configuración detallada de alertas
- **Gestión de Usuarios**: Registro y portafolios personalizados

## 🔧 APIs Disponibles

### Predicción y Análisis
- `POST /predict` - Predicción básica
- `GET /api/advanced_prediction/<symbol>` - Predicción avanzada
- `GET /api/technical_analysis/<symbol>` - Análisis técnico
- `GET /api/sentiment_analysis/<symbol>` - Análisis de sentimiento

### Backtesting
- `GET /api/backtest/<symbol>/<strategy>` - Ejecutar backtest

### Mercado y Comparación
- `GET /api/market_overview` - Vista general del mercado
- `POST /api/compare_cryptos` - Comparación múltiple
- `GET /api/popular_cryptos` - Criptomonedas populares

### Alertas y Notificaciones
- `POST /api/advanced_alert` - Configurar alerta avanzada
- `GET /api/check_advanced_alerts` - Verificar alertas
- `GET /api/notifications/<user_id>` - Obtener notificaciones

### Usuarios y Portafolio
- `POST /api/register_user` - Registrar usuario
- `POST /api/save_portfolio` - Guardar portafolio
- `POST /api/portfolio_simulation` - Simular portafolio

### Exportación
- `POST /api/export_report` - Exportar reportes
- `GET /api/dashboard_data` - Datos del dashboard

## 🎯 Ejemplos de Uso

### Predicción de Bitcoin
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTC"}'
```

### Análisis Técnico de Ethereum
```bash
curl http://localhost:5000/api/technical_analysis/ETH
```

### Backtesting de Estrategia
```bash
curl http://localhost:5000/api/backtest/BTC/moving_average
```

## 🛡️ Seguridad

- Validación de entrada en todas las APIs
- Manejo de errores robusto
- Sanitización de datos
- Rate limiting recomendado para producción

## 📊 Tecnologías Utilizadas

- **Backend**: Flask, Python
- **Machine Learning**: Scikit-learn, NumPy, Pandas
- **Datos**: Yahoo Finance API
- **Frontend**: HTML5, CSS3, JavaScript, Tailwind CSS
- **Gráficos**: Plotly
- **Análisis**: TextBlob (sentimiento)

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 📞 Soporte

Si tienes problemas o preguntas:
- Abre un issue en GitHub
- Contacta: [tu-email@ejemplo.com]

## 🚀 Roadmap

- [ ] Integración con más exchanges
- [ ] Análisis de on-chain data
- [ ] Alertas por email/SMS
- [ ] App móvil
- [ ] Más modelos de ML
- [ ] Análisis de correlaciones
- [ ] Backtesting con más estrategias

---

**¡Disfruta usando CryptoPredictor AI! 🎉** 