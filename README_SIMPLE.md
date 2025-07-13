# CryptoPredictor - Predicción de Criptomonedas con IA

Una aplicación web moderna para predecir precios de criptomonedas utilizando inteligencia artificial y análisis técnico.

## 🚀 Características

- **Predicción de Precios**: Modelo de machine learning para predecir precios futuros
- **Análisis Técnico**: Indicadores RSI, MACD, SMA y más
- **Comparación de Criptomonedas**: Compara múltiples criptomonedas con score de oportunidad
- **Simulador de Portafolio**: Simula inversiones y calcula ROI esperado
- **Alertas de Precio**: Configura alertas para precios específicos
- **Interfaz Moderna**: Diseño responsive con Tailwind CSS
- **Gráficos Interactivos**: Visualización de predicciones con Plotly

## 📋 Requisitos

- Python 3.8 o superior
- Conexión a internet (para obtener datos de Yahoo Finance)

## 🛠️ Instalación y Ejecución

### Opción 1: Ejecución Automática (Recomendada)

```bash
# Ejecutar el script automático
python run_simple.py
```

Este script:
- Instala automáticamente todas las dependencias
- Inicia la aplicación
- Abre el navegador automáticamente

### Opción 2: Instalación Manual

```bash
# 1. Instalar dependencias
pip install -r requirements_simple.txt

# 2. Ejecutar la aplicación
python app_simple.py
```

## 🌐 Acceso a la Aplicación

Una vez ejecutada, la aplicación estará disponible en:
- **URL Local**: http://127.0.0.1:5000
- **URL Red**: http://0.0.0.0:5000

## 📊 Funcionalidades Principales

### 1. Predicción de Precios
- Selecciona una criptomoneda
- Elige el número de días a predecir (1-30)
- Obtén predicciones basadas en análisis técnico y machine learning

### 2. Comparación de Criptomonedas
- Selecciona múltiples criptomonedas
- Compara métricas y scores de oportunidad
- Visualiza recomendaciones de inversión

### 3. Simulador de Portafolio
- Define tu inversión inicial
- Selecciona criptomonedas para tu portafolio
- Calcula ROI esperado y valor futuro

### 4. Alertas de Precio
- Configura alertas para precios específicos
- Recibe notificaciones cuando se alcancen los objetivos
- Monitoreo automático de precios

## 🎯 Criptomonedas Soportadas

- Bitcoin (BTC)
- Ethereum (ETH)
- Binance Coin (BNB)
- Cardano (ADA)
- Solana (SOL)
- Polkadot (DOT)
- Dogecoin (DOGE)
- Avalanche (AVAX)
- Chainlink (LINK)
- Litecoin (LTC)

## 🔧 Tecnologías Utilizadas

- **Backend**: Flask (Python)
- **Machine Learning**: Scikit-learn
- **Datos**: Yahoo Finance API
- **Frontend**: HTML, JavaScript, Tailwind CSS
- **Gráficos**: Plotly.js
- **Análisis Técnico**: Indicadores personalizados

## 📈 Métricas de Análisis

- **RSI (Relative Strength Index)**: Mide sobrecompra/sobreventa
- **MACD**: Momentum y tendencias
- **SMA (Simple Moving Average)**: Promedios móviles
- **Volumen**: Análisis de actividad de mercado
- **Cambio de Precio**: Variaciones porcentuales

## ⚠️ Advertencia

**IMPORTANTE**: Esta aplicación es solo para fines educativos y de investigación. Las predicciones no constituyen consejos financieros. Siempre realiza tu propia investigación antes de invertir.

## 🐛 Solución de Problemas

### Error de Módulo no encontrado
```bash
# Reinstalar dependencias
pip install --upgrade -r requirements_simple.txt
```

### Error de Conexión
- Verifica tu conexión a internet
- Asegúrate de que Yahoo Finance esté accesible

### Puerto en uso
```bash
# Cambiar puerto en app_simple.py línea final
app.run(debug=True, host='0.0.0.0', port=5001)
```

## 📝 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o pull request.

---

**¡Disfruta prediciendo el futuro de las criptomonedas! 🚀** 