from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Configuración simple
app.config['SECRET_KEY'] = 'tu_clave_secreta_aqui'
app.config['DATABASE_FILE'] = 'crypto_data.json'

# Base de datos simple en JSON
def load_data():
    if os.path.exists(app.config['DATABASE_FILE']):
        with open(app.config['DATABASE_FILE'], 'r') as f:
            return json.load(f)
    return {'predictions': [], 'alerts': [], 'portfolio': []}

def save_data(data):
    with open(app.config['DATABASE_FILE'], 'w') as f:
        json.dump(data, f, indent=2, default=str)

# Funciones de análisis técnico básico
def calculate_sma(data, window=20):
    return data['Close'].rolling(window=window).mean()

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data):
    exp1 = data['Close'].ewm(span=12).mean()
    exp2 = data['Close'].ewm(span=26).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9).mean()
    return macd, signal

def get_crypto_data(symbol, days=365):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=f"{days}d")
        if data.empty:
            return None
        return data
    except Exception as e:
        print(f"Error obteniendo datos para {symbol}: {e}")
        return None

def prepare_features(data):
    if data is None or len(data) < 30:
        return None, None
    
    # Características básicas
    data['SMA_20'] = calculate_sma(data, 20)
    data['SMA_50'] = calculate_sma(data, 50)
    data['RSI'] = calculate_rsi(data)
    data['MACD'], data['MACD_Signal'] = calculate_macd(data)
    
    # Características adicionales
    data['Price_Change'] = data['Close'].pct_change()
    data['Volume_Change'] = data['Volume'].pct_change()
    data['High_Low_Ratio'] = data['High'] / data['Low']
    data['Close_Open_Ratio'] = data['Close'] / data['Open']
    
    # Eliminar filas con valores NaN
    data = data.dropna()
    
    if len(data) < 30:
        return None, None
    
    # Preparar características para el modelo
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 
               'RSI', 'MACD', 'MACD_Signal', 'Price_Change', 'Volume_Change',
               'High_Low_Ratio', 'Close_Open_Ratio']
    
    X = data[features].values
    y = data['Close'].values
    
    return X, y

def train_model(X, y):
    if X is None or y is None or len(X) < 30:
        return None, None
    
    # Dividir datos
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Escalar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

def predict_future(model, scaler, last_data, days=7):
    if model is None or scaler is None:
        return None
    
    predictions = []
    current_data = last_data.copy()
    
    for _ in range(days):
        # Preparar características para predicción
        features = current_data.reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        # Predecir
        pred = model.predict(features_scaled)[0]
        predictions.append(pred)
        
        # Actualizar datos para siguiente predicción
        current_data = np.roll(current_data, -1)
        current_data[-1] = pred
    
    return predictions

def get_investment_advice(current_price, predicted_prices, rsi, macd):
    if not predicted_prices:
        return "No hay suficientes datos para dar consejos"
    
    avg_prediction = np.mean(predicted_prices)
    price_change = ((avg_prediction - current_price) / current_price) * 100
    
    advice = []
    
    if price_change > 5:
        advice.append("🟢 TENDENCIA ALCISTA: Se espera un aumento significativo")
    elif price_change > 2:
        advice.append("🟡 TENDENCIA POSITIVA: Se espera un aumento moderado")
    elif price_change < -5:
        advice.append("🔴 TENDENCIA BAJISTA: Se espera una caída significativa")
    elif price_change < -2:
        advice.append("🟠 TENDENCIA NEGATIVA: Se espera una caída moderada")
    else:
        advice.append("⚪ TENDENCIA LATERAL: Se espera estabilidad")
    
    if rsi > 70:
        advice.append("⚠️ SOBRECOMPRADO: RSI alto, posible corrección")
    elif rsi < 30:
        advice.append("💡 SOBREVENDIDO: RSI bajo, posible rebote")
    
    if macd > 0:
        advice.append("📈 MACD positivo: Momentum alcista")
    else:
        advice.append("📉 MACD negativo: Momentum bajista")
    
    return " | ".join(advice)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/popular_cryptos')
def popular_cryptos():
    cryptos = [
        {'symbol': 'BTC-USD', 'name': 'Bitcoin', 'color': '#f7931a'},
        {'symbol': 'ETH-USD', 'name': 'Ethereum', 'color': '#627eea'},
        {'symbol': 'BNB-USD', 'name': 'Binance Coin', 'color': '#f3ba2f'},
        {'symbol': 'ADA-USD', 'name': 'Cardano', 'color': '#0033ad'},
        {'symbol': 'SOL-USD', 'name': 'Solana', 'color': '#14f195'},
        {'symbol': 'DOT-USD', 'name': 'Polkadot', 'color': '#e6007a'},
        {'symbol': 'DOGE-USD', 'name': 'Dogecoin', 'color': '#c2a633'},
        {'symbol': 'AVAX-USD', 'name': 'Avalanche', 'color': '#e84142'},
        {'symbol': 'LINK-USD', 'name': 'Chainlink', 'color': '#2a5ada'},
        {'symbol': 'LTC-USD', 'name': 'Litecoin', 'color': '#a6a9aa'}
    ]
    return jsonify(cryptos)

@app.route('/api/market_overview')
def market_overview():
    try:
        cryptos = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD']
        overview = []
        
        for symbol in cryptos:
            data = get_crypto_data(symbol, 30)
            if data is not None and not data.empty:
                current_price = data['Close'].iloc[-1]
                prev_price = data['Close'].iloc[-2]
                change_24h = ((current_price - prev_price) / prev_price) * 100
                
                overview.append({
                    'symbol': symbol,
                    'price': round(current_price, 2),
                    'change_24h': round(change_24h, 2),
                    'volume': round(data['Volume'].iloc[-1], 0)
                })
        
        return jsonify(overview)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'BTC-USD')
        days = int(data.get('days', 7))
        
        # Obtener datos
        crypto_data = get_crypto_data(symbol, 365)
        if crypto_data is None or crypto_data.empty:
            return jsonify({'error': 'No se pudieron obtener datos para esta criptomoneda'}), 400
        
        # Preparar características
        X, y = prepare_features(crypto_data)
        if X is None:
            return jsonify({'error': 'Datos insuficientes para el análisis'}), 400
        
        # Entrenar modelo
        model, scaler = train_model(X, y)
        if model is None:
            return jsonify({'error': 'Error al entrenar el modelo'}), 400
        
        # Predecir
        last_features = X[-1]
        predictions = predict_future(model, scaler, last_features, days)
        
        if predictions is None:
            return jsonify({'error': 'Error en la predicción'}), 400
        
        # Calcular métricas
        current_price = crypto_data['Close'].iloc[-1]
        rsi = crypto_data['RSI'].iloc[-1] if 'RSI' in crypto_data.columns else 50
        macd = crypto_data['MACD'].iloc[-1] if 'MACD' in crypto_data.columns else 0
        
        # Consejos de inversión
        advice = get_investment_advice(current_price, predictions, rsi, macd)
        
        # Guardar predicción
        db_data = load_data()
        prediction_record = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'current_price': float(current_price),
            'predictions': [float(p) for p in predictions],
            'advice': advice
        }
        db_data['predictions'].append(prediction_record)
        save_data(db_data)
        
        # Preparar respuesta
        dates = [(datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(days)]
        
        response = {
            'symbol': symbol,
            'current_price': float(current_price),
            'predictions': [float(p) for p in predictions],
            'dates': dates,
            'advice': advice,
            'metrics': {
                'rsi': float(rsi),
                'macd': float(macd),
                'sma_20': float(crypto_data['SMA_20'].iloc[-1]) if 'SMA_20' in crypto_data.columns else 0,
                'sma_50': float(crypto_data['SMA_50'].iloc[-1]) if 'SMA_50' in crypto_data.columns else 0
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Error en la predicción: {str(e)}'}), 500

@app.route('/api/prediction_history/<symbol>')
def prediction_history(symbol):
    try:
        db_data = load_data()
        history = [p for p in db_data['predictions'] if p['symbol'] == symbol]
        return jsonify(history[-10:])  # Últimas 10 predicciones
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare_cryptos', methods=['POST'])
def compare_cryptos():
    try:
        data = request.get_json()
        symbols = data.get('symbols', [])
        
        if not symbols:
            return jsonify({'error': 'No se proporcionaron símbolos'}), 400
        
        comparison = []
        
        for symbol in symbols:
            crypto_data = get_crypto_data(symbol, 30)
            if crypto_data is not None and not crypto_data.empty:
                current_price = crypto_data['Close'].iloc[-1]
                prev_price = crypto_data['Close'].iloc[-7]  # Precio hace 7 días
                change_7d = ((current_price - prev_price) / prev_price) * 100
                
                # Calcular score de oportunidad
                rsi = crypto_data['RSI'].iloc[-1] if 'RSI' in crypto_data.columns else 50
                macd = crypto_data['MACD'].iloc[-1] if 'MACD' in crypto_data.columns else 0
                
                # Score basado en RSI y cambio de precio
                score = 0
                if rsi < 30:  # Sobrevendido
                    score += 30
                elif rsi > 70:  # Sobrecomprado
                    score -= 20
                
                if change_7d > 5:
                    score += 20
                elif change_7d < -5:
                    score -= 10
                
                if macd > 0:
                    score += 10
                else:
                    score -= 10
                
                comparison.append({
                    'symbol': symbol,
                    'current_price': float(current_price),
                    'change_7d': float(change_7d),
                    'rsi': float(rsi),
                    'macd': float(macd),
                    'opportunity_score': int(score)
                })
        
        # Ordenar por score de oportunidad
        comparison.sort(key=lambda x: x['opportunity_score'], reverse=True)
        
        return jsonify(comparison)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio_simulation', methods=['POST'])
def portfolio_simulation():
    try:
        data = request.get_json()
        initial_investment = float(data.get('initial_investment', 10000))
        symbols = data.get('symbols', ['BTC-USD', 'ETH-USD'])
        allocation = data.get('allocation', {})
        
        if not allocation:
            # Distribución equitativa
            allocation = {symbol: 100 / len(symbols) for symbol in symbols}
        
        portfolio_data = []
        total_value = 0
        
        for symbol in symbols:
            crypto_data = get_crypto_data(symbol, 30)
            if crypto_data is not None and not crypto_data.empty:
                current_price = crypto_data['Close'].iloc[-1]
                allocation_percent = allocation.get(symbol, 100 / len(symbols))
                investment_amount = (initial_investment * allocation_percent) / 100
                coins = investment_amount / current_price
                
                # Simular valor futuro (simplificado)
                future_price = current_price * 1.05  # +5% estimado
                future_value = coins * future_price
                
                portfolio_data.append({
                    'symbol': symbol,
                    'current_price': float(current_price),
                    'coins': float(coins),
                    'current_value': float(investment_amount),
                    'future_value': float(future_value),
                    'allocation': float(allocation_percent)
                })
                
                total_value += future_value
        
        # Guardar simulación
        db_data = load_data()
        simulation_record = {
            'timestamp': datetime.now().isoformat(),
            'initial_investment': initial_investment,
            'portfolio': portfolio_data,
            'total_future_value': float(total_value),
            'roi': float(((total_value - initial_investment) / initial_investment) * 100)
        }
        db_data['portfolio'].append(simulation_record)
        save_data(db_data)
        
        return jsonify({
            'portfolio': portfolio_data,
            'total_future_value': float(total_value),
            'roi': float(((total_value - initial_investment) / initial_investment) * 100)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/set_alert', methods=['POST'])
def set_alert():
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        target_price = float(data.get('target_price'))
        alert_type = data.get('alert_type', 'above')  # above/below
        
        if not symbol or not target_price:
            return jsonify({'error': 'Datos incompletos'}), 400
        
        db_data = load_data()
        alert = {
            'symbol': symbol,
            'target_price': target_price,
            'alert_type': alert_type,
            'created_at': datetime.now().isoformat(),
            'active': True
        }
        
        db_data['alerts'].append(alert)
        save_data(db_data)
        
        return jsonify({'message': 'Alerta configurada exitosamente'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_alerts/<symbol>')
def get_alerts(symbol):
    try:
        db_data = load_data()
        alerts = [a for a in db_data['alerts'] if a['symbol'] == symbol and a['active']]
        return jsonify(alerts)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/check_alerts')
def check_alerts():
    try:
        db_data = load_data()
        active_alerts = [a for a in db_data['alerts'] if a['active']]
        triggered_alerts = []
        
        for alert in active_alerts:
            crypto_data = get_crypto_data(alert['symbol'], 1)
            if crypto_data is not None and not crypto_data.empty:
                current_price = crypto_data['Close'].iloc[-1]
                
                if alert['alert_type'] == 'above' and current_price >= alert['target_price']:
                    triggered_alerts.append({
                        'symbol': alert['symbol'],
                        'target_price': alert['target_price'],
                        'current_price': float(current_price),
                        'type': 'above'
                    })
                    alert['active'] = False
                elif alert['alert_type'] == 'below' and current_price <= alert['target_price']:
                    triggered_alerts.append({
                        'symbol': alert['symbol'],
                        'target_price': alert['target_price'],
                        'current_price': float(current_price),
                        'type': 'below'
                    })
                    alert['active'] = False
        
        if triggered_alerts:
            save_data(db_data)
        
        return jsonify(triggered_alerts)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 