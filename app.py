from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objs as go
import plotly.utils
import json
from datetime import datetime, timedelta
import warnings
import os
import csv
import io
import zipfile
from textblob import TextBlob
import requests
from io import BytesIO
import base64
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import pickle
import threading
import time
import schedule
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Lista de criptomonedas populares
POPULAR_CRYPTOS = [
    {'symbol': 'BTC', 'name': 'Bitcoin', 'category': 'Store of Value'},
    {'symbol': 'ETH', 'name': 'Ethereum', 'category': 'Smart Contracts'},
    {'symbol': 'BNB', 'name': 'Binance Coin', 'category': 'Exchange'},
    {'symbol': 'ADA', 'name': 'Cardano', 'category': 'Smart Contracts'},
    {'symbol': 'SOL', 'name': 'Solana', 'category': 'Smart Contracts'},
    {'symbol': 'XRP', 'name': 'Ripple', 'category': 'Payments'},
    {'symbol': 'DOT', 'name': 'Polkadot', 'category': 'Interoperability'},
    {'symbol': 'DOGE', 'name': 'Dogecoin', 'category': 'Meme'},
    {'symbol': 'AVAX', 'name': 'Avalanche', 'category': 'Smart Contracts'},
    {'symbol': 'MATIC', 'name': 'Polygon', 'category': 'Scaling'},
    {'symbol': 'LINK', 'name': 'Chainlink', 'category': 'Oracle'},
    {'symbol': 'UNI', 'name': 'Uniswap', 'category': 'DeFi'},
    {'symbol': 'LTC', 'name': 'Litecoin', 'category': 'Payments'},
    {'symbol': 'BCH', 'name': 'Bitcoin Cash', 'category': 'Payments'},
    {'symbol': 'ATOM', 'name': 'Cosmos', 'category': 'Interoperability'},
    {'symbol': 'FTT', 'name': 'FTX Token', 'category': 'Exchange'},
    {'symbol': 'NEAR', 'name': 'NEAR Protocol', 'category': 'Smart Contracts'},
    {'symbol': 'ALGO', 'name': 'Algorand', 'category': 'Smart Contracts'},
    {'symbol': 'VET', 'name': 'VeChain', 'category': 'Supply Chain'},
    {'symbol': 'FIL', 'name': 'Filecoin', 'category': 'Storage'}
]

class CryptoPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        self.prediction_history = {}  # Para guardar historial
        
    def get_crypto_data(self, symbol, period='1y'):
        try:
            ticker = yf.Ticker(f"{symbol}-USD")
            data = ticker.history(period=period)
            return data
        except Exception as e:
            print(f"Error obteniendo datos: {e}")
            return None
    
    def prepare_features(self, data):
        df = data.copy()
        # Solo características básicas
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df = df.dropna()
        return df
    
    def train_model(self, symbol):
        data = self.get_crypto_data(symbol)
        if data is None or len(data) < 100:
            return False
        df = self.prepare_features(data)
        feature_columns = ['Price_Change', 'Volume_Change', 'High_Low_Ratio', 'Day_of_Week', 'Month']
        X = df[feature_columns].values
        y = df['Close'].values
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        return True
    
    def predict_next_days(self, symbol, days=7):
        if not self.is_trained:
            if not self.train_model(symbol):
                return None
        data = self.get_crypto_data(symbol, period='3mo')
        if data is None:
            return None
        df = self.prepare_features(data)
        feature_columns = ['Price_Change', 'Volume_Change', 'High_Low_Ratio', 'Day_of_Week', 'Month']
        last_features = df[feature_columns].iloc[-1:].values
        last_features_scaled = self.scaler.transform(last_features)
        predictions = []
        current_price = df['Close'].iloc[-1]
        for i in range(days):
            pred = self.model.predict(last_features_scaled)[0]
            predictions.append(pred)
            last_features_scaled[0][0] = (pred - current_price) / current_price  # Price_Change
            current_price = pred
        return predictions
    
    def get_investment_advice(self, symbol):
        data = self.get_crypto_data(symbol, period='1mo')
        if data is None:
            return "No se pudieron obtener datos para análisis"
        df = self.prepare_features(data)
        current_price = df['Close'].iloc[-1]
        price_change = df['Price_Change'].iloc[-1]
        volume_change = df['Volume_Change'].iloc[-1]
        signals = []
        if price_change > 0.01:
            signals.append("El precio sube con fuerza")
        elif price_change < -0.01:
            signals.append("El precio baja con fuerza")
        else:
            signals.append("El precio está estable")
        if volume_change > 0.05:
            signals.append("Aumento de volumen significativo")
        elif volume_change < -0.05:
            signals.append("Caída de volumen significativa")
        else:
            signals.append("Volumen estable")
        bullish_signals = sum([
            price_change > 0,
            volume_change > 0
        ])
        if bullish_signals == 2:
            recommendation = "COMPRAR"
            confidence = "Media"
        elif bullish_signals == 1:
            recommendation = "MANTENER"
            confidence = "Baja"
        else:
            recommendation = "VENDER"
            confidence = "Media"
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'signals': signals,
            'current_price': current_price,
            'price_change': price_change,
            'volume_change': volume_change
        }

    def save_prediction(self, symbol, predictions, advice):
        """Guardar predicción en el historial"""
        timestamp = datetime.now().isoformat()
        if symbol not in self.prediction_history:
            self.prediction_history[symbol] = []
        
        self.prediction_history[symbol].append({
            'timestamp': timestamp,
            'predictions': predictions,
            'advice': advice
        })
        
        # Mantener solo las últimas 10 predicciones por símbolo
        if len(self.prediction_history[symbol]) > 10:
            self.prediction_history[symbol] = self.prediction_history[symbol][-10:]

predictor = CryptoPredictor()

# Sistema de alertas
alerts = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/advanced_dashboard')
def advanced_dashboard():
    return render_template('advanced_dashboard.html')

@app.route('/api/popular_cryptos')
def get_popular_cryptos():
    """Obtener lista de criptomonedas populares"""
    return jsonify(POPULAR_CRYPTOS)

@app.route('/api/search_crypto')
def search_crypto():
    """Buscar criptomonedas por nombre o símbolo"""
    query = request.args.get('q', '').upper()
    if not query:
        return jsonify([])
    
    results = []
    for crypto in POPULAR_CRYPTOS:
        if query in crypto['symbol'] or query in crypto['name'].upper():
            results.append(crypto)
    
    return jsonify(results[:10])  # Máximo 10 resultados

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symbol = data.get('symbol', 'BTC').upper()
    try:
        if not predictor.train_model(symbol):
            return jsonify({'error': 'No se pudieron obtener datos para entrenar el modelo'})
        predictions = predictor.predict_next_days(symbol, 7)
        if predictions is None:
            return jsonify({'error': 'Error en la predicción'})
        advice = predictor.get_investment_advice(symbol)
        
        # Guardar predicción en historial
        predictor.save_prediction(symbol, predictions, advice)
        
        dates = pd.date_range(start=datetime.now(), periods=8, freq='D')[1:]
        trace = go.Scatter(
            x=dates,
            y=predictions,
            mode='lines+markers',
            name='Predicción',
            line=dict(color='blue', width=2)
        )
        layout = go.Layout(
            title=f'Predicción de Precio - {symbol}',
            xaxis=dict(title='Fecha'),
            yaxis=dict(title='Precio (USD)'),
            template='plotly_white'
        )
        fig = go.Figure(data=[trace], layout=layout)
        plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify({
            'predictions': predictions,
            'dates': [d.strftime('%Y-%m-%d') for d in dates],
            'plot': plot_json,
            'advice': advice
        })
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'})

@app.route('/get_crypto_info', methods=['POST'])
def get_crypto_info():
    data = request.get_json()
    symbol = data.get('symbol', 'BTC').upper()
    try:
        ticker = yf.Ticker(f"{symbol}-USD")
        info = ticker.info
        current_price = ticker.history(period='1d')['Close'].iloc[-1]
        return jsonify({
            'name': info.get('longName', symbol),
            'current_price': current_price,
            'market_cap': info.get('marketCap', 'N/A'),
            'volume': info.get('volume', 'N/A'),
            'change_24h': info.get('regularMarketChangePercent', 'N/A')
        })
    except Exception as e:
        return jsonify({'error': f'Error obteniendo información: {str(e)}'})

@app.route('/api/prediction_history/<symbol>')
def get_prediction_history(symbol):
    """Obtener historial de predicciones para un símbolo"""
    symbol = symbol.upper()
    if symbol in predictor.prediction_history:
        return jsonify(predictor.prediction_history[symbol])
    return jsonify([])

@app.route('/api/compare_cryptos', methods=['POST'])
def compare_cryptos():
    """Comparar múltiples criptomonedas"""
    data = request.get_json()
    symbols = data.get('symbols', [])
    
    if not symbols or len(symbols) > 5:  # Máximo 5 criptomonedas
        return jsonify({'error': 'Debes seleccionar entre 1 y 5 criptomonedas'})
    
    results = []
    for symbol in symbols:
        try:
            # Obtener información básica
            ticker = yf.Ticker(f"{symbol}-USD")
            info = ticker.info
            current_price = float(ticker.history(period='1d')['Close'].iloc[-1])
            
            # Obtener predicción
            if not predictor.train_model(symbol):
                continue
                
            predictions = predictor.predict_next_days(symbol, 7)
            if predictions is None:
                continue
                
            advice = predictor.get_investment_advice(symbol)
            
            # Calcular score de oportunidad
            opportunity_score = calculate_opportunity_score(advice, predictions)
            
            results.append({
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'current_price': float(current_price),
                'predicted_price': float(predictions[-1]) if predictions else float(current_price),
                'price_change_7d': float(((predictions[-1] - current_price) / current_price * 100)) if predictions else 0.0,
                'recommendation': advice['recommendation'],
                'confidence': advice['confidence'],
                'opportunity_score': int(opportunity_score),
                'market_cap': float(info.get('marketCap', 0)),
                'volume': float(info.get('volume', 0))
            })
            
        except Exception as e:
            print(f"Error analizando {symbol}: {e}")
            continue
    
    # Ordenar por score de oportunidad
    results.sort(key=lambda x: x['opportunity_score'], reverse=True)
    
    return jsonify({
        'comparison': results,
        'summary': {
            'total_analyzed': len(results),
            'buy_recommendations': len([r for r in results if r['recommendation'] == 'COMPRAR']),
            'sell_recommendations': len([r for r in results if r['recommendation'] == 'VENDER']),
            'hold_recommendations': len([r for r in results if r['recommendation'] == 'MANTENER'])
        }
    })

def calculate_opportunity_score(advice, predictions):
    """Calcular score de oportunidad basado en múltiples factores"""
    score = 0
    
    # Factor de recomendación
    if advice['recommendation'] == 'COMPRAR':
        score += 30
    elif advice['recommendation'] == 'MANTENER':
        score += 10
    else:  # VENDER
        score += 0
    
    # Factor de confianza
    if advice['confidence'] == 'Alta':
        score += 20
    elif advice['confidence'] == 'Media':
        score += 15
    else:  # Baja
        score += 5
    
    # Factor de cambio de precio proyectado
    if predictions and len(predictions) > 0:
        current_price = float(advice['current_price'])
        predicted_price = float(predictions[-1])
        price_change_pct = (predicted_price - current_price) / current_price * 100
        
        if price_change_pct > 10:
            score += 25
        elif price_change_pct > 5:
            score += 15
        elif price_change_pct > 0:
            score += 10
        else:
            score += 0
    
    # Factor de señales técnicas
    bullish_signals = sum([
        float(advice['price_change']) > 0,
        float(advice['volume_change']) > 0
    ])
    score += bullish_signals * 10
    
    return min(score, 100)  # Máximo 100

@app.route('/api/market_overview')
def get_market_overview():
    """Obtener vista general del mercado"""
    try:
        # Analizar las top 10 criptomonedas
        top_symbols = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'XRP', 'DOT', 'DOGE', 'AVAX', 'MATIC']
        
        market_data = []
        for symbol in top_symbols:
            try:
                ticker = yf.Ticker(f"{symbol}-USD")
                info = ticker.info
                current_price = ticker.history(period='1d')['Close'].iloc[-1]
                change_24h = info.get('regularMarketChangePercent', 0)
                
                market_data.append({
                    'symbol': symbol,
                    'name': info.get('longName', symbol),
                    'price': current_price,
                    'change_24h': change_24h,
                    'market_cap': info.get('marketCap', 0),
                    'volume': info.get('volume', 0)
                })
            except:
                continue
        
        # Calcular estadísticas del mercado
        total_market_cap = sum(crypto['market_cap'] for crypto in market_data)
        avg_change_24h = sum(crypto['change_24h'] for crypto in market_data) / len(market_data)
        
        # Determinar sentimiento del mercado
        bullish_count = len([c for c in market_data if c['change_24h'] > 0])
        bearish_count = len([c for c in market_data if c['change_24h'] < 0])
        
        if bullish_count > bearish_count:
            market_sentiment = "Alcista"
        elif bearish_count > bullish_count:
            market_sentiment = "Bajista"
        else:
            market_sentiment = "Neutral"
        
        return jsonify({
            'market_data': market_data,
            'summary': {
                'total_market_cap': total_market_cap,
                'avg_change_24h': avg_change_24h,
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'market_sentiment': market_sentiment
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Error obteniendo vista del mercado: {str(e)}'})

@app.route('/api/set_alert', methods=['POST'])
def set_alert():
    """Configurar alerta de precio"""
    data = request.get_json()
    symbol = data.get('symbol', '').upper()
    target_price = data.get('target_price')
    alert_type = data.get('type', 'above')  # 'above' or 'below'
    
    if not symbol or not target_price:
        return jsonify({'error': 'Símbolo y precio objetivo son requeridos'})
    
    if symbol not in alerts:
        alerts[symbol] = []
    
    alert_id = len(alerts[symbol]) + 1
    alert = {
        'id': alert_id,
        'symbol': symbol,
        'target_price': float(target_price),
        'type': alert_type,
        'created_at': datetime.now().isoformat(),
        'triggered': False
    }
    
    alerts[symbol].append(alert)
    
    return jsonify({
        'success': True,
        'alert_id': alert_id,
        'message': f'Alerta configurada para {symbol} a ${target_price}'
    })

@app.route('/api/get_alerts/<symbol>')
def get_alerts(symbol):
    """Obtener alertas para un símbolo"""
    symbol = symbol.upper()
    if symbol in alerts:
        return jsonify(alerts[symbol])
    return jsonify([])

@app.route('/api/delete_alert', methods=['POST'])
def delete_alert():
    """Eliminar alerta"""
    data = request.get_json()
    symbol = data.get('symbol', '').upper()
    alert_id = data.get('alert_id')
    
    if symbol in alerts and alert_id:
        alerts[symbol] = [a for a in alerts[symbol] if a['id'] != alert_id]
        return jsonify({'success': True, 'message': 'Alerta eliminada'})
    
    return jsonify({'error': 'Alerta no encontrada'})

@app.route('/api/check_alerts')
def check_alerts():
    """Verificar alertas activas"""
    triggered_alerts = []
    
    for symbol, symbol_alerts in alerts.items():
        try:
            ticker = yf.Ticker(f"{symbol}-USD")
            current_price = ticker.history(period='1d')['Close'].iloc[-1]
            
            for alert in symbol_alerts:
                if alert['triggered']:
                    continue
                
                triggered = False
                if alert['type'] == 'above' and current_price >= alert['target_price']:
                    triggered = True
                elif alert['type'] == 'below' and current_price <= alert['target_price']:
                    triggered = True
                
                if triggered:
                    alert['triggered'] = True
                    alert['triggered_at'] = datetime.now().isoformat()
                    alert['triggered_price'] = current_price
                    triggered_alerts.append(alert)
                    
        except Exception as e:
            print(f"Error verificando alertas para {symbol}: {e}")
            continue
    
    return jsonify({
        'triggered_alerts': triggered_alerts,
        'total_alerts': sum(len(alerts[symbol]) for symbol in alerts)
    })

@app.route('/api/portfolio_simulation', methods=['POST'])
def portfolio_simulation():
    """Simular portafolio de inversión"""
    data = request.get_json()
    investments = data.get('investments', [])
    
    if not investments:
        return jsonify({'error': 'No hay inversiones para simular'})
    
    portfolio_value = 0
    portfolio_change = 0
    total_invested = 0
    
    for investment in investments:
        symbol = investment['symbol']
        amount = investment['amount']
        buy_price = investment['buy_price']
        
        try:
            ticker = yf.Ticker(f"{symbol}-USD")
            current_price = ticker.history(period='1d')['Close'].iloc[-1]
            
            current_value = (amount / buy_price) * current_price
            profit_loss = current_value - amount
            profit_loss_pct = (profit_loss / amount) * 100
            
            portfolio_value += current_value
            portfolio_change += profit_loss
            total_invested += amount
            
            investment.update({
                'current_price': current_price,
                'current_value': current_value,
                'profit_loss': profit_loss,
                'profit_loss_pct': profit_loss_pct
            })
            
        except Exception as e:
            print(f"Error calculando {symbol}: {e}")
            continue
    
    portfolio_change_pct = (portfolio_change / total_invested) * 100 if total_invested > 0 else 0
    
    return jsonify({
        'investments': investments,
        'summary': {
            'total_invested': total_invested,
            'portfolio_value': portfolio_value,
            'portfolio_change': portfolio_change,
            'portfolio_change_pct': portfolio_change_pct
        }
    })

# ============================================================================
# FUNCIONALIDADES AVANZADAS
# ============================================================================

# 1. ANÁLISIS DE SENTIMIENTO
class SentimentAnalyzer:
    def __init__(self):
        self.news_sources = [
            'https://api.coingecko.com/api/v3/news',
            'https://cryptonews-api.com/api/v1/news'
        ]
    
    def analyze_news_sentiment(self, symbol):
        """Analizar sentimiento de noticias relacionadas con una criptomoneda"""
        try:
            # Simular análisis de noticias (en producción usarías APIs reales)
            news_data = [
                {"title": f"Positive news about {symbol}", "sentiment": 0.8},
                {"title": f"Market analysis for {symbol}", "sentiment": 0.5},
                {"title": f"Technical analysis {symbol}", "sentiment": 0.6}
            ]
            
            avg_sentiment = sum(news['sentiment'] for news in news_data) / len(news_data)
            
            if avg_sentiment > 0.6:
                sentiment_label = "Positivo"
            elif avg_sentiment < 0.4:
                sentiment_label = "Negativo"
            else:
                sentiment_label = "Neutral"
            
            return {
                'sentiment_score': avg_sentiment,
                'sentiment_label': sentiment_label,
                'news_count': len(news_data),
                'confidence': 'Media'
            }
        except Exception as e:
            return {'error': f'Error analizando sentimiento: {str(e)}'}

sentiment_analyzer = SentimentAnalyzer()

# 2. BACKTESTING DE ESTRATEGIAS
class Backtester:
    def __init__(self):
        self.strategies = {
            'buy_and_hold': self.buy_and_hold,
            'moving_average': self.moving_average_crossover,
            'rsi_strategy': self.rsi_strategy,
            'momentum': self.momentum_strategy
        }
    
    def buy_and_hold(self, data):
        """Estrategia de comprar y mantener"""
        initial_price = data['Close'].iloc[0]
        final_price = data['Close'].iloc[-1]
        return (final_price - initial_price) / initial_price * 100
    
    def moving_average_crossover(self, data, short_window=20, long_window=50):
        """Estrategia de cruce de medias móviles"""
        data = data.copy()
        data['SMA_short'] = data['Close'].rolling(window=short_window).mean()
        data['SMA_long'] = data['Close'].rolling(window=long_window).mean()
        
        position = 0
        returns = []
        
        for i in range(1, len(data)):
            if data['SMA_short'].iloc[i] > data['SMA_long'].iloc[i] and position == 0:
                position = 1
            elif data['SMA_short'].iloc[i] < data['SMA_long'].iloc[i] and position == 1:
                position = 0
                returns.append((data['Close'].iloc[i] - data['Close'].iloc[i-1]) / data['Close'].iloc[i-1])
        
        return sum(returns) * 100 if returns else 0
    
    def rsi_strategy(self, data, period=14, oversold=30, overbought=70):
        """Estrategia basada en RSI"""
        data = data.copy()
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        position = 0
        returns = []
        
        for i in range(1, len(data)):
            rsi = data['RSI'].iloc[i]
            if rsi < oversold and position == 0:
                position = 1
            elif rsi > overbought and position == 1:
                position = 0
                returns.append((data['Close'].iloc[i] - data['Close'].iloc[i-1]) / data['Close'].iloc[i-1])
        
        return sum(returns) * 100 if returns else 0
    
    def momentum_strategy(self, data, period=10):
        """Estrategia de momentum"""
        data = data.copy()
        data['Momentum'] = data['Close'].pct_change(period)
        
        position = 0
        returns = []
        
        for i in range(period, len(data)):
            momentum = data['Momentum'].iloc[i]
            if momentum > 0.02 and position == 0:  # 2% de momentum positivo
                position = 1
            elif momentum < -0.02 and position == 1:  # -2% de momentum
                position = 0
                returns.append((data['Close'].iloc[i] - data['Close'].iloc[i-1]) / data['Close'].iloc[i-1])
        
        return sum(returns) * 100 if returns else 0
    
    def run_backtest(self, symbol, strategy_name, period='1y'):
        """Ejecutar backtest de una estrategia"""
        try:
            data = predictor.get_crypto_data(symbol, period)
            if data is None:
                return {'error': 'No se pudieron obtener datos'}
            
            if strategy_name not in self.strategies:
                return {'error': 'Estrategia no encontrada'}
            
            strategy_func = self.strategies[strategy_name]
            returns = strategy_func(data)
            
            # Calcular métricas adicionales
            buy_hold_return = self.buy_and_hold(data)
            excess_return = returns - buy_hold_return
            
            return {
                'strategy': strategy_name,
                'returns': returns,
                'buy_hold_return': buy_hold_return,
                'excess_return': excess_return,
                'period': period,
                'symbol': symbol
            }
        except Exception as e:
            return {'error': f'Error en backtest: {str(e)}'}

backtester = Backtester()

# 3. EXPORTACIÓN Y REPORTES
class ReportGenerator:
    def __init__(self):
        self.report_types = ['csv', 'json', 'pdf', 'excel']
    
    def generate_csv_report(self, data, filename):
        """Generar reporte CSV"""
        output = io.StringIO()
        writer = csv.writer(output)
        
        if isinstance(data, list) and len(data) > 0:
            # Escribir headers
            writer.writerow(data[0].keys())
            # Escribir datos
            for row in data:
                writer.writerow(row.values())
        
        output.seek(0)
        return output.getvalue()
    
    def generate_json_report(self, data):
        """Generar reporte JSON"""
        return json.dumps(data, indent=2, default=str)
    
    def generate_market_report(self, market_data):
        """Generar reporte completo del mercado"""
        report = {
            'generated_at': datetime.now().isoformat(),
            'market_summary': market_data.get('summary', {}),
            'top_performers': [],
            'worst_performers': [],
            'recommendations': []
        }
        
        # Analizar top performers
        market_data_list = market_data.get('market_data', [])
        if market_data_list:
            sorted_data = sorted(market_data_list, key=lambda x: x.get('change_24h', 0), reverse=True)
            report['top_performers'] = sorted_data[:5]
            report['worst_performers'] = sorted_data[-5:]
        
        return report

report_generator = ReportGenerator()

# 4. ANÁLISIS TÉCNICO AVANZADO
class TechnicalAnalyzer:
    def __init__(self):
        self.indicators = {}
    
    def calculate_sma(self, data, period):
        """Calcular Media Móvil Simple"""
        return data['Close'].rolling(window=period).mean()
    
    def calculate_ema(self, data, period):
        """Calcular Media Móvil Exponencial"""
        return data['Close'].ewm(span=period).mean()
    
    def calculate_bollinger_bands(self, data, period=20, std_dev=2):
        """Calcular Bandas de Bollinger"""
        sma = self.calculate_sma(data, period)
        std = data['Close'].rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
    
    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        """Calcular MACD"""
        ema_fast = self.calculate_ema(data, fast)
        ema_slow = self.calculate_ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_rsi(self, data, period=14):
        """Calcular RSI"""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_technical_analysis(self, symbol):
        """Obtener análisis técnico completo"""
        try:
            data = predictor.get_crypto_data(symbol, period='6mo')
            if data is None:
                return {'error': 'No se pudieron obtener datos'}
            
            # Calcular indicadores
            sma_20 = self.calculate_sma(data, 20)
            sma_50 = self.calculate_sma(data, 50)
            ema_12 = self.calculate_ema(data, 12)
            ema_26 = self.calculate_ema(data, 26)
            
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(data)
            macd_line, signal_line, histogram = self.calculate_macd(data)
            rsi = self.calculate_rsi(data)
            
            # Obtener valores actuales
            current_price = data['Close'].iloc[-1]
            current_sma_20 = sma_20.iloc[-1]
            current_sma_50 = sma_50.iloc[-1]
            current_rsi = rsi.iloc[-1]
            current_macd = macd_line.iloc[-1]
            current_signal = signal_line.iloc[-1]
            
            # Generar señales
            signals = []
            
            # Señal de cruce de medias móviles
            if current_sma_20 > current_sma_50:
                signals.append("Señal alcista: SMA 20 > SMA 50")
            else:
                signals.append("Señal bajista: SMA 20 < SMA 50")
            
            # Señal RSI
            if current_rsi > 70:
                signals.append("RSI sobrecomprado (>70)")
            elif current_rsi < 30:
                signals.append("RSI sobrevendido (<30)")
            else:
                signals.append("RSI en rango normal")
            
            # Señal MACD
            if current_macd > current_signal:
                signals.append("MACD alcista")
            else:
                signals.append("MACD bajista")
            
            # Bandas de Bollinger
            if current_price > bb_upper.iloc[-1]:
                signals.append("Precio por encima de banda superior")
            elif current_price < bb_lower.iloc[-1]:
                signals.append("Precio por debajo de banda inferior")
            else:
                signals.append("Precio dentro de las bandas")
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'indicators': {
                    'sma_20': current_sma_20,
                    'sma_50': current_sma_50,
                    'rsi': current_rsi,
                    'macd': current_macd,
                    'macd_signal': current_signal,
                    'bb_upper': bb_upper.iloc[-1],
                    'bb_lower': bb_lower.iloc[-1]
                },
                'signals': signals,
                'recommendation': self._generate_technical_recommendation(signals)
            }
        except Exception as e:
            return {'error': f'Error en análisis técnico: {str(e)}'}

    def _generate_technical_recommendation(self, signals):
        """Generar recomendación basada en señales técnicas"""
        bullish_signals = sum(1 for signal in signals if 'alcista' in signal.lower())
        bearish_signals = sum(1 for signal in signals if 'bajista' in signal.lower())
        
        if bullish_signals > bearish_signals:
            return "COMPRAR"
        elif bearish_signals > bullish_signals:
            return "VENDER"
        else:
            return "MANTENER"

technical_analyzer = TechnicalAnalyzer()

# 5. MACHINE LEARNING AVANZADO
class AdvancedMLPredictor:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression(),
            'svr': SVR(kernel='rbf'),
            'neural_network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        self.trained_models = {}
        self.model_performance = {}
    
    def prepare_advanced_features(self, data):
        """Preparar características avanzadas para ML"""
        df = data.copy()
        
        # Características básicas
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        
        # Características temporales
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Day_of_Year'] = df.index.dayofyear
        
        # Características técnicas
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['Price_vs_SMA20'] = df['Close'] / df['SMA_20']
        df['Price_vs_SMA50'] = df['Close'] / df['SMA_50']
        
        # Volatilidad
        df['Volatility'] = df['Close'].rolling(window=20).std()
        df['Volatility_Ratio'] = df['Volatility'] / df['Close']
        
        # Momentum
        df['Momentum_5'] = df['Close'].pct_change(5)
        df['Momentum_10'] = df['Close'].pct_change(10)
        df['Momentum_20'] = df['Close'].pct_change(20)
        
        # Volumen
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        df = df.dropna()
        return df
    
    def train_advanced_model(self, symbol, model_type='random_forest'):
        """Entrenar modelo avanzado"""
        try:
            data = predictor.get_crypto_data(symbol, period='2y')
            if data is None or len(data) < 100:
                return False
            
            df = self.prepare_advanced_features(data)
            
            # Características para el modelo
            feature_columns = [
                'Price_Change', 'Volume_Change', 'High_Low_Ratio',
                'Day_of_Week', 'Month', 'Day_of_Year',
                'Price_vs_SMA20', 'Price_vs_SMA50',
                'Volatility_Ratio', 'Momentum_5', 'Momentum_10', 'Momentum_20',
                'Volume_Ratio'
            ]
            
            X = df[feature_columns].values
            y = df['Close'].values
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Entrenar modelo
            model = self.models[model_type]
            model.fit(X_train, y_train)
            
            # Evaluar modelo
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Guardar modelo y métricas
            model_key = f"{symbol}_{model_type}"
            self.trained_models[model_key] = model
            self.model_performance[model_key] = {
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse)
            }
            
            return True
        except Exception as e:
            print(f"Error entrenando modelo avanzado: {e}")
            return False
    
    def predict_with_advanced_model(self, symbol, model_type='random_forest', days=7):
        """Predecir usando modelo avanzado"""
        try:
            model_key = f"{symbol}_{model_type}"
            if model_key not in self.trained_models:
                if not self.train_advanced_model(symbol, model_type):
                    return None
            
            model = self.trained_models[model_key]
            data = predictor.get_crypto_data(symbol, period='3mo')
            if data is None:
                return None
            
            df = self.prepare_advanced_features(data)
            feature_columns = [
                'Price_Change', 'Volume_Change', 'High_Low_Ratio',
                'Day_of_Week', 'Month', 'Day_of_Year',
                'Price_vs_SMA20', 'Price_vs_SMA50',
                'Volatility_Ratio', 'Momentum_5', 'Momentum_10', 'Momentum_20',
                'Volume_Ratio'
            ]
            
            predictions = []
            current_features = df[feature_columns].iloc[-1:].values
            
            for i in range(days):
                pred = model.predict(current_features)[0]
                predictions.append(pred)
                
                # Actualizar características para siguiente predicción
                if i < days - 1:
                    # Simular actualización de características
                    current_features[0][0] = (pred - df['Close'].iloc[-1]) / df['Close'].iloc[-1]
            
            return {
                'predictions': predictions,
                'model_type': model_type,
                'performance': self.model_performance.get(model_key, {}),
                'confidence': self._calculate_confidence(predictions)
            }
        except Exception as e:
            print(f"Error en predicción avanzada: {e}")
            return None
    
    def _calculate_confidence(self, predictions):
        """Calcular nivel de confianza de las predicciones"""
        if not predictions:
            return "Baja"
        
        # Calcular volatilidad de predicciones
        volatility = np.std(predictions) / np.mean(predictions)
        
        if volatility < 0.05:
            return "Alta"
        elif volatility < 0.1:
            return "Media"
        else:
            return "Baja"

advanced_ml_predictor = AdvancedMLPredictor()

# 6. SISTEMA DE ALERTAS AVANZADAS
class AdvancedAlertSystem:
    def __init__(self):
        self.alerts = {}
        self.alert_types = {
            'price': self._check_price_alert,
            'volume': self._check_volume_alert,
            'technical': self._check_technical_alert,
            'sentiment': self._check_sentiment_alert
        }
    
    def _check_price_alert(self, symbol, alert_config):
        """Verificar alerta de precio"""
        try:
            ticker = yf.Ticker(f"{symbol}-USD")
            current_price = ticker.history(period='1d')['Close'].iloc[-1]
            target_price = alert_config['target_price']
            alert_type = alert_config['type']
            
            if alert_type == 'above' and current_price >= target_price:
                return True, f"Precio de {symbol} alcanzó ${target_price}"
            elif alert_type == 'below' and current_price <= target_price:
                return True, f"Precio de {symbol} cayó a ${target_price}"
            
            return False, None
        except:
            return False, None
    
    def _check_volume_alert(self, symbol, alert_config):
        """Verificar alerta de volumen"""
        try:
            ticker = yf.Ticker(f"{symbol}-USD")
            data = ticker.history(period='5d')
            current_volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].mean()
            threshold = alert_config.get('threshold', 2.0)
            
            if current_volume > avg_volume * threshold:
                return True, f"Volumen inusual en {symbol}: {current_volume:,.0f}"
            
            return False, None
        except:
            return False, None
    
    def _check_technical_alert(self, symbol, alert_config):
        """Verificar alerta técnica"""
        try:
            analysis = technical_analyzer.get_technical_analysis(symbol)
            if 'error' in analysis:
                return False, None
            
            indicator = alert_config['indicator']
            threshold = alert_config['threshold']
            current_value = analysis['indicators'].get(indicator, 0)
            
            if indicator == 'rsi':
                if current_value > threshold:
                    return True, f"RSI de {symbol} sobrecomprado: {current_value:.2f}"
                elif current_value < (100 - threshold):
                    return True, f"RSI de {symbol} sobrevendido: {current_value:.2f}"
            
            return False, None
        except:
            return False, None
    
    def _check_sentiment_alert(self, symbol, alert_config):
        """Verificar alerta de sentimiento"""
        try:
            sentiment = sentiment_analyzer.analyze_news_sentiment(symbol)
            if 'error' in sentiment:
                return False, None
            
            sentiment_score = sentiment['sentiment_score']
            threshold = alert_config.get('threshold', 0.7)
            
            if sentiment_score > threshold:
                return True, f"Sentimiento muy positivo para {symbol}: {sentiment_score:.2f}"
            elif sentiment_score < (1 - threshold):
                return True, f"Sentimiento muy negativo para {symbol}: {sentiment_score:.2f}"
            
            return False, None
        except:
            return False, None
    
    def add_advanced_alert(self, symbol, alert_type, config):
        """Agregar alerta avanzada"""
        if symbol not in self.alerts:
            self.alerts[symbol] = []
        
        alert_id = len(self.alerts[symbol]) + 1
        alert = {
            'id': alert_id,
            'symbol': symbol,
            'type': alert_type,
            'config': config,
            'created_at': datetime.now().isoformat(),
            'triggered': False
        }
        
        self.alerts[symbol].append(alert)
        return alert_id
    
    def check_advanced_alerts(self):
        """Verificar todas las alertas avanzadas"""
        triggered_alerts = []
        
        for symbol, symbol_alerts in self.alerts.items():
            for alert in symbol_alerts:
                if alert['triggered']:
                    continue
                
                alert_type = alert['type']
                if alert_type in self.alert_types:
                    triggered, message = self.alert_types[alert_type](symbol, alert['config'])
                    
                    if triggered:
                        alert['triggered'] = True
                        alert['triggered_at'] = datetime.now().isoformat()
                        alert['message'] = message
                        triggered_alerts.append(alert)
        
        return triggered_alerts

advanced_alert_system = AdvancedAlertSystem()

# ============================================================================
# NUEVAS RUTAS API
# ============================================================================

@app.route('/api/sentiment_analysis/<symbol>')
def get_sentiment_analysis(symbol):
    """Obtener análisis de sentimiento"""
    return jsonify(sentiment_analyzer.analyze_news_sentiment(symbol))

@app.route('/api/backtest/<symbol>/<strategy>')
def run_backtest_api(symbol, strategy):
    """Ejecutar backtest de estrategia"""
    return jsonify(backtester.run_backtest(symbol, strategy))

@app.route('/api/technical_analysis/<symbol>')
def get_technical_analysis_api(symbol):
    """Obtener análisis técnico avanzado"""
    return jsonify(technical_analyzer.get_technical_analysis(symbol))

@app.route('/api/advanced_prediction/<symbol>')
def get_advanced_prediction(symbol):
    """Obtener predicción usando modelo avanzado"""
    model_type = request.args.get('model', 'random_forest')
    days = int(request.args.get('days', 7))
    
    result = advanced_ml_predictor.predict_with_advanced_model(symbol, model_type, days)
    return jsonify(result if result else {'error': 'Error en predicción avanzada'})

@app.route('/api/export_report', methods=['POST'])
def export_report():
    """Exportar reporte en diferentes formatos"""
    data = request.get_json()
    report_type = data.get('type', 'json')
    report_data = data.get('data', {})
    
    if report_type == 'csv':
        csv_data = report_generator.generate_csv_report(report_data, 'report.csv')
        return jsonify({'csv_data': csv_data})
    elif report_type == 'json':
        json_data = report_generator.generate_json_report(report_data)
        return jsonify({'json_data': json_data})
    else:
        return jsonify({'error': 'Formato no soportado'})

@app.route('/api/advanced_alert', methods=['POST'])
def add_advanced_alert():
    """Agregar alerta avanzada"""
    data = request.get_json()
    symbol = data.get('symbol', '').upper()
    alert_type = data.get('type', 'price')
    config = data.get('config', {})
    
    if not symbol or not config:
        return jsonify({'error': 'Símbolo y configuración son requeridos'})
    
    alert_id = advanced_alert_system.add_advanced_alert(symbol, alert_type, config)
    
    return jsonify({
        'success': True,
        'alert_id': alert_id,
        'message': f'Alerta avanzada configurada para {symbol}'
    })

@app.route('/api/check_advanced_alerts')
def check_advanced_alerts():
    """Verificar alertas avanzadas"""
    triggered = advanced_alert_system.check_advanced_alerts()
    return jsonify({'triggered_alerts': triggered})

@app.route('/api/train_advanced_model', methods=['POST'])
def train_advanced_model():
    """Entrenar modelo avanzado"""
    data = request.get_json()
    symbol = data.get('symbol', '').upper()
    model_type = data.get('model_type', 'random_forest')
    
    if not symbol:
        return jsonify({'error': 'Símbolo es requerido'})
    
    success = advanced_ml_predictor.train_advanced_model(symbol, model_type)
    
    if success:
        return jsonify({
            'success': True,
            'message': f'Modelo {model_type} entrenado para {symbol}',
            'performance': advanced_ml_predictor.model_performance.get(f"{symbol}_{model_type}", {})
        })
    else:
        return jsonify({'error': f'Error entrenando modelo para {symbol}'})

@app.route('/api/dashboard_data')
def get_dashboard_data():
    """Obtener datos para dashboard personalizado"""
    try:
        # Datos del mercado
        market_data = get_market_overview().get_json()
        
        # Top performers
        top_performers = []
        for crypto in POPULAR_CRYPTOS[:5]:
            try:
                ticker = yf.Ticker(f"{crypto['symbol']}-USD")
                current_price = ticker.history(period='1d')['Close'].iloc[-1]
                change_24h = ticker.info.get('regularMarketChangePercent', 0)
                
                top_performers.append({
                    'symbol': crypto['symbol'],
                    'name': crypto['name'],
                    'price': current_price,
                    'change_24h': change_24h
                })
            except:
                continue
        
        # Alertas activas
        active_alerts = sum(len(alerts[symbol]) for symbol in alerts)
        
        # Predicciones recientes
        recent_predictions = len(predictor.prediction_history)
        
        return jsonify({
            'market_summary': market_data.get('summary', {}),
            'top_performers': top_performers,
            'active_alerts': active_alerts,
            'recent_predictions': recent_predictions,
            'last_updated': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': f'Error obteniendo datos del dashboard: {str(e)}'})

# ============================================================================
# SISTEMA DE BASE DE DATOS SIMPLE
# ============================================================================

# Simular base de datos con archivos JSON
class SimpleDatabase:
    def __init__(self):
        self.data_dir = 'data'
        os.makedirs(self.data_dir, exist_ok=True)
    
    def save_data(self, collection, data):
        """Guardar datos en archivo JSON"""
        filename = os.path.join(self.data_dir, f'{collection}.json')
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"Error guardando datos: {e}")
            return False
    
    def load_data(self, collection):
        """Cargar datos desde archivo JSON"""
        filename = os.path.join(self.data_dir, f'{collection}.json')
        try:
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"Error cargando datos: {e}")
            return []
    
    def add_user(self, user_data):
        """Agregar usuario"""
        users = self.load_data('users')
        user_id = len(users) + 1
        user_data['id'] = user_id
        user_data['created_at'] = datetime.now().isoformat()
        users.append(user_data)
        self.save_data('users', users)
        return user_id
    
    def get_user(self, user_id):
        """Obtener usuario"""
        users = self.load_data('users')
        for user in users:
            if user.get('id') == user_id:
                return user
        return None
    
    def save_user_portfolio(self, user_id, portfolio_data):
        """Guardar portafolio de usuario"""
        portfolios = self.load_data('portfolios')
        portfolio_data['user_id'] = user_id
        portfolio_data['updated_at'] = datetime.now().isoformat()
        
        # Actualizar o agregar portafolio
        found = False
        for i, portfolio in enumerate(portfolios):
            if portfolio.get('user_id') == user_id:
                portfolios[i] = portfolio_data
                found = True
                break
        
        if not found:
            portfolios.append(portfolio_data)
        
        self.save_data('portfolios', portfolios)
        return True

db = SimpleDatabase()

@app.route('/api/register_user', methods=['POST'])
def register_user():
    """Registrar nuevo usuario"""
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    
    if not username or not email:
        return jsonify({'error': 'Username y email son requeridos'})
    
    user_id = db.add_user({
        'username': username,
        'email': email,
        'preferences': data.get('preferences', {}),
        'created_at': datetime.now().isoformat()
    })
    
    return jsonify({
        'success': True,
        'user_id': user_id,
        'message': 'Usuario registrado exitosamente'
    })

@app.route('/api/save_portfolio', methods=['POST'])
def save_portfolio():
    """Guardar portafolio de usuario"""
    data = request.get_json()
    user_id = data.get('user_id')
    portfolio_data = data.get('portfolio', {})
    
    if not user_id:
        return jsonify({'error': 'User ID es requerido'})
    
    success = db.save_user_portfolio(user_id, portfolio_data)
    
    return jsonify({
        'success': success,
        'message': 'Portafolio guardado exitosamente' if success else 'Error guardando portafolio'
    })

# ============================================================================
# SISTEMA DE NOTIFICACIONES
# ============================================================================

class NotificationSystem:
    def __init__(self):
        self.notifications = []
    
    def add_notification(self, user_id, message, notification_type='info'):
        """Agregar notificación"""
        notification = {
            'id': len(self.notifications) + 1,
            'user_id': user_id,
            'message': message,
            'type': notification_type,
            'created_at': datetime.now().isoformat(),
            'read': False
        }
        self.notifications.append(notification)
        return notification['id']
    
    def get_user_notifications(self, user_id):
        """Obtener notificaciones de usuario"""
        return [n for n in self.notifications if n['user_id'] == user_id]
    
    def mark_as_read(self, notification_id):
        """Marcar notificación como leída"""
        for notification in self.notifications:
            if notification['id'] == notification_id:
                notification['read'] = True
                return True
        return False

notification_system = NotificationSystem()

@app.route('/api/notifications/<user_id>')
def get_notifications(user_id):
    """Obtener notificaciones de usuario"""
    notifications = notification_system.get_user_notifications(int(user_id))
    return jsonify(notifications)

@app.route('/api/mark_notification_read', methods=['POST'])
def mark_notification_read():
    """Marcar notificación como leída"""
    data = request.get_json()
    notification_id = data.get('notification_id')
    
    success = notification_system.mark_as_read(notification_id)
    return jsonify({'success': success})

# ============================================================================
# SISTEMA DE TAREAS PROGRAMADAS
# ============================================================================

def scheduled_alerts_check():
    """Verificar alertas programadas"""
    try:
        # Verificar alertas básicas
        check_alerts()
        
        # Verificar alertas avanzadas
        triggered = advanced_alert_system.check_advanced_alerts()
        
        # Enviar notificaciones si hay alertas activadas
        for alert in triggered:
            notification_system.add_notification(
                user_id=1,  # Usuario por defecto
                message=alert.get('message', 'Alerta activada'),
                notification_type='alert'
            )
        
        print(f"Verificación de alertas completada: {len(triggered)} alertas activadas")
    except Exception as e:
        print(f"Error en verificación programada de alertas: {e}")

# Programar tareas
schedule.every(1).minutes.do(scheduled_alerts_check)

def run_scheduler():
    """Ejecutar programador de tareas"""
    while True:
        schedule.run_pending()
        time.sleep(60)

# Iniciar scheduler en thread separado
scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
scheduler_thread.start()

if __name__ == '__main__':
    # Para desarrollo local
    if os.environ.get('FLASK_ENV') == 'development':
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        # Para producción
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port) 