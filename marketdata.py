from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf

app = Flask(__name__)
api = Api(app)

class MarketData(Resource):
    def get(self, symbol):
        try:
            # Get market data for the given symbol
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="max")

            # Convert to JSON format
            data_json = data.to_dict(orient='records')

            return jsonify(data_json)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

class MarketSignals(Resource):
    def get(self, symbol):
        try:
            # Get market data for the given symbol
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="max")

            # Calculate technical indicators
            data = self.calculate_indicators(data)

            # Generate trading signals
            signals = self.generate_signals(data)

            return jsonify(signals)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def calculate_indicators(self, data):
        """Calculate technical indicators for the given data"""
        # Calculate 20-day and 50-day moving averages
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()

        # Calculate RSI (Relative Strength Index)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

        return data

    def generate_signals(self, data):
        """Generate buy/sell signals based on technical indicators"""
        signals = []

        # Create a copy of the data to avoid modifying original
        df = data.copy()

        # Initialize signal column
        df['Signal'] = 0

        # Generate signals based on moving average crossover
        # Buy when MA20 crosses above MA50
        df['MA20_Cross'] = df['MA20'] > df['MA50']
        df['Signal'] = df['MA20_Cross'].diff().fillna(0)

        # Generate buy signals (1) and sell signals (-1)
        df.loc[df['Signal'] == 1, 'Signal'] = 1
        df.loc[df['Signal'] == -1, 'Signal'] = -1

        # Convert to dictionary format for JSON response
        for index, row in df.iterrows():
            if row['Signal'] != 0:
                signals.append({
                    'date': index.strftime('%Y-%m-%d'),
                    'signal': 'buy' if row['Signal'] == 1 else 'sell',
                    'price': row['Close'],
                    'MA20': row['MA20'],
                    'MA50': row['MA50'],
                    'RSI': row['RSI']
                })

        return signals

api.add_resource(MarketData, '/marketdata/<string:symbol>')
api.add_resource(MarketSignals, '/marketsignals/<string:symbol>')

if __name__ == '__main__':
    app.run(debug=True)
