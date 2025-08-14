LSTM-Based Stock Price Prediction with Technical Indicators

OverviewThis project implements a deep learning pipeline for predicting stock prices using Long Short-Term Memory (LSTM) networks enhanced with technical indicators.It is designed for financial time-series forecasting, incorporating both trend-following and momentum-based features, alongside a custom asymmetric loss function to prioritize directional accuracy — a critical factor in trading systems.

Use Cases:

*   Institutional investors and traders seeking improved price forecasting.
    
*   Risk management teams leveraging AI for scenario testing.
    
*   Quantitative research for feature engineering in algorithmic trading.
    

Features

1.  Data Processing
    

*   Loads Japan Stock Market (1999–2024) dataset.
    
*   Filters by company ticker for targeted predictions.
    
*   Cleans and fills missing values using forward/backward fill.
    

1.  Technical IndicatorsCalculated for each stock:
    

*   Trend Indicators: SMA, EMA, ADX
    
*   Momentum Indicators: RSI, MACD
    
*   Volatility Indicators: Bollinger Bands, ATR
    
*   Volume Indicators: OBV, VWAP
    
*   Price Action Metrics: Daily returns, log returns, volatility, price range
    

1.  Feature Engineering
    

*   Combines multiple indicator classes into a unified feature set.
    
*   Scales features using MinMaxScaler for deep learning stability.
    
*   Prepares look-back sequences for time-series LSTM modeling.
    

1.  Model Architecture
    

*   Multi-layer LSTM with:
    
    *   128 → 64 → 32 LSTM units
        
    *   Batch Normalization for training stability
        
    *   Dropout regularization to prevent overfitting
        
    *   Dense layers for feature extraction
        
*   Optimized for temporal dependencies in price data.
    

1.  Custom Loss Function
    

*   Directional Asymmetric Loss: Penalizes incorrect price direction predictions more heavily.
    
*   Useful in trading where predicting correct movement direction matters more than absolute price accuracy.
    

1.  Training Enhancements
    

*   GPU acceleration for faster training.
    
*   Early stopping and checkpoint saving.
    
*   Adam optimizer with learning rate control.
    

Project Structure

*   data/ — Dataset storage (Kaggle input data or local CSVs)
    
*   lstm-indicators-jpnse-newest.ipynb — Main training notebook
    
*   README.txt — Project documentation
    
*   requirements.txt — Python dependencies
    

Usage Steps

1.  Install Dependenciespip install -r requirements.txt
    
2.  Run the NotebookExample:stock\_data = preprocess\_data(df, "HITACHI")X, y, scaler, features = prepare\_data(stock\_data)model = create\_enhanced\_lstm\_model((X.shape\[1\], X.shape\[2\]))
    
3.  Train the Modelmodel.compile(optimizer=Adam(learning\_rate=0.001), loss=directional\_asymmetric\_loss)model.fit(X\_train, y\_train, validation\_data=(X\_test, y\_test), epochs=50, callbacks=\[...\])
    
4.  Predict and Evaluate
    

*   Predict future prices
    
*   Plot actual vs predicted trends
    
*   Calculate metrics such as RMSE, R², and directional accuracy
    

Example Results

*   Directional accuracy prioritized over RMSE for trading relevance.
    
*   Model output includes next-day price forecast and trend direction signal.
    
*   Visualizations include technical indicator overlays with predicted vs actual prices.
    

Relevance to Finance and Trading

*   Predictive modeling using historical prices and technical signals to forecast future values.
    
*   Risk mitigation through asymmetric loss aligned with risk-averse strategies.
    
*   Quantitative strategy development with feature-rich datasets for algorithmic trading integration.
    

Requirements

*   Python 3.8+
    
*   TensorFlow / Keras
    
*   Pandas, NumPy, Scikit-learn
    
*   Matplotlib, Seaborn
    

LicenseMIT License

Author: Aman AgarwalKeywords: LSTM, Stock Forecasting, Technical Analysis, Deep Learning, Quantitative Finance
