# markert-risk-on-portfolio
# Import necessary libraries
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Step 1: Get User Inputs for Portfolio
print("Enter the stock tickers in your portfolio (comma-separated, e.g., AAPL,MSFT,GOOG):")
tickers = input().strip().split(",")  # Input tickers

print("Enter the portfolio weights for each stock (comma-separated, e.g., 0.4,0.3,0.3):")
weights = list(map(float, input().strip().split(",")))  # Input weights

# Validate weights sum to 1
if not np.isclose(sum(weights), 1.0):
    raise ValueError("Portfolio weights must sum to 1. Please try again.")

# Step 2: Fetch Historical Data
print("Fetching data for tickers:", tickers)
data = yf.download(tickers, start="2020-01-01", end="2023-12-01")['Adj Close']

# Calculate daily returns for all assets
returns = data.pct_change().dropna()

# Step 3: Compute Portfolio Returns
portfolio_returns = (returns * weights).sum(axis=1)

# Step 4: Feature Engineering
rolling_window = 20  # Rolling window size for feature computation

# Portfolio-level features
features = pd.DataFrame({
    'Portfolio_Volatility': portfolio_returns.rolling(rolling_window).std(),
    'Portfolio_Max': portfolio_returns.rolling(rolling_window).max(),
    'Portfolio_Min': portfolio_returns.rolling(rolling_window).min(),
    'Portfolio_Mean': portfolio_returns.rolling(rolling_window).mean(),
})

# Add lagged returns for each asset as predictors
for ticker in tickers:
    features[f"{ticker}_Lag1"] = returns[ticker].shift(1)

# Target Variable: Rolling VaR (99% confidence level)
confidence_level = 0.99
features['VaR_Label'] = portfolio_returns.rolling(rolling_window).apply(
    lambda x: np.quantile(x, 1 - confidence_level), raw=True
).shift(-1)

# Drop NaN values caused by rolling calculations
features = features.dropna()

# Step 5: Train-Test Split
X = features.drop(['VaR_Label'], axis=1)
y = features['VaR_Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the Machine Learning Model
var_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
var_model.fit(X_train, y_train)

# Predict VaR on the test data
y_pred = var_model.predict(X_test)

# Step 7: Calculate RMSE with the new root_mean_squared_error function
rmse = root_mean_squared_error(y_test, y_pred)  # Using root_mean_squared_error function directly
print(f"Root Mean Squared Error: {rmse}")

# Step 8: Calculate Portfolio ES (Expected Shortfall)
def calculate_es(returns, var):
    tail_losses = returns[returns <= var]
    return tail_losses.mean()

# Calculate ES for each predicted VaR
es_predictions = [calculate_es(portfolio_returns, var) for var in y_pred]
# Step 9: Visualization
plt.figure(figsize=(10, 6))
# Histogram of portfolio returns
plt.hist(portfolio_returns, bins=50, alpha=0.5, label="Portfolio Returns")
plt.axvline(x=np.mean(y_pred), color='r', linestyle='--', label='Predicted VaR (99%)')
plt.axvline(x=np.mean(es_predictions), color='g', linestyle='-', label='Predicted ES (99%)')
plt.legend()
plt.title("Portfolio Loss Distribution with Predicted VaR and ES")
plt.xlabel("Portfolio Returns")
plt.ylabel("Frequency")
plt.show()
