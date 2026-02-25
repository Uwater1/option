import nbformat as nbf

nb = nbf.v4.new_notebook()

code1 = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Import IRLS filter and enrich_data
from train_prod_model import filter_arbitrage_irls, enrich_data, IRLS_MIN_ABS_DEV, IRLS_C, IRLS_MIN_POINTS, IRLS_MAX_ITER, IRLS_L2_PENALTY
"""

code2 = """# Load options data
csv_file = "options_data/2026-02-20/aapl_20260320_calls_264_58.csv"
df = pd.read_csv(csv_file)
df = enrich_data(df)

# Filter criteria
if 'lastPrice' in df.columns:
    df = df[df['lastPrice'] > 0.01]
if 'volume' in df.columns:
    df = df[df['volume'] >= 10]
if 'impliedVolatility' in df.columns:
    df = df[(df['impliedVolatility'] > 2.0) & (df['impliedVolatility'] < 150.0)]
if 'daysToExpiration' in df.columns:
    df = df[df['daysToExpiration'] >= 0.9]

needed = {'strikePrice', 'underlyingPriceAtTrade', 'daysToExpiration', 'impliedVolatility'}
work = df.copy()
for c in needed:
    work[c] = pd.to_numeric(work[c], errors='coerce')
work = work.dropna(subset=list(needed))
work = work[work['daysToExpiration'] > 0]
work = work[work['underlyingPriceAtTrade'] > 0]
"""

code3 = """# Fit theoretical surface just like in filter_arbitrage_irls
m = np.log(work['strikePrice'] / work['underlyingPriceAtTrade']).values
t = np.sqrt(work['daysToExpiration']).values
y = work['impliedVolatility'].values

X = np.column_stack([
    np.ones(len(m)),   
    m,                  
    m ** 2,             
    t,                  
    t ** 2,             
    m * t,              
    m ** 2 * t,         
])

w = np.exp(-2.0 * np.abs(m))
beta = np.zeros(X.shape[1])

I_reg = np.eye(X.shape[1])
I_reg[0, 0] = 0

for _ in range(IRLS_MAX_ITER):
    W = np.diag(w)
    XtW = X.T @ W
    try:
        beta_new = np.linalg.solve(XtW @ X + IRLS_L2_PENALTY * I_reg, XtW @ y)
    except np.linalg.LinAlgError:
        print("Singular matrix!")
        break

    if np.max(np.abs(beta_new - beta)) < 1e-6:
        beta = beta_new
        break
    beta = beta_new

    residuals = y - X @ beta
    mad = np.median(np.abs(residuals))
    mad_scale = 1.4826 * mad if mad > 1e-10 else 1.0  
    r = residuals / mad_scale

    abs_r = np.abs(r)
    w = np.where(abs_r < IRLS_C, 1.0, IRLS_C / np.maximum(abs_r, 1e-10))

theoretical_iv = X @ beta
work['Theoretical_IV'] = theoretical_iv
"""

code4 = """# Plotting Real IV vs Theoretical IV Curve
work = work.sort_values(by='strikePrice')

plt.figure(figsize=(10, 6))
plt.scatter(work['strikePrice'], work['impliedVolatility'], label='Real IV (Data)', color='blue', alpha=0.6)
plt.plot(work['strikePrice'], work['Theoretical_IV'], label='Theoretical Option Curve (IRLS)', color='red', linewidth=2)

plt.title('Implied Volatility: Real vs Theoretical (IRLS surface in train_prod_model.py)')
plt.xlabel('Strike Price')
plt.ylabel('Implied Volatility')
plt.legend()
plt.grid(True)
plt.show()
"""

nb['cells'] = [
    nbf.v4.new_code_cell(code1),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_code_cell(code3),
    nbf.v4.new_code_cell(code4)
]

nbf.write(nb, 'display.ipynb')
print("Notebook generated successfully!")
