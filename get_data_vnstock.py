import pandas as pd
import matplotlib.pyplot as plt
from vnstock import Quote

def getData(symbol = 'PLX', start_date = '2019-01-01', end_date = '2026-01-21'):
    # Get the data from vnstock
    quote = Quote(symbol=symbol, source='VCI')
    df = quote.history(start=start_date, end=end_date, interval='1D')
    
    csv_filename = f'{symbol}_from_{start_date}_to_{end_date}.csv'
    df.to_csv(csv_filename, index=False)
    print(f"Data saved to: {csv_filename}")
    
    # Visualizing
    df['MA50'] = df['close'].rolling(window=50).mean()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, 
                                gridspec_kw={'height_ratios': [3, 1]})

    # --- Top Chart: Price & MA ---
    ax1.set_title(f'{symbol} Stock Price Analysis (2023-2026)', fontsize=14, fontweight='bold')
    ax1.plot(df.index, df['close'], label='Close Price', color='#1f77b4', linewidth=1.5)
    ax1.plot(df.index, df['MA50'], label='MA50 (Trend Support)', color='#ff7f0e', linestyle='--', linewidth=1.5)
    ax1.set_ylabel('Price (VND)')
    ax1.legend(loc='upper left')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    # --- Bottom Chart: Volume ---
    ax2.bar(df.index, df['volume'], label='Volume', color='gray', alpha=0.6)
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper left')
    ax2.grid(True, which='both', linestyle='--', alpha=0.3)

    # Show the plot
    plt.tight_layout()
    plt.show()
    
getData()