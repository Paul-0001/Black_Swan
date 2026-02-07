"""
Black Swan Event Analysis: Comparing Normal vs Lévy Distribution
Analyses extreme market events occurances in S&P500 returns using different statistical distributions
"""
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
from scipy import stats
import pickle

# MAGIC NUMBERS
HISTORICAL_PERIOD = "10y"
N_EXTREME_VALUES = 5
N_POINTS_PDF = 1000
N_HISTOGRAM_BINS = 30
OUTPUT_DPI = 300
OUTPUT_FILENAME = 'black_swan.png'

def main():
    # Get S&P500 Data
    ticker = yf.Ticker('^GSPC')
    data = ticker.history(period=HISTORICAL_PERIOD)

    # Use log transformation to calculate returns
    log_rendite = np.log(data['Close'] / data['Close'].shift(1))

    # Sort the return data
    sortierte_log_rendite = np.sort(log_rendite)

    # Extract extreme values from the sorted dataset (Top 5 positive & negative values)
    negativen_Extremwerte = sortierte_log_rendite[:N_EXTREME_VALUES].tolist()
    positiven_Extremwerte = sortierte_log_rendite[-N_EXTREME_VALUES:].tolist()[::-1]

    # Load pre fitted Lévy Parameters for the S&P500 data
    with open('levy_ergebnisse.pkl', 'rb') as f:
        ergebnisse = pickle.load(f)

        log_rendite = ergebnisse['log_rendite']
        alpha = ergebnisse['alpha']
        beta = ergebnisse['beta']
        loc = ergebnisse['loc']
        scale = ergebnisse['scale']

    # Fit parameters of the normal distribution on the S&P500 data (Mü & Sigma)
    log_rendite_mu = log_rendite.mean()
    log_rendite_std = log_rendite.std()

    # Calculate expected frequency of negative extreme events under normal distribution
    for i in range(len(negativen_Extremwerte)):
        m = negativen_Extremwerte[i]
        p = stats.norm.cdf(m, loc = log_rendite_mu, scale=log_rendite_std)
        print(f"Using Normal distribution we expect to take {1/p:.2f} days for an {m:.6f} = {(np.exp(m)-1)*100:.6f}% return event to occur!")

    # Calculate expected frequency of negative extreme events under Lévy distribution
    for i in range(len(negativen_Extremwerte)):
        c = negativen_Extremwerte[i]
        d=stats.levy_stable.cdf(c, alpha, beta, loc, scale)
        print(f"Using Lévy distribution we expect to take {1/d:.2f} days for an {c:.6f} = {(np.exp(c) - 1)*100:.6f}% return event to occur!")

    # Calculate expected frequency of positive extreme events under normal distribution
    for i in range(len(positiven_Extremwerte)):
        m = positiven_Extremwerte[i] 
        q = stats.norm.cdf(m, loc=log_rendite_mu, scale=log_rendite_std)
        p = 1 - q
        print(f"Using Normal distribution we expect to take {1/p:.2f} days for an {m:.6f} = {(np.exp(m)-1)*100:.6f}% return event to occur!")

    # Calculate expected frequency of positive extreme events under Lévy distribution
    for i in range(len(positiven_Extremwerte)):
        c = positiven_Extremwerte[i]
        z = stats.levy_stable.cdf(c, alpha, beta, loc, scale)
        d = 1 - z
        print(f"Using Lévy distribution we expect to take {1/d:.2f} days for an {c:.6f} = {(np.exp(c)-1)*100:.6f}% return event to occur!")

    # Visualization: Compare distributions with extreme events
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Normal distribution with extreme events
    ax1 = axes[0]
    x = np.linspace(start=min(log_rendite), stop=max(log_rendite), num=N_POINTS_PDF, endpoint=True)
    x_pos = x[x>0]
    norm = stats.norm.pdf(x_pos, loc=log_rendite_mu, scale=log_rendite_std)
    levy = stats.levy_stable.pdf(x_pos, alpha, beta, loc, scale)

    ax1.plot(x_pos, norm, 'r-', linewidth=2, label='Normal Distribution')
    ax1.hist(log_rendite[log_rendite>0], bins = N_HISTOGRAM_BINS, density=True, alpha=0.5, log=True, label='Empirical Distribution')
    ax1.grid(alpha=0.3)

    # Mark extreme events
    for val in positiven_Extremwerte:
        ax1.axvline(val, color='purple', linestyle='--', linewidth=2)

    ax1.set_title('Black Swans using Normal Distribution')
    ax1.set_xscale('log')
    ax1.set_xlabel('log_return')
    ax1.set_ylabel('Density')
    ax1.legend()

    # Right plot: Lévy distribution with extreme events
    ax2 = axes[1]
    ax2.plot(x_pos, levy, 'r-', linewidth=2, label=f'Lévy Distribution with Alpha={alpha:.2f}')
    ax2.hist(log_rendite[log_rendite>0], bins = N_HISTOGRAM_BINS, density=True, alpha=0.5, log=True, label='Empirical Distribution')
    ax2.grid(alpha=0.3)

    # Synchronize y-axes
    ymin = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)

    ax2.set_title(f'Black Swans using Lévy Distribution with Alpha={alpha:.2f}')
    ax2.set_xscale('log')
    ax2.set_xlabel('log_return')
    ax2.set_ylabel('Density')
    ax2.legend()

    # Mark extreme events
    for val in positiven_Extremwerte:
        ax2.axvline(val, color='purple', linestyle='--', linewidth=2)

    # Display results
    plt.tight_layout()
    plt.savefig(OUTPUT_FILENAME, dpi=OUTPUT_DPI)
    plt.show()

if __name__ == "__main__":
    main()