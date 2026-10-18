
# ============================================================
# PROJECT: GLAXO STOCK MARKET DATA ANALYSIS AND VISUALIZATION
# Tools: Python, Pandas, Matplotlib, Seaborn
# ============================================================

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

FILE_PATH = (
    r"C:\Users\Sanket B Mane\Downloads"
    r"\6a92af60da3205.60096137_GLAXO_ (1).csv"
)

OUTPUT_DIR = (
    r"C:\Users\Sanket B Mane\Downloads\GLAXO_Project_Output"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 11


# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

print("=" * 65)
print("       GLAXO STOCK MARKET ANALYSIS PROJECT")
print("=" * 65)

df = pd.read_csv(FILE_PATH)

# Clean column names
df.columns = df.columns.str.strip()

print("\nOriginal dataset shape:", df.shape)
print("\nDataset columns:")
print(df.columns.tolist())


# ------------------------------------------------------------
# 3. DATA CLEANING AND PREPROCESSING
# ------------------------------------------------------------

required_columns = [
    "Date", "Open", "High", "Low", "Last", "Close",
    "Total Trade Quantity", "Turnover (Lacs)"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in CSV: {missing_columns}\n"
        f"Available columns: {df.columns.tolist()}"
    )

# Parse dates in day-month-year format
df["Date"] = pd.to_datetime(
    df["Date"],
    format="%d-%m-%Y",
    errors="coerce"
)

# Convert stock data to numeric values
numeric_columns = [
    "Open", "High", "Low", "Last", "Close",
    "Total Trade Quantity", "Turnover (Lacs)"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove invalid dates and rows missing essential prices
df = df.dropna(subset=["Date", "Open", "High", "Low", "Close"])

# Remove duplicate dates and sort chronologically
df = df.drop_duplicates(subset=["Date"], keep="last")
df = df.sort_values("Date").reset_index(drop=True)

if df.empty:
    raise ValueError("No valid stock records remain after cleaning.")

# Save cleaned dataset
df.to_csv(
    os.path.join(OUTPUT_DIR, "glaxo_cleaned_data.csv"),
    index=False
)

print("\nCleaned dataset shape:", df.shape)
print("\nMissing values after cleaning:")
print(df.isnull().sum())


# ------------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("                 EXPLORATORY DATA ANALYSIS")
print("=" * 65)

print("\nFirst five records:")
print(df.head())

print("\nLast five records:")
print(df.tail())

print("\nStatistical summary:")
print(df[numeric_columns].describe().round(2))

print("\nDate range:")
print("Start date:", df["Date"].min().date())
print("End date:  ", df["Date"].max().date())


# ------------------------------------------------------------
# 5. FEATURE ENGINEERING
# ------------------------------------------------------------

# Daily price change in rupees
df["Daily Change"] = df["Close"].diff()

# Daily percentage return
df["Daily Return (%)"] = df["Close"].pct_change() * 100

# Moving averages
df["MA20"] = df["Close"].rolling(window=20).mean()
df["MA50"] = df["Close"].rolling(window=50).mean()
df["MA200"] = df["Close"].rolling(window=200).mean()

# Rolling annualized volatility using 252 trading days
df["Volatility (%)"] = (
    df["Daily Return (%)"].rolling(window=20).std()
    * np.sqrt(252)
)

# Daily trading range
df["Daily Range"] = df["High"] - df["Low"]

# Drawdown relative to previous peak
df["Running Peak"] = df["Close"].cummax()
df["Drawdown (%)"] = (
    df["Close"] / df["Running Peak"] - 1
) * 100

# Calendar features
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Month Name"] = df["Date"].dt.strftime("%b")


# ------------------------------------------------------------
# 6. KEY PERFORMANCE INDICATORS (KPIs)
# ------------------------------------------------------------

first_close = df["Close"].iloc[0]
last_close = df["Close"].iloc[-1]

absolute_change = last_close - first_close

percentage_change = (
    (last_close / first_close - 1) * 100
    if first_close != 0 else np.nan
)

number_of_days = len(df)

# Annualized compound growth rate
years = (
    (df["Date"].iloc[-1] - df["Date"].iloc[0]).days / 365.25
)

if years > 0 and first_close > 0 and last_close > 0:
    cagr = ((last_close / first_close) ** (1 / years) - 1) * 100
else:
    cagr = np.nan

average_close = df["Close"].mean()
highest_price = df["High"].max()
lowest_price = df["Low"].min()

highest_date = df.loc[df["High"].idxmax(), "Date"]
lowest_date = df.loc[df["Low"].idxmin(), "Date"]

average_volume = df["Total Trade Quantity"].mean()
total_volume = df["Total Trade Quantity"].sum()
average_turnover = df["Turnover (Lacs)"].mean()

daily_returns = df["Daily Return (%)"].dropna()

average_daily_return = daily_returns.mean()
daily_volatility = daily_returns.std()
annualized_volatility = daily_volatility * np.sqrt(252)

best_day = df.loc[df["Daily Return (%)"].idxmax()]
worst_day = df.loc[df["Daily Return (%)"].idxmin()]

max_drawdown = df["Drawdown (%)"].min()

kpis = {
    "Trading Days": number_of_days,
    "First Closing Price (Rs.)": first_close,
    "Latest Closing Price (Rs.)": last_close,
    "Absolute Price Change (Rs.)": absolute_change,
    "Price Change (%)": percentage_change,
    "CAGR (%)": cagr,
    "Average Closing Price (Rs.)": average_close,
    "Highest High (Rs.)": highest_price,
    "Highest High Date": highest_date.date(),
    "Lowest Low (Rs.)": lowest_price,
    "Lowest Low Date": lowest_date.date(),
    "Average Daily Volume": average_volume,
    "Total Trading Volume": total_volume,
    "Average Turnover (Lacs)": average_turnover,
    "Average Daily Return (%)": average_daily_return,
    "Daily Volatility (%)": daily_volatility,
    "Annualized Volatility (%)": annualized_volatility,
    "Maximum Drawdown (%)": max_drawdown,
}

print("\n" + "=" * 65)
print("                 BUSINESS KPI REPORT")
print("=" * 65)

for key, value in kpis.items():
    if isinstance(value, (float, np.floating)):
        print(f"{key:<35}: {value:,.2f}")
    else:
        print(f"{key:<35}: {value}")

print("\nBest daily return:")
print(
    f"{best_day['Date'].date()} : "
    f"{best_day['Daily Return (%)']:.2f}%"
)

print("\nWorst daily return:")
print(
    f"{worst_day['Date'].date()} : "
    f"{worst_day['Daily Return (%)']:.2f}%"
)

# Save KPI report
kpi_df = pd.DataFrame(
    list(kpis.items()),
    columns=["Metric", "Value"]
)

kpi_df.to_csv(
    os.path.join(OUTPUT_DIR, "glaxo_kpi_report.csv"),
    index=False
)


# ------------------------------------------------------------
# 7. VISUALIZATION 1: CLOSING PRICE TREND
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(
    df["Date"], df["Close"],
    color="navy", linewidth=1.5,
    label="Closing Price"
)

ax.set_title("GLAXO Historical Closing Stock Price")
ax.set_xlabel("Date")
ax.set_ylabel("Closing Price (Rs.)")
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "01_closing_price.png"),
    dpi=300
)
plt.show()


# ------------------------------------------------------------
# 8. VISUALIZATION 2: MOVING AVERAGES
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(
    df["Date"], df["Close"],
    label="Close", color="black", alpha=0.65
)
ax.plot(
    df["Date"], df["MA20"],
    label="20-Day MA", color="blue"
)
ax.plot(
    df["Date"], df["MA50"],
    label="50-Day MA", color="orange"
)
ax.plot(
    df["Date"], df["MA200"],
    label="200-Day MA", color="red"
)

ax.set_title("GLAXO Stock Price with Moving Averages")
ax.set_xlabel("Date")
ax.set_ylabel("Price (Rs.)")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "02_moving_averages.png"),
    dpi=300
)
plt.show()


# ------------------------------------------------------------
# 9. VISUALIZATION 3: DAILY RETURNS
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(
    df["Date"], df["Daily Return (%)"],
    color="teal", linewidth=0.8
)
ax.axhline(0, color="black", linewidth=1)

ax.set_title("GLAXO Daily Stock Returns")
ax.set_xlabel("Date")
ax.set_ylabel("Daily Return (%)")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "03_daily_returns.png"),
    dpi=300
)
plt.show()


# ------------------------------------------------------------
# 10. VISUALIZATION 4: TRADING VOLUME
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 5))

ax.bar(
    df["Date"], df["Total Trade Quantity"],
    color="steelblue", width=1.0
)

ax.set_title("GLAXO Daily Trading Volume")
ax.set_xlabel("Date")
ax.set_ylabel("Shares Traded")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "04_trading_volume.png"),
    dpi=300
)
plt.show()


# ------------------------------------------------------------
# 11. VISUALIZATION 5: MONTHLY RETURNS HEATMAP
# ------------------------------------------------------------

monthly_close = (
    df.set_index("Date")["Close"]
    .resample("ME")
    .last()
)

monthly_returns = monthly_close.pct_change() * 100

monthly_df = monthly_returns.rename(
    "Monthly Return (%)"
).reset_index()

monthly_df["Year"] = monthly_df["Date"].dt.year
monthly_df["Month"] = monthly_df["Date"].dt.month

monthly_pivot = monthly_df.pivot(
    index="Year",
    columns="Month",
    values="Monthly Return (%)"
)

# Use calendar month labels
monthly_pivot = monthly_pivot.reindex(columns=range(1, 13))
monthly_pivot.columns = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

plt.figure(figsize=(14, max(5, len(monthly_pivot) * 0.4)))

sns.heatmap(
    monthly_pivot,
    cmap="RdYlGn",
    center=0,
    annot=True,
    fmt=".1f",
    linewidths=0.3,
    cbar_kws={"label": "Monthly Return (%)"}
)

plt.title("GLAXO Monthly Returns Heatmap (%)")
plt.xlabel("Month")
plt.ylabel("Year")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "05_monthly_returns_heatmap.png"),
    dpi=300
)
plt.show()


# ------------------------------------------------------------
# 12. VISUALIZATION 6: DAILY RETURN DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

sns.histplot(
    daily_returns,
    bins=50,
    kde=True,
    color="purple"
)

plt.axvline(
    daily_returns.mean(),
    color="red",
    linestyle="--",
    label="Mean Daily Return"
)

plt.title("Distribution of GLAXO Daily Returns")
plt.xlabel("Daily Return (%)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "06_return_distribution.png"),
    dpi=300
)
plt.show()


# ------------------------------------------------------------
# 13. VISUALIZATION 7: DRAWDOWN
# ------------------------------------------------------------

plt.figure(figsize=(14, 5))

plt.fill_between(
    df["Date"],
    df["Drawdown (%)"],
    0,
    color="crimson",
    alpha=0.35
)

plt.plot(
    df["Date"], df["Drawdown (%)"],
    color="darkred", linewidth=0.8
)

plt.title("GLAXO Historical Drawdown")
plt.xlabel("Date")
plt.ylabel("Drawdown (%)")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "07_drawdown.png"),
    dpi=300
)
plt.show()


# ------------------------------------------------------------
# 14. VISUALIZATION 8: CORRELATION HEATMAP
# ------------------------------------------------------------

correlation_columns = [
    "Open", "High", "Low", "Last", "Close",
    "Total Trade Quantity", "Turnover (Lacs)"
]

correlation_matrix = df[correlation_columns].corr()

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    vmin=-1,
    vmax=1
)

plt.title("GLAXO Stock Variables Correlation Matrix")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "08_correlation_heatmap.png"),
    dpi=300
)
plt.show()


# ------------------------------------------------------------
# 15. YEARLY PERFORMANCE ANALYSIS
# ------------------------------------------------------------

yearly = df.groupby("Year").agg(
    First_Close=("Close", "first"),
    Last_Close=("Close", "last"),
    Average_Close=("Close", "mean"),
    Highest_High=("High", "max"),
    Lowest_Low=("Low", "min"),
    Total_Volume=("Total Trade Quantity", "sum"),
    Average_Turnover=("Turnover (Lacs)", "mean")
)

yearly["Annual Return (%)"] = (
    yearly["Last_Close"] / yearly["First_Close"] - 1
) * 100

print("\n" + "=" * 65)
print("                 YEARLY PERFORMANCE")
print("=" * 65)
print(yearly.round(2))

yearly.to_csv(
    os.path.join(OUTPUT_DIR, "glaxo_yearly_performance.csv")
)

# Plot yearly returns
plt.figure(figsize=(13, 6))

colors = [
    "seagreen" if value >= 0 else "indianred"
    for value in yearly["Annual Return (%)"]
]

yearly["Annual Return (%)"].plot(
    kind="bar", color=colors
)

plt.axhline(0, color="black", linewidth=0.8)
plt.title("GLAXO Annual Stock Returns")
plt.xlabel("Year")
plt.ylabel("Annual Return (%)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "09_yearly_returns.png"),
    dpi=300
)
plt.show()


# ------------------------------------------------------------
# 16. SAVE FINAL ANALYSIS DATA
# ------------------------------------------------------------

df.to_csv(
    os.path.join(OUTPUT_DIR, "glaxo_final_analysis.csv"),
    index=False
)

print("\n" + "=" * 65)
print("                 PROJECT COMPLETED")
print("=" * 65)

print("\nAll outputs saved to:")
print(OUTPUT_DIR)

print("\nGenerated outputs include:")
print("1. Cleaned stock dataset")
print("2. Final analysis dataset")
print("3. KPI report")
print("4. Yearly performance report")
print("5. Closing price chart")
print("6. Moving averages chart")
print("7. Daily returns chart")
print("8. Trading volume chart")
print("9. Monthly returns heatmap")
print("10. Daily return distribution")
print("11. Drawdown chart")
print("12. Correlation heatmap")
print("13. Annual returns chart")

print("\nThank you for running the GLAXO analysis project!")
