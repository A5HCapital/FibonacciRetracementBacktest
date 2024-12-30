import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import traceback  # For detailed error tracing

# Streamlit App Title
st.title("Fibonacci Retracement Backtesting Program")

# User Inputs: Stock Symbol, Date Range, Starting Balance, and Risk Percentage
st.sidebar.header("Input Parameters")
symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL, MSFT):", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", value=datetime.now())
starting_balance = st.sidebar.number_input("Starting Account Balance ($)", value=10000, min_value=1000, step=100)
risk_percent = st.sidebar.number_input("Percentage of Account for Each Trade (%)", value=5.0, min_value=1.0, max_value=100.0)

# Debug Info for Inputs
st.write("Debug Info:")
st.write(f"Symbol: {symbol}, Start Date: {start_date}, End Date: {end_date}")
st.write(f"Starting Balance: ${starting_balance}, Risk Percentage: {risk_percent}%")

# Fetch Historical Data
if st.button("Fetch Data"):
    try:
        # Convert dates to datetime for yfinance compatibility
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)

        st.write(f"Fetching data for {symbol} from {start_date} to {end_date}...")

        # Fetch data from Yahoo Finance
        data = yf.download(symbol, start=start_date, end=end_date)
        st.write("Data fetched successfully.")

        # Flatten MultiIndex columns if they exist
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]  # Use the first level of the MultiIndex
            st.write("Flattened MultiIndex columns:")
            st.write(data.columns)

        # Reset index to make Date a column
        data.reset_index(inplace=True)
        data.rename(columns={"Adj Close": "Close"}, inplace=True)

        # Validate Data
        if data.empty:
            st.error("No data found for the selected symbol and date range.")
            st.stop()

        # Ensure Required Columns Exist
        required_columns = ["High", "Low", "Close"]
        for col in required_columns:
            if col not in data.columns:
                st.error(f"Missing column: {col} in the fetched data.")
                st.stop()

        st.write("Data validation passed.")

        # Strategy Parameters
        swing_length = st.slider("Swing Length", min_value=5, max_value=50, value=10)

        # Validate Dataset Size
        st.write("Dataset size:", len(data))
        st.write("Swing length:", swing_length)
        if len(data) < swing_length:
            st.error("The dataset size is smaller than the rolling window (swing length). Please choose a smaller swing length or a larger dataset.")
            st.stop()

        st.write("Calculating swing highs, lows, and Fibonacci levels...")

        # Calculate Swing Highs and Lows
        data["SwingHigh"] = data["High"].rolling(window=swing_length, center=True).max()
        data["SwingLow"] = data["Low"].rolling(window=swing_length, center=True).min()

        # Validate SwingHigh and SwingLow
        st.write("Validating SwingHigh and SwingLow columns...")
        if data["SwingHigh"].isnull().all() or data["SwingLow"].isnull().all():
            st.error("SwingHigh and SwingLow contain only NaN values. Please adjust your swing length or dataset.")
            st.stop()

        # Debug: Check for NaN in SwingHigh and SwingLow
        st.write("SwingHigh and SwingLow columns after rolling calculations:")
        st.write(data[["SwingHigh", "SwingLow"]].head())

        # Fibonacci Levels Calculation
        data["Fib38"] = data["SwingHigh"] - (data["SwingHigh"] - data["SwingLow"]) * 0.382
        data["Fib50"] = data["SwingHigh"] - (data["SwingHigh"] - data["SwingLow"]) * 0.5
        data["Fib61"] = data["SwingHigh"] - (data["SwingHigh"] - data["SwingLow"]) * 0.618

        # Validate Fibonacci Levels
        st.write("Validating Fibonacci levels...")
        missing_fibs = data[["Fib38", "Fib50", "Fib61"]].isnull().all()
        if missing_fibs.any():
            st.error("Fibonacci levels contain only NaN values. Please adjust your swing length or dataset.")
            st.stop()

        # Debug: Check for NaN in Fibonacci levels
        st.write("Fibonacci level columns after calculations:")
        st.write(data[["Fib38", "Fib50", "Fib61"]].head())

        # Drop rows with NaN values in required columns
        st.write("Dropping rows with NaN values in Fib50, SwingHigh, and SwingLow...")
        data.dropna(subset=["Fib50", "SwingHigh", "SwingLow"], inplace=True)
        st.write("Rows with NaN values dropped. DataFrame preview:")
        st.write(data.head())

        # Backtesting Logic
        account_balance = starting_balance
        total_pnl = 0
        total_pnl_percent = 0
        transactions = []

        st.write("Starting backtesting...")

        for i in range(len(data)):
            # Entry Signal
            if (
                i > 0 and
                data.iloc[i]["Close"] > data.iloc[i]["Fib50"] and
                data.iloc[i - 1]["Close"] <= data.iloc[i]["Fib50"]
            ):
                # Calculate position size
                position_size = (risk_percent / 100) * account_balance
                shares = position_size / data.iloc[i]["Close"]
                entry_price = data.iloc[i]["Close"]
                entry_date = data.iloc[i]["Date"]

                # Debugging: Log entry signal
                st.write(f"Entry signal triggered on {entry_date} at {entry_price:.2f}")

                # Exit Signal: Sell when reaching Swing High
                for j in range(i + 1, len(data)):
                    if data.iloc[j]["High"] >= data.iloc[j]["SwingHigh"]:
                        exit_price = data.iloc[j]["SwingHigh"]
                        exit_date = data.iloc[j]["Date"]
                        pnl = (exit_price - entry_price) * shares
                        pnl_percent = (pnl / (entry_price * shares)) * 100

                        # Update account balance
                        account_balance += pnl
                        total_pnl += pnl
                        total_pnl_percent += pnl_percent

                        # Record transaction
                        transactions.append({
                            "Entry Date": entry_date,
                            "Exit Date": exit_date,
                            "Entry Price": entry_price,
                            "Exit Price": exit_price,
                            "Shares Bought": shares,
                            "Shares Sold": shares,
                            "P/L": pnl,
                            "P/L (%)": pnl_percent,
                            "Total P/L": total_pnl,
                            "Total P/L (%)": total_pnl_percent,
                            "Account Balance": account_balance,
                        })

                        # Debugging: Log exit signal
                        st.write(f"Exit signal triggered on {exit_date} at {exit_price:.2f}")
                        break

        # Handle Empty Transactions
        transactions_df = pd.DataFrame(transactions)
        if transactions_df.empty:
            st.warning("No transactions were executed based on the selected parameters.")
        else:
            # Display Transaction Table
            st.subheader("Transaction Details")
            st.dataframe(transactions_df)

            # Download CSV
            st.download_button(
                label="Download Transactions CSV",
                data=transactions_df.to_csv(index=False),
                file_name="transactions.csv",
                mime="text/csv",
            )

            # Plot P/L and Account Balance
            st.subheader("Performance Chart")
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # P/L over time
            ax1.plot(transactions_df["Entry Date"], transactions_df["Total P/L"], label="Total P/L", color="green")
            ax1.set_xlabel("Date")
            ax1.set_ylabel("Total P/L ($)", color="green")
            ax1.tick_params(axis="y", labelcolor="green")

            # Account balance over time
            ax2 = ax1.twinx()
            ax2.plot(transactions_df["Entry Date"], transactions_df["Account Balance"], label="Account Balance", color="blue")
            ax2.set_ylabel("Account Balance ($)", color="blue")
            ax2.tick_params(axis="y", labelcolor="blue")

            fig.tight_layout()
            st.pyplot(fig)

            # Summary
            st.subheader("Summary")
            st.write(f"Starting Balance: ${starting_balance:.2f}")
            st.write(f"Ending Balance: ${account_balance:.2f}")
            st.write(f"Total P/L: ${total_pnl:.2f}")
            st.write(f"Total P/L (%): {total_pnl_percent:.2f}%")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.write(traceback.format_exc())


