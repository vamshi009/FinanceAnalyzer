import pandas as pd



INPUT_FILE = "synthetic_financial_dataset.csv"
OUTPUT_FILE = "transactions_with_anomalies.csv"
ANOMALY_MULTIPLIER = 2.0




def load_transactions(path):
    df = pd.read_csv(path)

    # Ensure amount is numeric
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    return df




def calculate_avg_amount(df):
    """
    Calculate average transaction amount
    per customer and transaction type
    """
    avg_df = (
        df
        .groupby(["customer_id", "transaction_type"])["amount"]
        .mean()
        .reset_index()
        .rename(columns={"amount": "avg_amount"})
    )

    return avg_df


# =========================
# Anomaly detection
# =========================

def flag_anomalies(df, avg_df):
    """
    Flag transactions where amount >= 2x average
    """
    df = df.merge(
        avg_df,
        on=["customer_id", "transaction_type"],
        how="left"
    )

    df["is_potential_anomaly"] = (
        df["amount"] >= ANOMALY_MULTIPLIER * df["avg_amount"]
    )

    return df



def main():
    df = load_transactions(INPUT_FILE)

    avg_df = calculate_avg_amount(df)

    df = flag_anomalies(df, avg_df)

    df.to_csv(OUTPUT_FILE, index=False)

    print("Anomaly detection completed.")
    print(f"Output saved to: {OUTPUT_FILE}")

    # Optional summary
    print("\nSummary:")
    print(
        df["is_potential_anomaly"]
        .value_counts()
        .rename("count")
    )


if __name__ == "__main__":
    main()
