import os
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Config
# =========================

OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# Load data
# =========================

def load_data(path):
    df = pd.read_csv(path)
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


# =========================
# Trend analysis functions
# =========================

def disputes_over_time(df, freq="D"):
    return (
        df
        .set_index("created_at")
        .resample(freq)
        .size()
        .rename("dispute_count")
    )


def category_trend(df, freq="M"):
    return (
        df
        .set_index("created_at")
        .groupby([pd.Grouper(freq=freq), "predicted_category"])
        .size()
        .unstack(fill_value=0)
    )


def confidence_distribution(df):
    return df["confidence"].describe()


def high_risk_disputes(df, threshold=0.8):
    return df[
        (df["predicted_category"] == "FRAUD") &
        (df["confidence"] >= threshold)
    ]


def amount_impact_by_category(df):
    return (
        df
        .groupby("predicted_category")["amount"]
        .sum()
        .sort_values(ascending=False)
    )


def action_distribution(df):
    return df["suggested_action"].value_counts()


def channel_analysis(df):
    return df["channel"].value_counts()


# =========================
# Plot helpers (SAVE ONLY)
# =========================

def save_series_plot(series, title, ylabel, filename):
    plt.figure()
    series.plot()
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Time")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()


def save_bar_plot(series, title, ylabel, filename):
    plt.figure()
    series.plot(kind="bar")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()


# =========================
# Main analysis pipeline
# =========================

def analyze_disputes(csv_path):
    df = load_data(csv_path)

    # Overall trend
    trend = disputes_over_time(df, freq="W")
    save_series_plot(
        trend,
        "Weekly Dispute Trend",
        "Dispute Count",
        "weekly_dispute_trend.png"
    )

    # Category-wise trend
    cat_trend = category_trend(df)
    plt.figure()
    cat_trend.plot()
    plt.title("Monthly Disputes by Category")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "monthly_category_trend.png"))
    plt.close()

    # Amount impact
    amount_by_cat = amount_impact_by_category(df)
    save_bar_plot(
        amount_by_cat,
        "Disputed Amount by Category",
        "Total Amount",
        "amount_by_category.png"
    )

    # Suggested actions
    actions = action_distribution(df)
    save_bar_plot(
        actions,
        "Suggested Action Distribution",
        "Count",
        "suggested_actions.png"
    )

    # Channel analysis
    channels = channel_analysis(df)
    save_bar_plot(
        channels,
        "Disputes by Channel",
        "Count",
        "channel_distribution.png"
    )

    # Console summaries (lightweight)
    print("\n=== Confidence Distribution ===")
    print(confidence_distribution(df))

    fraud_df = high_risk_disputes(df)
    print(f"\nHigh-confidence fraud cases: {len(fraud_df)}")

    print(f"\nPlots saved to: ./{OUTPUT_DIR}/")


# =========================
# Entry point
# =========================

if __name__ == "__main__":
    analyze_disputes("resolutions.csv")
