import pandas as pd
from rapidfuzz import fuzz
from itertools import combinations
from caculate_similarity import  cosine_similarity_numpy, model

TRANSACTION_THRESHOLD = 0.6

def get_verbal_description_of_txn(txn):
    #transaction_id,customer_id,account_number,transaction_date,transaction_time,amount,transaction_type,merchant,description,status,channel,location

    return "The transaction id is " + txn["transaction_id"] + " of customer " + txn["customer_id"] + " with account number " + txn["account_number"] +\
    " transaction happend on " + txn["transaction_date"] + " at " + txn["transaction_time"] + \
    "amount is " + txn["amount"] + " transaction type is " + txn["transaction_type"] + \
    " merchant is " + txn["merchant"] + " description is " + txn["description"] + \
    " status is " + txn["status"] + " channel is " + txn["channel"] + " location is " + txn["location"]


def get_llm_decision(txn1, txn2):
    verbal_description_of_txn1 = get_verbal_description_of_txn(txn1)
    verbal_description_of_txn2 =  get_verbal_description_of_txn(txn2)

    prompt = " Decide if these two transaction are similar or not: return either True or False. " + verbal_description_of_txn1 + " and " + verbal_description_of_txn2

    response = chat(
            model='yasserrmd/Text2SQL-1.5B',
            messages=[{'role': 'user', 'content': prompt}],
        )
    verdict = response.message.content

    if(verdict == "True"):
        return True
    else:
        return False



def semantically_similar(txn1, txn2):

    verbal_description_of_txn1 = get_verbal_description_of_txn(txn1)
    verbal_description_of_txn2 =  get_verbal_description_of_txn(txn2)

    emb_of_txn1 = model.encode(verbal_description_of_txn1)
    emb_of_txn2 = model.encode(verbal_description_of_txn2)
    score =  cosine_similarity_numpy(emb_of_txn1, emb_of_txn2)

    if(score > TRANSACTION_THRESHOLD):
        return True
    else:
        return False



TIME_WINDOW_MINUTES = 5
FUZZY_THRESHOLD = 60

INPUT_FILE = "synthetic_financial_dataset.csv"
OUTPUT_FILE = "duplicate_transactions.csv"




def load_transactions(path):
    df = pd.read_csv(path)

    # Combine date + time into timestamp
    df["transaction_datetime"] = pd.to_datetime(
        df["transaction_date"] + " " + df["transaction_time"]
    )

    return df

#fuzzy logic

def is_fuzzy_duplicate(txn1, txn2):
    # Amount check
    if txn1["amount"] != txn2["amount"]:
        return False

    # Time window check
    time_diff = abs(
        (txn1["transaction_datetime"] - txn2["transaction_datetime"])
        .total_seconds()
    ) / 60

    if time_diff > TIME_WINDOW_MINUTES:
        return False

    # Fuzzy merchant similarity
    merchant_score = fuzz.token_set_ratio(
        str(txn1["merchant"]),
        str(txn2["merchant"])
    )

    # Fuzzy description similarity
    description_score = fuzz.token_set_ratio(
        str(txn1["description"]),
        str(txn2["description"])
    )

    return (
        merchant_score >= FUZZY_THRESHOLD or
        description_score >= FUZZY_THRESHOLD
    )



def detect_duplicates(df):
    duplicate_rows = []

    grouped = df.groupby(["customer_id", "account_number"])

    for (_, _), group in grouped:
        records = group.to_dict("records")

        for txn1, txn2 in combinations(records, 2):
            if is_fuzzy_duplicate(txn1, txn2) or semantically_similar(txn1, txn2) or get_llm_decision(txn1, txn2):
                duplicate_rows.append({
                    "transaction_id_1": txn1["transaction_id"],
                    "transaction_id_2": txn2["transaction_id"],
                    "customer_id": txn1["customer_id"],
                    "account_number": txn1["account_number"],
                    "amount": txn1["amount"],
                    "time_diff_minutes": round(
                        abs(
                            (txn1["transaction_datetime"] - txn2["transaction_datetime"])
                            .total_seconds()
                        ) / 60, 2
                    ),
                    "merchant_1": txn1["merchant"],
                    "merchant_2": txn2["merchant"],
                    "description_1": txn1["description"],
                    "description_2": txn2["description"]
                })

    return pd.DataFrame(duplicate_rows)



def mark_duplicates(df, duplicates_df):
    df["is_duplicate"] = False

    dup_ids = set(
        duplicates_df["transaction_id_1"]
    ).union(
        set(duplicates_df["transaction_id_2"])
    )

    df.loc[df["transaction_id"].isin(dup_ids), "is_duplicate"] = True
    return df


def main():
    df = load_transactions(INPUT_FILE)

    duplicates_df = detect_duplicates(df)

    if duplicates_df.empty:
        print("No duplicate transactions detected.")
    else:
        print(f"Detected {len(duplicates_df)} duplicate pairs.")

    df = mark_duplicates(df, duplicates_df)

    # Save outputs
    duplicates_df.to_csv("duplicate_pairs.csv", index=False)
    df.to_csv(OUTPUT_FILE, index=False)

    print("Saved:")
    print(" - duplicate_pairs.csv")
    print(f" - {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
