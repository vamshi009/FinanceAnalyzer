import pandas as pd

def suggest_resolution(row):
    """
    Rule-based resolution logic based on category and amount.
    Returns: (action, justification)
    """
    category = row['predicted_category']
    amount = row['amount']
    
    if category == 'DUPLICATE_CHARGE':
        if amount < 50:
            return 'Auto-refund', 'Low value duplicate charge, safe to auto-refund'
        else:
            return 'Manual review', 'High value duplicate charge, requires verification'
            
    elif category == 'FRAUD':
        return 'Mark as potential fraud', 'Fraud category detected, blocking card/account recommended'
        
    elif category == 'FAILED_TRANSACTION':
        return 'Ask for more info', 'Need to verify if transaction eventually settled'
        
    elif category == 'REFUND_PENDING':
        return 'Escalate to bank', 'Refund delays usually require bank intervention'
        
    else:
        return 'Manual review', 'Unclassified dispute type requires human agent'

def main():
    input_file = 'classified_disputes.csv'
    output_file = 'resolutions.csv'
    
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Please run Task 1 first.")
        return

    print("Generating resolutions...")
    
    # Apply resolution logic
    results = df.apply(suggest_resolution, axis=1)
    
    df['suggested_action'] = [res[0] for res in results]
    df['justification'] = [res[1] for res in results]
    
    # Prepare output
    output_df = df[['dispute_id', 'description', 'predicted_category', 'confidence', 'explanation','suggested_action', 'justification', 'created_at', 'amount', 'txn_type', 'channel']]
    
    output_df.to_csv(output_file, index=False)
    print(f"Resolutions generated. Results saved to {output_file}")
    print(output_df.to_string(index=False))

if __name__ == "__main__":
    main()