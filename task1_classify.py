import pandas as pd
import re
from caculate_similarity import get_confidence_score

def classify_dispute(description):
    """
    Rule-based classification logic.
    Returns: (category, confidence, explanation)
    """
    desc_lower = description.lower()
    
    # Rule 1: Duplicate Charges
    if any(word in desc_lower for word in ['twice', 'double', 'duplicate', 'two times']):
        return 'DUPLICATE_CHARGE', get_confidence_score(description, 'DUPLICATE_CHARGE'), "Keywords found: twice/double/duplicate"
    
    # Rule 2: Fraud
    if any(word in desc_lower for word in ['unrecognized', 'unauthorized', 'stolen', 'did not make', 'suspicious']):
        return 'FRAUD', get_confidence_score(description, 'FRAUD'),"Keywords found: unrecognized/unauthorized/did not make"
    
    # Rule 3: Failed Transactions
    if any(word in desc_lower for word in ['failed', 'error', 'declined', 'deducted']):
        return 'FAILED_TRANSACTION', get_confidence_score(description, 'FAILED_TRANSACTION'), "Keywords found: failed/error/deducted"
    
    # Rule 4: Refund Pending
    if any(word in desc_lower for word in ['refund', 'waiting', 'pending', 'return']):
        return 'REFUND_PENDING',  get_confidence_score(description, 'REFUND_PENDING'), "Keywords found: refund/waiting"
    
    # Default
    return 'OTHERS', get_confidence_score(description, 'OTHERS'), "No specific keywords matched"

def main():
    input_file = 'disputes.csv'
    output_file = 'classified_disputes.csv'
    
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Please ensure the data file exists.")
        return

    print("Classifying disputes...")
    
    # Apply classification
    results = df['description'].apply(classify_dispute)
    
    # Unpack results into new columns
    df['predicted_category'] = [res[0] for res in results]
    df['confidence'] = [res[1] for res in results]
    df['explanation'] = [res[2] for res in results]
    
    # Select columns for output as per requirements
    #add additional c
    output_df = df[['dispute_id', 'description',  'predicted_category', 'confidence', 'explanation', 'amount', 'created_at', 'txn_type', 'channel']]
    
    output_df.to_csv(output_file, index=False)
    print(f"Classification complete. Results saved to {output_file}")
    print(output_df[['dispute_id', 'predicted_category']].to_string(index=False))

if __name__ == "__main__":
    main()