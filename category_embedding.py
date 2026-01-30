

import numpy as np
from sentence_transformers import SentenceTransformer

DUPLICATE_CHARGE = [
    "The user reports that the same payment was charged more than once.",
    "Money was debited twice for the same transaction.",
    "User says they were charged two times for a single payment.",
    "Duplicate transaction where the same amount was deducted again.",
    "The same UPI or card payment shows multiple debits."
]

FAILED_TRANSACTION = [
    "The user reports that the transaction failed or did not complete successfully.",
    "Payment failed but the amount may have been debited.",
    "Transaction was unsuccessful due to a technical or bank error.",
    "UPI or card payment failed or was declined.",
    "Money deducted but transaction status shows failed."
]

FRAUD = [
    "The user reports an unauthorized or suspicious transaction.",
    "User says they did not make this payment.",
    "Fraudulent transaction where someone else used the account or card.",
    "Unknown charge that the user does not recognize.",
    "User suspects misuse of their card, account, or UPI."
]

REFUND_PENDING = [
    "The user reports that a refund was initiated but not yet received.",
    "Refund is pending or still processing.",
    "User is waiting for a refund after cancellation or failed transaction.",
    "Refund amount has not been credited yet.",
    "User is asking about delayed refund status."
]

OTHERS = [
    "The issue does not clearly match duplicate charge, failed transaction, fraud, or refund pending.",
    "User has a general or unclear payment-related query.",
    "The problem description is vague or missing important details.",
    "General support request without specific transaction issue.",
    "Issue unrelated to payments, charges, or refunds."
]


MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)



def get_class_embedding(text_list):
    """
    Compute the mean normalized embedding for a class
    """
    embeddings = model.encode(text_list, normalize_embeddings=True)
    return np.mean(embeddings, axis=0)


def build_class_embeddings():
    """
    Build embeddings for all classes
    """
    return {
        "DUPLICATE_CHARGE": get_class_embedding(DUPLICATE_CHARGE),
        "FAILED_TRANSACTION": get_class_embedding(FAILED_TRANSACTION),
        "FRAUD": get_class_embedding(FRAUD),
        "REFUND_PENDING": get_class_embedding(REFUND_PENDING),
        "OTHERS": get_class_embedding(OTHERS),
    }


CLASS_EMBEDDINGS = build_class_embeddings()

