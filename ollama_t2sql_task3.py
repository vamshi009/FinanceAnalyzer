from ollama import chat

import pandas as pd
import sqlite3


prompt = '''
   The table is named 'resolutions'. Its description is as follows:
        Columns:
        - dispute_id: INTEGER
        - description: VARCHAR(16)
        - predicted_category: VARCHAR(16)
        - confidence: FLOAT
        - explanation: VARCHAR(16)
        - suggested_action: VARCHAR(16)
        - justification: VARCHAR(16)
        - created_at: VARCHAR(16)
        - txn_type: VARCHAR(16)
        - channel: VARCHAR(16)
        - amount: FLOAT


        predicted_category: can have values: DUPLICATE_CHARGE, FRAUD, FAILED_TRANSACTION , REFUND_PENDING, OTHERS

        suggested_action: can have values: Auto-refund, Mark as potential fraud, Ask for more info, Escalate to bank, Manual review

        txn_type can be UPI, NEFT, CARD

        channel  can be Mobile, Web, POS 

        generate SQL query for following question: 
        Instructions:
        1. JUST Return the SQL Query only

        2. There is only table called resolutions
        '''

correcting_prompt = prompt + " Please regenarte, The earlier query failed with following error message ,"

# 1. Load CSV into pandas
df = pd.read_csv("resolutions.csv")

# 2. Create in-memory SQLite database
conn = sqlite3.connect(":memory:")

# 3. Load DataFrame into SQL table
df.to_sql("resolutions", conn, index=False, if_exists="replace")

# 4. Write your SQL query
query_sample = """
SELECT client_name, MAX(amount) AS max_amount
FROM data
"""

# 5. Run SQL query

def answer_user_question(question, mode='default', error_message=''):

    try:
        if(mode == 'retry'):
            prompt = correcting_prompt + error_message
        else:
            prompt = prompt

        response = chat(
            model='yasserrmd/Text2SQL-1.5B',
            messages=[{'role': 'user', 'content': prompt + "question: " + question}],
        )
        print(response.message.content)

        query = response.message.content

        print("Query found to be ", query)

        result = pd.read_sql_query(query, conn)

        print("obtained results...")
        print(result)
    except Exception as e:
        if(mode=='default'):
            raise RetryException(str(e))
        elif(mode=="retry"):
            print("retrying failed")
            raise Exception(str(e))
    

class RetryException(Exception):
    pass



if(__name__ == "__main__"):
    #print("Try Statement like ")
    print("--- AI Dispute Assistant CLI ---")
    print("Try asking:")
    print("  - 'How many duplicate charges?'")
    print("  - 'List fraud disputes'")
    print("  - 'Break down disputes by type'")
    while(1):
        print("User> ", end="")
        text_input = input()
        try:
            answer_user_question(text_input)
        except RetryException as e:
            print("failed to answer question, retrying")
            answer_user_question(text_input, mode='retry', error_message=str(e))



