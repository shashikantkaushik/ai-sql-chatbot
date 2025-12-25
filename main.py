import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import re

from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_groq import ChatGroq

load_dotenv()

# Database connection
engine = create_engine("mysql+pymysql://root:@localhost/retail_sales_db")
db = SQLDatabase(engine, sample_rows_in_table_info=3)

# LLM setup
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    groq_api_key=os.environ["GROQ_API_KEY"],
)

chain = SQLDatabaseChain.from_llm(
    llm=llm,
    db=db,
    verbose=True,
    return_intermediate_steps=True,
)


def extract_sql_query(intermediate_steps):
    """Extract SQL query from intermediate steps or text"""
    if not intermediate_steps:
        return None

    # SQLDatabaseChain returns intermediate_steps as a list
    # The SQL query is usually at index 1
    for step in intermediate_steps:
        if isinstance(step, str) and 'SELECT' in step.upper():
            # Extract SQL from the string
            sql_match = re.search(r'SQLQuery:\s*(SELECT.*?);', step, re.IGNORECASE | re.DOTALL)
            if sql_match:
                return sql_match.group(1).strip() + ';'

            # Fallback: try to get any SELECT statement
            sql_match = re.search(r'(SELECT.*?);', step, re.IGNORECASE | re.DOTALL)
            if sql_match:
                return sql_match.group(1).strip() + ';'

    return None


# Execute question
def execute_query(question):
    try:
        result = chain(question)

        # Extract SQL from intermediate_steps
        intermediate = result.get("intermediate_steps", [])

        # Debug: show what we got
        st.write("Debug - Intermediate steps:", intermediate)

        sql_query = extract_sql_query(intermediate)

        if not sql_query:
            st.warning("No SQL generated")
            return None, None

        st.write(f"Debug - Extracted SQL: {sql_query}")

        # Execute SQL query using text() for safer execution
        with engine.connect() as conn:
            query_result = conn.execute(text(sql_query)).fetchall()

        # Convert to DataFrame
        if query_result:
            # Get column names from the result
            df = pd.DataFrame(query_result)

            # Try to get proper column names
            if hasattr(query_result[0], '_mapping'):
                df.columns = list(query_result[0]._mapping.keys())
        else:
            df = pd.DataFrame()
            st.info("Query executed successfully but returned no results.")

        return sql_query, df

    except Exception as e:
        st.error(f"An error occurred: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None


# Streamlit UI
st.title("🧠 Chat with SQL Database")
question = st.text_input("Enter your question:")

if st.button("Execute") and question:
    with st.spinner("Processing your question..."):
        sql_query, df = execute_query(question)

    if sql_query:
        st.subheader("Generated SQL Query")
        st.code(sql_query, language="sql")

        st.subheader("Query Result")
        if df is not None and not df.empty:
            st.dataframe(df)
            st.success(f"Retrieved {len(df)} row(s)")
        else:
            st.write("No results returned.")