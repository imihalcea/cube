import streamlit as st
import datetime
import os
import psycopg2

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document


def log(message, elem):
    current_time = datetime.datetime.now()
    milliseconds = current_time.microsecond // 1000
    timestamp = current_time.strftime(
        "[%Y-%m-%d %H:%M:%S.{:03d}] ".format(milliseconds)
    )
    elem.text(message)


def check_input(question: str):
    if question == "":
        raise Exception("Please enter a question.")
    else:
        pass


_postgres_prompt = """\
You are a PostgreSQL expert. Your task is to create a syntactically correct PostgreSQL query that answers the user’s question while strictly following these rules:

1. **Allowed Columns Only**
   Use only the columns explicitly listed under **Columns Info**. Do not reference any columns that are not provided.

2. **Select Only Necessary Columns**
   Select only the columns needed to answer the user’s question. Never use `SELECT *`.


3. **Handling Aggregation**
   - **Non-Aggregated Queries**: If the question asks for non-aggregated data, list the needed columns directly (e.g., `SELECT status, count FROM orders`).
   - **Aggregated Queries**: If the question requires aggregation, use `GROUP BY` along with an appropriate aggregator function or `MEASURE()`. For example:
     ```sql
     SELECT status, SUM(count)
     FROM orders
     GROUP BY 1
     ```

4. **Row Limits**
   Unless the user explicitly asks for a specific number of rows, limit your results to `{top_k}` using `LIMIT {top_k}`.

5. **Aliases for Clarity**
   Provide clear, meaningful aliases for columns when needed. For example, use `users.count AS total_users_count` instead of just repeating `count`.

6. **No Valid Query?**
   If no valid query can be constructed under these rules, return `{no_answer_text}`.

7. **Context**
   - **Table Info**:
     {table_info}

   - **Columns Info**:
     {columns_info}

**Question**:
{input_question}

---

### Example of How It Works

- **User Question**: “What is the name of the youngest user?”
- **Answer Approach**:
  - We need the user’s name (which might be first and/or last name).
  - We must include a measure column. For instance, `users.count` could be selected to satisfy the “at least one measure” rule.
  - We do not need all columns—only those required for identifying the user and complying with the measure requirement.
  - Sort by `users.age` ascending to get the youngest.
  - Limit results to `{top_k}` unless otherwise specified.

A non-aggregated query (since we only want one user’s name, not grouped data) might look like:
```sql
SELECT
  users.first_name AS user_first_name,
  users.last_name AS user_last_name,
  users.count AS total_users_count
FROM users
ORDER BY users.age ASC
LIMIT {top_k};
```

If no measure column (e.g., `users.count`) exists or if the user’s question cannot be answered with the available columns, you would return `{no_answer_text}`.
"""

_answer_prompt_text = """
Respond in markdown format to the user's question given the information retrieved from the database.
Respond with same language as the user's question.
Respond with same language as the user's question. The user language is French or English.

**User's Question**:
{input_question}

**Retrieved Information**:
{retrieved_information}
"""

_table_answer_prompt = """
Respond with a table in a markdown format to the user's question given the information retrieved from the database.
Give meaningful column names and provide the data in a tabular MARKDOWN format.
Respond with same language as the user's question. If the user's question is in French, respond in French. If the user's question is in English, respond in English.

**User's Question**:
{input_question}

**Retrieved Information Data**:
```csv
{retrieved_information}
```
"""

TEXT_ANSWER_PROMPT = PromptTemplate(
    input_variables=["input_question", "retrieved_information"],
    template=_answer_prompt_text,
)

TABLE_ANSWER_PROMPT = PromptTemplate(
    input_variables=["input_question", "retrieved_information"],
    template=_table_answer_prompt,
)

PROMPT_POSTFIX = """\
Return the answer as a JSON object with the following format:

{
    "query": "",
    "filters": [{"column": \"\", "operator": \"\", "value": "\"\"}]
}
"""

CUBE_SQL_API_PROMPT = PromptTemplate(
    input_variables=[
        "input_question",
        "table_info",
        "columns_info",
        "top_k",
        "no_answer_text",
    ],
    template=_postgres_prompt,
)

_NO_ANSWER_TEXT = "I can't answer this question."


def call_sql_api(sql_query: str):
    load_dotenv()
    CONN_STR = os.environ["DATABASE_URL"]

    # Initializing Cube SQL API connection)
    connection = psycopg2.connect(CONN_STR)

    cursor = connection.cursor()
    try:
      cursor.execute(sql_query)
    except Exception as e:
      raise NotImplementedError(f"Error occurred, the question is to vague or the system in not able to answer it. Please try again with a more specific question.")

    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()

    cursor.close()
    connection.close()

    return columns, rows


def create_docs_from_values(columns_values, table_name, column_name):
    value_docs = []

    for column_value in columns_values:
        print(column_value)
        metadata = dict(
            table_name=table_name,
            column_name=column_name,
        )

        page_content = column_value
        value_docs.append(Document(page_content=page_content, metadata=metadata))

    return value_docs
