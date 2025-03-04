import streamlit as st
import datetime
import os
import psycopg2

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document


def log(message):
    current_time = datetime.datetime.now()
    milliseconds = current_time.microsecond // 1000
    timestamp = current_time.strftime(
        "[%Y-%m-%d %H:%M:%S.{:03d}] ".format(milliseconds)
    )
    st.text(message)


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

3. **Always Include at Least One Measure**
   The columns marked as `(member_type: measure)` are numeric and must appear at least once in your query (either directly in a non-aggregated query or aggregated with a function if needed).

4. **Handling Aggregation**
   - **Non-Aggregated Queries**: If the question asks for non-aggregated data, list the needed columns directly (e.g., `SELECT status, count FROM orders`).
   - **Aggregated Queries**: If the question requires aggregation, use `GROUP BY` along with an appropriate aggregator function or `MEASURE()`. For example:
     ```sql
     SELECT status, SUM(count)
     FROM orders
     GROUP BY 1
     ```

5. **Row Limits**
   Unless the user explicitly asks for a specific number of rows, limit your results to `{top_k}` using `LIMIT {top_k}`.

6. **Aliases for Clarity**
   Provide clear, meaningful aliases for columns when needed. For example, use `users.count AS total_users_count` instead of just repeating `count`.

7. **No Valid Query?**
   If no valid query can be constructed under these rules, return `{no_answer_text}`.

8. **Context**
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
