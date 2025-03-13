import streamlit as st
import pandas as pd
import os
import re

from dotenv import load_dotenv
from langchain_openai import OpenAI, ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CubeSemanticLoader
from pathlib import Path

from utils import (
    check_input,
    log,
    call_sql_api,
    CUBE_SQL_API_PROMPT,
    TABLE_ANSWER_PROMPT,
    TEXT_ANSWER_PROMPT,
    _NO_ANSWER_TEXT,
)
st.set_page_config(layout="wide")
load_dotenv()

def ingest_cube_meta():
    token = os.environ["CUBE_API_SECRET"]

    loader = CubeSemanticLoader(os.environ["CUBE_API_URL"], token, load_dimension_values=True, dimension_values_limit=1000)
    documents = loader.load()
    log(f"Loaded {len(documents)} documents from Cube API", st)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    # Save vectorstore
    vectorstore.save_local("vectorstore.pkl")

if not Path("vectorstore.pkl").exists():
    with st.spinner('Loading context from Cube API...'):
        ingest_cube_meta()

llm = ChatOpenAI(
    openai_api_key=os.environ.get("OPENAI_API_KEY"), verbose=True, model_name="o1-mini"
)
st.image("img/logo_akawan_black.svg", caption=None, width=200, use_column_width=None, clamp=False, channels="RGB", output_format="auto", use_container_width=False)
col1, col2 = st.columns(2)
col1.title("Chat with the data 🤖")

multi = '''
You can use these sample questions to quickly test the demo --
* How many orders?
* How many completed orders?
* What are top selling product categories?
* What product category drives the highest average order value?
* Quelles sont les 10 villes avec le plus d'utilisateurs?
'''
col1.markdown(multi)


question = col1.text_input(
    "Your question: ", placeholder="Ask me anything ...", key="input"
)

if col1.button("Submit", type="primary"):
    check_input(question)
    if not Path("vectorstore.pkl").exists():
      col1.warning("vectorstore.pkl does not exist.")
    vectorstore = FAISS.load_local("vectorstore.pkl", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

    # log("Quering vectorstore and building the prompt...")

    docs = vectorstore.similarity_search(question)
    # take the first document as the best guess
    table_name = docs[0].metadata["table_name"]

    # Columns
    columns_question = "All available columns"
    column_docs = vectorstore.similarity_search(
        columns_question, filter=dict(table_name=table_name), k=15
    )

    lines = []
    for column_doc in column_docs:
        column_title = column_doc.metadata["column_title"]
        column_name = column_doc.metadata["column_name"]
        column_data_type = column_doc.metadata["column_data_type"]
        line = f"title: {column_title}, column name: {column_name}, datatype: {column_data_type}, member type: {column_doc.metadata['column_member_type']}"
        print(line)
        lines.append(line)
    columns = "\n\n".join(lines)

    # Construct the prompt
    prompt = CUBE_SQL_API_PROMPT.format(
        input_question=question,
        table_info=table_name,
        columns_info=columns,
        top_k=1000,
        no_answer_text=_NO_ANSWER_TEXT,
    )

    # Call LLM API to get the SQL query
    log("(AI) Generation de la requête :", col1)
    llm_answer = llm.invoke(prompt).text()

    if llm_answer.strip() == _NO_ANSWER_TEXT:
        log("(AI) Reponse : " + llm_answer, col1)
        st.stop()

    pattern = r"```sql\s*(.*?)\s*```"

    match = re.search(pattern, llm_answer, flags=re.DOTALL)

    if match:
      sql_query =  match.group(1).strip()
    else:
      st.stop()
    log("Requête SQL générée : ", col1)
    col1.info(sql_query)

    # Call Cube SQL API
    log("Recherche de l'information dans la base de données...", col1)
    columns, rows = call_sql_api(sql_query)

    # Display the result
    df = pd.DataFrame(rows, columns=columns, index=None)
    log("Formuler la reponse...", col1)
    if df.empty:
        col2.warning("Aucun résultat.")
    elif df.shape[0] > 1:
        prompt = TABLE_ANSWER_PROMPT.format(
            input_question=question,
            retrieved_information=df.to_csv(index=False)
        )
        llm_answer = llm.invoke(prompt).text()
        col2.markdown(llm_answer)
    else:
        prompt = TEXT_ANSWER_PROMPT.format(
            input_question=question,
            retrieved_information=df.to_csv(index=False)
        )
        llm_answer = llm.invoke(prompt).text()
        col2.markdown(llm_answer)
