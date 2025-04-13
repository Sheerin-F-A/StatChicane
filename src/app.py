from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from urllib.parse import quote_plus
import streamlit as st
import os
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize database
def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    encoded_password = quote_plus(password)
    db_uri = f"mysql+mysqlconnector://{user}:{encoded_password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

# Function to call Gemini model
def call_gemini_model(prompt_str: str, model_name="models/gemini-1.5-flash"):
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt_str)
    try:
        return response.text
    except AttributeError:
        return response.candidates[0].content.parts[0].text

# Create SQL generation chain
def get_sql_chain(db):
    template = """
        You are a data analyst at a company which deals with Formula 1 stats. You are interacting with a user who is asking you questions about the company's database as well as Formula 1 stats.
        Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
 
        <SCHEMA>{schema}</SCHEMA>
 
        Conversation History: {chat_history}
 
        Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
 
        Important:
        - Do not assume any filtering criteria unless it is explicitly mentioned in the user's question.
        - Do not use a LIMIT clause unless the user specifically asks for a limited number of results (e.g., "top 3", "first 5").
        - Always return the full result set if not otherwise constrained.
        - Ensure all rows are included unless there is a condition stated.
 
        For example:
        Question: which Team have the highest win?
        SQL Query: SELECT * FROM Team WHERE Wins = (SELECT MAX(Wins) FROM Team);
        Question: Name all teams sorted by the most championship wins
        SQL Query: SELECT * FROM Team ORDER BY Championship_Wins DESC;
 
        Your turn:
 
        Question: {question}
        SQL Query:
    """
    prompt = ChatPromptTemplate.from_template(template)

    def get_schema(_):
        return db.get_table_info()

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | (lambda d: call_gemini_model(d.to_string()))
        | StrOutputParser()
    )

# Build full response chain
def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)

    template = """
        You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
        Based on the table schema below, question, sql query, and sql response, write a natural language response.
        <SCHEMA>{schema}</SCHEMA>

        Conversation History: {chat_history}
        SQL Query: <SQL>{query}</SQL>
        User question: {question}
        SQL Response: {response}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: (
                print("Generated SQL query:", vars["query"]) or db.run(vars["query"])
            ),
        )
        | prompt
        | (lambda d: call_gemini_model(d.to_string()))
        | StrOutputParser()
    )

    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })

# Streamlit UI setup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Ciao! I'm your assistant. Ask me anything about Formula 1."),
    ]

st.set_page_config(page_title="StatChicane", page_icon=":speech_balloon:")
st.title("StatChicane")

# Sidebar for DB connection
with st.sidebar:
    st.subheader("Settings")
    st.write("Your Formula 1 Stats Agent. Connect to database and start chatting.")

    st.text_input("Host", value="Sarthaks-MacBook-Air-6.local", key="Host")
    st.text_input("Port", value="3306", key="Port")
    st.text_input("User", value="root", key="User")
    st.text_input("Password", type="password", value="Sql@1234", key="Password")
    st.text_input("Database", value="F1", key="Database")

    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            try:
                db = init_database(
                    st.session_state["User"],
                    st.session_state["Password"],
                    st.session_state["Host"],
                    st.session_state["Port"],
                    st.session_state["Database"]
                )
                st.session_state.db = db
                st.success("Connected to database!")
            except Exception as e:
                st.error(f"Connection failed: {e}")

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

# Handle user input
user_query = st.chat_input("Ask Freely...")
if user_query and user_query.strip():
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        if "db" not in st.session_state:
            st.error("Please connect to the database first from the sidebar.")
        else:
            try:
                response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
                st.markdown(response)
                st.session_state.chat_history.append(AIMessage(content=response))
            except Exception as e:
                st.error(f"Failed to generate response: {e}")