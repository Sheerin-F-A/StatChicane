from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
import datetime
import google.generativeai as genai
from urllib.parse import quote_plus
import time

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=st.secrets["GEMINI"]["API_KEY"])

# Function to call Gemini model
def call_gemini_model(prompt_str: str, model_name="models/gemini-1.5-flash"):
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt_str)
    try:
        return response.text
    except AttributeError:
        return response.candidates[0].content.parts[0].text


def check_credentials(username: str, password: str, db: SQLDatabase) -> bool:
    query = f"SELECT Password FROM User WHERE Username = '{username}'"
    try:
        result = db.run(query)

        if isinstance(result, str):
            import ast
            result = ast.literal_eval(result)

        if result and isinstance(result, (list, tuple)) and isinstance(result[0], (list, tuple)):
            stored_password = result[0][0]
            return stored_password.strip() == password.strip()
    except Exception as e:
        print("Error during credential check:", e)
    return False

def user_exists(username: str, db: SQLDatabase) -> bool:
    result = db.run(f"SELECT * FROM User WHERE Username = '{username}'")
    return len(result) > 0

# Helper functions to check if email or phone exists
def email_exists(email: str, db: SQLDatabase) -> bool:
    result = db.run(f"SELECT * FROM User WHERE Email = '{email}'")
    return len(result) > 0

def phone_exists(phone: str, db: SQLDatabase) -> bool:
    result = db.run(f"SELECT * FROM User WHERE Phone_No = '{phone}'")
    return len(result) > 0

def insert_user(user_data: dict, db: SQLDatabase) -> bool:
    try:
        values = (
            user_data["User_ID"].replace("'", "''"),
            user_data["First_Name"].replace("'", "''"),
            user_data["Last_Name"].replace("'", "''"),
            user_data["DOB"],
            user_data["Phone_No"].replace("'", "''"),
            user_data["Email"].replace("'", "''"),
            user_data["Username"].replace("'", "''"),
            user_data["Password"].replace("'", "''")
        )
        sql = (
            "INSERT INTO User (User_ID, First_Name, Last_Name, DOB, Phone_No, Email, Username, Password) "
            "VALUES ('{0}', '{1}', '{2}', '{3}', '{4}', '{5}', '{6}', '{7}')"
        ).format(*values)
        db.run(sql)
        return True
    except Exception as e:
        print("Error inserting user:", e)
        return False

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

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

st.set_page_config(page_title="StatChicane", page_icon=":speech_balloon:")
st.title("StatChicane")

try:
    password = quote_plus("Sql@1234")
    db = SQLDatabase.from_uri(f"mysql+mysqlconnector://root:{password}@Sarthaks-MacBook-Air-10.local:3306/F1")
    st.session_state.db = db
except Exception as e:
    st.error(f"Database connection failed: {e}")

if not st.session_state.authenticated:
    auth_mode = st.sidebar.radio("Login / Sign Up", ["Login", "Sign Up"])

    with st.sidebar:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if auth_mode == "Sign Up":
            first_name = st.text_input("First Name")
            last_name = st.text_input("Last Name")
            dob = st.date_input("Date of Birth", min_value=datetime.date(1947, 1, 1), max_value=datetime.date.today())
            phone_no = st.text_input("Phone Number")
            email = st.text_input("Email")

            if st.button("Create Account"):
                user_id = f"U{str(abs(hash(username)))[0:6]}"
                user_data = {
                    "User_ID": user_id,
                    "First_Name": first_name,
                    "Last_Name": last_name,
                    "DOB": dob.strftime('%Y-%m-%d'),
                    "Phone_No": phone_no,
                    "Email": email,
                    "Username": username,
                    "Password": password,
                }
                if not user_exists(username, st.session_state.db):
                    if email_exists(email, st.session_state.db):
                        st.error("Email already exists.")
                    elif phone_exists(phone_no, st.session_state.db):
                        st.error("Phone number already exists.")
                    else:
                        if insert_user(user_data, st.session_state.db):
                            st.success("Account created. You can now login.")
                        else:
                            st.error("Failed to create account.")
                else:
                    st.error("Username already exists.")

        elif st.button("Login"):
            if check_credentials(username, password, st.session_state.db):
                st.success("Login successful.")
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.password = password
            else:
                st.error("Invalid credentials.")

    if not st.session_state.authenticated:
        st.info("Please login or signup from the side bar to use StatChicane.")

if st.session_state.authenticated:
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.success("You have been logged out.")
        time.sleep(1)
        st.rerun()
    if st.sidebar.button("Delete Account"):
        st.session_state.confirm_delete_visible = True

    if st.session_state.get("confirm_delete_visible"):
        st.subheader("Confirm Account Deletion")
        password_input = st.text_input("Re-enter your password to confirm", type="password")
        confirm_checkbox = st.checkbox("I understand this action is irreversible")

        if confirm_checkbox and st.button("Confirm Delete"):
            if password_input.strip() == st.session_state.password.strip():
                try:
                    st.session_state.db.run(f"DELETE FROM User WHERE Username = '{st.session_state.username}'")
                    st.success("Account deleted successfully.")
                    st.session_state.authenticated = False
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to delete account: {e}")
            else:
                st.error("Incorrect password.")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Ciao! I'm your assistant. Ask me anything about Formula 1."),
        ]

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
            try:
                response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
                st.markdown(response)
                st.session_state.chat_history.append(AIMessage(content=response))
            except Exception as e:
                st.error(f"Failed to generate response: {e}")
